import os
import json
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import SimpleITK as sitk
import argparse
from tqdm import tqdm
import pickle
import h5py
from timm.models.swin_transformer import SwinTransformer
import timm.models.swin_transformer as swin


class OmniSwinTransformer(swin.SwinTransformer):
    def __init__(self, num_classes_list, projector_features=None, use_mlp=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes_list is not None
        
        self.projector = None 
        if projector_features:
            encoder_features = self.num_features
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(nn.Linear(encoder_features, self.num_features), nn.ReLU(inplace=True), nn.Linear(self.num_features, self.num_features))
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)

        self.omni_heads = []
        for num_classes in num_classes_list:
            self.omni_heads.append(nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        self.omni_heads = nn.ModuleList(self.omni_heads)

    def forward(self, x, head_n=None):
        x = self.forward_features(x)
        if self.projector:
            x = self.projector(x)
        if head_n is not None:
            return x, self.omni_heads[head_n](x)
        else:
            return [head(x) for head in self.omni_heads]
    
    def generate_embeddings(self, x, after_proj=True):
        x = self.forward_features(x)
        if after_proj:
            x = self.projector(x)
        return x


class MimicImageDataset(Dataset):
    def __init__(self, image_dir, ann_path, split='train'):
        self.image_dir = image_dir
        self.ann_path = ann_path
        self.split = split
        
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)
        self.examples = self.ann[split]
        
        # Precompute normalization values
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_paths'][0]  # Use first image path
        
        try:
            # Load and preprocess image for Ark+
            full_path = os.path.join(self.image_dir, image_path)
            imageData = sitk.ReadImage(full_path)
            image_array = sitk.GetArrayFromImage(imageData)
            imageData = Image.fromarray(image_array).convert('RGB').resize((768, 768))
            image_ = np.array(imageData) / 255.0
            image_ = (image_ - self.mean) / self.std
            image_ = image_.transpose(2, 0, 1).astype('float32')
            image_ = torch.from_numpy(image_)
            
            return image_id, image_, image_path
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return dummy data in case of error
            dummy_image = torch.zeros((3, 768, 768), dtype=torch.float32)
            return image_id, dummy_image, image_path


def load_ark_model(device):
    """Load the Ark+ model"""
    model = OmniSwinTransformer(
        [14, 14, 14, 3, 6, 1], 
        projector_features=1376, 
        use_mlp=False, 
        img_size=768, 
        patch_size=4, 
        window_size=12, 
        embed_dim=192, 
        depths=(2, 2, 18, 2), 
        num_heads=(6, 12, 24, 48)
    )
    
    checkpoint = torch.load('Ark6_swinLarge768_ep50.pth.tar', map_location='cpu')
    state_dict = checkpoint['teacher']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    k_del = [k for k in state_dict.keys() if "attn_mask" in k] + ['head.weight', 'head.bias']
    print(f"Removing key(s) {k_del} from pretrained checkpoint for scaled input size")
    for k in k_del:
        if k in state_dict:
            del state_dict[k]
    
    load_result = model.load_state_dict(state_dict, strict=False)
    print(f"Model loading result: {load_result}")
    
    model = model.to(device)
    model.eval()
    return model


def extract_features_worker(gpu_id, image_ids, image_paths, image_dir, output_dir, start_idx, end_idx):
    """Worker function for extracting features on a specific GPU"""
    device = torch.device(f'cuda:{gpu_id}')
    
    # Load model on this GPU
    model = load_ark_model(device)
    
    # Create dataset for this worker's subset
    class WorkerDataset(Dataset):
        def __init__(self, image_ids, image_paths, image_dir, start_idx, end_idx):
            self.image_ids = image_ids[start_idx:end_idx]
            self.image_paths = image_paths[start_idx:end_idx]
            self.image_dir = image_dir
            self.mean = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])
            
        def __len__(self):
            return len(self.image_ids)
        
        def __getitem__(self, idx):
            image_id = self.image_ids[idx]
            image_path = self.image_paths[idx]
            
            try:
                full_path = os.path.join(self.image_dir, image_path)
                imageData = sitk.ReadImage(full_path)
                image_array = sitk.GetArrayFromImage(imageData)
                imageData = Image.fromarray(image_array).convert('RGB').resize((768, 768))
                image_ = np.array(imageData) / 255.0
                image_ = (image_ - self.mean) / self.std
                image_ = image_.transpose(2, 0, 1).astype('float32')
                image_ = torch.from_numpy(image_)
                
                return image_id, image_, image_path, True
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                dummy_image = torch.zeros((3, 768, 768), dtype=torch.float32)
                return image_id, dummy_image, image_path, False
    
    dataset = WorkerDataset(image_ids, image_paths, image_dir, start_idx, end_idx)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Create output file for this worker
    output_file = os.path.join(output_dir, f'ark_features_gpu_{gpu_id}.h5')
    
    with h5py.File(output_file, 'w') as f:
        with torch.no_grad():
            for batch_idx, (batch_image_ids, batch_images, batch_paths, batch_valid) in enumerate(tqdm(dataloader, desc=f'GPU {gpu_id}')):
                batch_images = batch_images.to(device)
                
                # Extract features
                try:
                    pre_logits = model(batch_images)
                    preds = [torch.sigmoid(out) for out in pre_logits]
                    predictions = torch.cat(preds, dim=1)
                    embeddings = model.generate_embeddings(batch_images)
                    
                    # Save features for each image in the batch
                    for i in range(len(batch_image_ids)):
                        if batch_valid[i]:  # Only save if image was loaded successfully
                            image_id = batch_image_ids[i]
                            embedding = embeddings[i].cpu().numpy()
                            prediction = predictions[i].cpu().numpy()
                            
                            # Create group for this image
                            img_group = f.create_group(image_id)
                            img_group.create_dataset('embedding', data=embedding)
                            img_group.create_dataset('prediction', data=prediction)
                            img_group.attrs['image_path'] = batch_paths[i]
                            
                except Exception as e:
                    print(f"Error processing batch {batch_idx} on GPU {gpu_id}: {e}")
                    continue
    
    print(f"GPU {gpu_id} finished processing {end_idx - start_idx} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='<mimic_data/files>',
                        help='Path to MIMIC image directory')
    parser.add_argument('--ann_path', type=str, default='mimic_all.json',
                        help='Path to annotation file')
    parser.add_argument('--output_dir', type=str, default='ark_features',
                        help='Directory to save extracted features')
    parser.add_argument('--n_gpu', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--test_mode', action='store_true', help='Test mode with small subset')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset info
    with open(args.ann_path, 'r') as f:
        ann = json.load(f)
    
    # Collect all image IDs and paths from all splits
    all_image_ids = []
    all_image_paths = []
    
    for split in ['train', 'val', 'test']:
        if split in ann:
            for example in ann[split]:
                image_id = example['id']
                image_path = example['image_paths'][0]  # Use first image path
                all_image_ids.append(image_id)
                all_image_paths.append(image_path)
    
    print(f"Total images to process: {len(all_image_ids)}")
    
    # Test mode: use only a small subset
    if args.test_mode:
        subset_size = min(100, len(all_image_ids))
        all_image_ids = all_image_ids[:subset_size]
        all_image_paths = all_image_paths[:subset_size]
        print(f"Test mode: processing only {subset_size} images")
    
    # Split work among GPUs
    images_per_gpu = len(all_image_ids) // args.n_gpu
    processes = []
    
    for gpu_id in range(args.n_gpu):
        start_idx = gpu_id * images_per_gpu
        if gpu_id == args.n_gpu - 1:  # Last GPU handles remaining images
            end_idx = len(all_image_ids)
        else:
            end_idx = (gpu_id + 1) * images_per_gpu
        
        print(f"GPU {gpu_id}: processing images {start_idx} to {end_idx-1}")
        
        p = mp.Process(target=extract_features_worker, 
                      args=(gpu_id, all_image_ids, all_image_paths, args.image_dir, args.output_dir, start_idx, end_idx))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("Feature extraction completed!")
    
    # Merge all feature files into one
    print("Merging feature files...")
    merged_file = os.path.join(args.output_dir, 'ark_features_merged.h5')
    
    with h5py.File(merged_file, 'w') as merged_f:
        for gpu_id in range(args.n_gpu):
            gpu_file = os.path.join(args.output_dir, f'ark_features_gpu_{gpu_id}.h5')
            if os.path.exists(gpu_file):
                with h5py.File(gpu_file, 'r') as gpu_f:
                    for image_id in gpu_f.keys():
                        # Copy the entire group
                        gpu_f.copy(image_id, merged_f)
                
                # Remove individual GPU file
                os.remove(gpu_file)
    
    print(f"All features merged into {merged_file}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main() 