import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
import SimpleITK as sitk
import numpy as np
import h5py


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length + 1  # Length_bos == 1 + max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        
        # Quick test mode - limit dataset size
        self.quick_test = getattr(args, 'quick_test', False)
        self.max_samples_per_epoch = getattr(args, 'max_samples_per_epoch', 100)
        self.max_test_samples = getattr(args, 'max_test_samples', 100)  # New parameter for test set size
        
        if self.quick_test:
            if self.split == 'train':
                # Limit training data for quick testing
                max_samples = min(len(self.examples), self.max_samples_per_epoch)
                self.examples = self.examples[:max_samples]
                print(f"Quick test mode: Limited {self.split} dataset to {len(self.examples)} samples")
            elif self.split in ['val', 'test']:
                # Limit test/val data for quick testing
                max_samples = min(len(self.examples), self.max_test_samples)
                self.examples = self.examples[:max_samples]
                print(f"Quick test mode: Limited {self.split} dataset to {len(self.examples)} samples")

    def __len__(self):
        return len(self.examples)


class MimiccxrSingleImageDatasetOptimized(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super().__init__(args, tokenizer, split, transform)
        self.ark_features_path = getattr(args, 'ark_features_path', None)
        self.use_precomputed_features = getattr(args, 'use_precomputed_features', False)
        
        # Don't load HDF5 file in __init__ to avoid multiprocessing issues
        # Will be loaded in each worker process separately
        self.ark_features = None
        self._ark_features_loaded = False
        
        if self.use_precomputed_features and self.ark_features_path and os.path.exists(self.ark_features_path):
            print(f"Will load precomputed Ark+ features from {self.ark_features_path}")
        else:
            print("Will compute Ark+ features on-the-fly")
            
        # Precompute normalization values for Ark+ images
        self.ark_mean = np.array([0.485, 0.456, 0.406])
        self.ark_std = np.array([0.229, 0.224, 0.225])

    def _load_ark_features(self):
        """Load HDF5 features file in worker process"""
        if self.use_precomputed_features and self.ark_features_path and os.path.exists(self.ark_features_path) and not self._ark_features_loaded:
            try:
                # Use swmr (single writer multiple reader) mode for better multiprocessing support
                self.ark_features = h5py.File(self.ark_features_path, 'r', swmr=True)
                self._ark_features_loaded = True
                print(f"Loaded features for {len(self.ark_features.keys())} images in worker process")
            except Exception as e:
                print(f"Error loading HDF5 features: {e}")
                self.ark_features = None
                self._ark_features_loaded = False

    def __getitem__(self, idx):
        # Load ark features if not already loaded in this worker process
        if self.use_precomputed_features and not self._ark_features_loaded:
            self._load_ark_features()
            
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_paths']
        
        # Load regular image for ResNet
        if self.transform is not None:
            if len(image_path) == 1:
                image0 = self.transform(Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB'))
                image = image0
            elif len(image_path) == 2:
                image0 = self.transform(Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB'))
                image1 = self.transform(Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB'))
                image = (image0 + image1) / 2
            else:
                image0 = self.transform(Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB'))
                image1 = self.transform(Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB'))
                image2 = self.transform(Image.open(os.path.join(self.image_dir, image_path[2])).convert('RGB'))
                image = (image0 + image1 + image2) / 3

        # Handle Ark+ features
        ark_embedding = None
        ark_prediction = None
        imageData = None
        
        if self.ark_features is not None and image_id in self.ark_features:
            # Load precomputed features
            ark_embedding = torch.from_numpy(self.ark_features[image_id]['embedding'][:])
            ark_prediction = torch.from_numpy(self.ark_features[image_id]['prediction'][:])
            # Create dummy imageData tensor (won't be used)
            imageData = torch.zeros((3, 768, 768), dtype=torch.float32)
        else:
            # Compute features on-the-fly (original method)
            try:
                path_msg = os.path.join(self.image_dir, image_path[0])
                imageData_sitk = sitk.ReadImage(path_msg)
                image_array = sitk.GetArrayFromImage(imageData_sitk)
                imageData_pil = Image.fromarray(image_array).convert('RGB').resize((768, 768))
                image_ = np.array(imageData_pil) / 255.0
                image_ = (image_ - self.ark_mean) / self.ark_std
                image_ = image_.transpose(2, 0, 1).astype('float32')
                imageData = torch.from_numpy(image_)
            except Exception as e:
                print(f"Error loading image {image_path[0]} for Ark+: {e}")
                imageData = torch.zeros((3, 768, 768), dtype=torch.float32)

        # Process text data
        example['ids'] = self.tokenizer(example['report'])[:self.max_seq_length]
        example['mask'] = [1] * len(example['ids'])
        report_ids = example['ids']
        seq_length = len(report_ids)
        report_masks = example['mask']

        # Process tags
        token = ""
        for j in example['Tags']:
            token = token + j + " "
        token = token + ". "

        example['ids_tok'] = self.tokenizer(token + example['report'])[:self.max_seq_length + 10]
        example['mask_tok'] = [1] * len(example['ids_tok'])
        tok_ids = example['ids_tok']
        length_tok = len(example['ids_tok'])
        tok_masks = example['mask_tok']

        # Return data with optional precomputed features
        sample = (
            image_id, 
            image, 
            report_ids, 
            report_masks, 
            seq_length, 
            tok_ids, 
            tok_masks, 
            length_tok, 
            imageData,
            ark_embedding,
            ark_prediction
        )
        return sample

    def __del__(self):
        # Close HDF5 file when dataset is destroyed
        if hasattr(self, 'ark_features') and self.ark_features is not None:
            self.ark_features.close()


class MimiccxrSingleImageDataset(BaseDataset):
    """Original dataset class for backward compatibility"""
    def __getitem__(self, idx):
        example = self.examples[idx]
       
        image_id = example['id']
        image_path = example['image_paths']
        # Ark
        if self.transform is not None:
            path_msg = os.path.join(self.image_dir, image_path[0])
            imageData = sitk.ReadImage(path_msg) 
            image_array = sitk.GetArrayFromImage(imageData)
            imageData = Image.fromarray(image_array).convert('RGB').resize((768,768))
            image_ = np.array(imageData) / 255.
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            image_ = (image_ - mean)/std
            image_ = image_.transpose(2, 0, 1).astype('float32')
            image_ = torch.from_numpy(image_)

            
            if len(image_path)==1:
                image0 = self.transform(Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB'))
                image = image0
            elif len(image_path)==2:
                image0 = self.transform(Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB'))
                image1 = self.transform(Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB'))
                image = (image0 + image1)/2
            else:
                image0 = self.transform(Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB'))
                image1 = self.transform(Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB'))
                image2 = self.transform(Image.open(os.path.join(self.image_dir, image_path[2])).convert('RGB'))
                image = (image0 + image1 + image2)/3

        example['ids'] = self.tokenizer(example['report'])[:self.max_seq_length]
        example['mask'] = [1] * len(example['ids'])
        report_ids = example['ids']
        seq_length = len(report_ids)
        report_masks = example['mask']
        # random.shuffle(example['Tags'])
        #print(example['Tags'])

        token = ""
        for j in example['Tags']:
            token = token + j + " " 
        token = token + ". "

        example['ids_tok'] = self.tokenizer(token+example['report'])[:self.max_seq_length+10]
        example['mask_tok'] = [1] * len(example['ids_tok'])
        tok_ids = example['ids_tok']
        length_tok = len(example['ids_tok'])
        tok_masks = example['mask_tok']

        # print(image_id)
        # print(report_ids)
        # print(report_masks)
        # print(seq_length)
        # print(tok_ids)
        # print(tok_masks)
        # print(length_tok)

        sample = (image_id, image, report_ids, report_masks, seq_length, tok_ids, tok_masks, length_tok, image_)  # images_id, images, reports_ids, reports_masks, tok_ids
        return sample 