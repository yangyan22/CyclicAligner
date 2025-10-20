import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import MimiccxrSingleImageDatasetOptimized


class R2DataLoaderOptimized(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle, drop_last):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split
        self.drop_last = drop_last

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.483, 0.483, 0.483),
                                     (0.235, 0.235, 0.235))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.483, 0.483, 0.483),
                                     (0.235, 0.235, 0.235))])

        if self.dataset_name == 'mimic_cxr':
            self.dataset = MimiccxrSingleImageDatasetOptimized(self.args, self.tokenizer, self.split, transform=self.transform)

        # Use limited workers when using precomputed features to balance performance and stability
        if getattr(self.args, 'use_precomputed_features', False):
            effective_num_workers = min(4, self.num_workers)  # Limit to 4 workers for HDF5 safety
        else:
            effective_num_workers = self.num_workers
        
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': effective_num_workers,
            'drop_last': self.drop_last,
            'pin_memory': True  # Pin memory for faster GPU transfer
        }
        
        # Add prefetch_factor only when using multiple workers
        if effective_num_workers > 0:
            self.init_kwargs['prefetch_factor'] = 2
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        (images_id, images, reports_ids, reports_masks, seq_lengths, 
         tok_ids, tok_masks, length_tok, imageData, ark_embeddings, ark_predictions) = zip(*data)
        
        images = torch.stack(images, 0)
        imageData = torch.stack(imageData, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        max_tok_length = max(length_tok)
        targets_tok = np.zeros((len(tok_ids), max_tok_length), dtype=int)
        targets_tok_masks = np.zeros((len(tok_ids), max_tok_length), dtype=int)

        for i, tok in enumerate(tok_ids):
            targets_tok[i, :len(tok)] = tok

        for i, tok_mask in enumerate(tok_masks):
            targets_tok_masks[i, :len(tok_mask)] = tok_mask

        # Handle precomputed features
        batch_ark_embeddings = None
        batch_ark_predictions = None
        
        if ark_embeddings[0] is not None:
            # Stack precomputed features
            batch_ark_embeddings = torch.stack(ark_embeddings, 0)
            batch_ark_predictions = torch.stack(ark_predictions, 0)

        return (images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks), 
                torch.LongTensor(targets_tok), torch.FloatTensor(targets_tok_masks), 
                imageData, batch_ark_embeddings, batch_ark_predictions)


# Original dataloader for backward compatibility
class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle, drop_last):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split
        self.drop_last = drop_last

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.483, 0.483, 0.483),
                                     (0.235, 0.235, 0.235))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.483, 0.483, 0.483),
                                     (0.235, 0.235, 0.235))])

        if self.dataset_name == 'mimic_cxr':
            from .datasets import MimiccxrSingleImageDataset
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'drop_last': self.drop_last
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths, tok_ids, tok_masks, length_tok, imageData = zip(*data)  
        
        images = torch.stack(images, 0)
        imageData = torch.stack(imageData, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        max_tok_length = max(length_tok)
        targets_tok = np.zeros((len(tok_ids), max_tok_length), dtype=int)
        targets_tok_masks = np.zeros((len(tok_ids), max_tok_length), dtype=int)

        for i, tok in enumerate(tok_ids):
            targets_tok[i, :len(tok)] = tok

        for i, tok_mask in enumerate(tok_masks):
            targets_tok_masks[i, :len(tok_mask)] = tok_mask

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks), torch.LongTensor(targets_tok), torch.FloatTensor(targets_tok_masks), imageData 