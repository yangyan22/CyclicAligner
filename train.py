import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoaderOptimized
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import TrainerOptimized
from modules.loss import compute_loss
from modules.models import Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='<mimic_data/files>',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='<mimic_all.json>',
                        help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr')
    parser.add_argument('--max_seq_length', type=int, default=100, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=10, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=4, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=32, help='the number of samples in a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet50', help='the visual extractor to be used.')  # resnet18 resnet50 resnet101 densenet121 pvt
    parser.add_argument('--pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')
    parser.add_argument('--tags', type=int, default=1, help='whether to concatenate the MeSH in report')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers_encoder', type=int, default=0, help='the number of layers of Transformer.')
    parser.add_argument('--num_layers_decoder', type=int, default=6, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the encoder output layer.')

    # Sample related  
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--epochs', type=int, default=36, help='the number of training epochs.')  # 4 startup + 8*4 = 36
    parser.add_argument('--save_dir', type=str, default='./results/release', help='the patch to save the models.')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=25, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='AdamW', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=2e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=20, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9, help='.')
    parser.add_argument('--resume', type=str, help='resume training from the checkpiont') # , default="/public/home/jw12138/YZQ/IU_Backbone/results/res50_TD12_IU_Y_BS64_900/current_checkpoint.pth", help='resume training from the checkpoint')
    parser.add_argument('--n_gpu', type=int, default=4, help='the number of gpus to be used.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--gpus', type=str, default='0, 1, 2, 3', help='GPU IDs')
    parser.add_argument('--gpus_id', type=list, default=[0, 1, 2, 3], help='GPU IDs')

    # Optimization settings
    parser.add_argument('--use_precomputed_features', type=bool, default=True, help='whether to use precomputed Ark+ features')
    parser.add_argument('--ark_features_path', type=str, default='./ark_features/ark_features_merged.h5', 
                        help='path to precomputed Ark+ features')

    # Quick test settings for error reproduction
    parser.add_argument('--quick_test', action='store_true', help='enable quick test mode with limited data and epochs')
    parser.add_argument('--max_samples_per_epoch', type=int, default=1000, help='maximum number of samples per epoch for quick testing')
    parser.add_argument('--max_test_samples', type=int, default=100, help='maximum number of samples for test/val sets in quick testing')

    args = parser.parse_args()
    
    # Adjust epochs for quick test mode
    if args.quick_test:
        args.epochs = 6  # 2 startup + 1*4 = 6 epochs for quick test
        
    return args


def main():
    # parse arguments
    args = parse_args()
    
    # Check if precomputed features exist
    import os
    if args.use_precomputed_features and not os.path.exists(args.ark_features_path):
        print(f"Warning: Precomputed features not found at {args.ark_features_path}")
        print("Falling back to on-the-fly computation")
        args.use_precomputed_features = False
    
    if args.use_precomputed_features:
        print(f"Using precomputed Ark+ features from {args.ark_features_path}")
    else:
        print("Using on-the-fly Ark+ feature computation")
    
    # Quick test mode
    if args.quick_test:
        print(f"Quick test mode enabled - limiting to {args.max_samples_per_epoch} samples per epoch")
        print(f"Test/val sets limited to {args.max_test_samples} samples")
        print(f"Running with 6 epochs (2 startup + 1 cycle with 1 epoch per stage)")
    
    # fix random seeds
    torch.manual_seed(args.seed)
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # build model architecture
    model = Model(args, tokenizer)

    # create data loader
    train_dataloader = R2DataLoaderOptimized(args, tokenizer, split='train', shuffle=True, drop_last=True)
    val_dataloader = R2DataLoaderOptimized(args, tokenizer, split='val', shuffle=False, drop_last=False)
    test_dataloader = R2DataLoaderOptimized(args, tokenizer, split='test', shuffle=False, drop_last=False)

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = TrainerOptimized(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main() 