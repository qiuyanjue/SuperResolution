import argparse
from args.utils import str2bool


def parse_kitti_args():
    """
    Parse arguments for KITTI point cloud upsampling
    
    Supports both 2x and 4x upsampling:
        - 2x: 2048 points -> 4096 points
        - 4x: 1024 points -> 4096 points
    """
    parser = argparse.ArgumentParser(description='KITTI Point Cloud Upsampling Arguments')
    
    # ========== Random Seed ==========
    parser.add_argument('--seed', default=42, type=int, 
                       help='random seed for reproducibility')
    
    # ========== Dataset Configuration ==========
    parser.add_argument('--dataset', default='kitti', type=str, 
                       help='dataset name')
    parser.add_argument('--h5_data_path', default='./data/split_64_patch_sparse.h5', type=str,
                       help='h5 file path template (split and sparse will be replaced)')
    parser.add_argument('--up_rate', default=2, type=int, choices=[2, 4],
                       help='upsampling rate: 2x (2048->4096) or 4x (1024->4096)')
    parser.add_argument('--use_random_input', default=True, type=str2bool,
                       help='whether use random sampling for input generation')
    parser.add_argument('--jitter_sigma', type=float, default=0.01,
                       help='sigma for jitter augmentation')
    parser.add_argument('--jitter_max', type=float, default=0.03,
                       help='max clip value for jitter augmentation')
    
    # ========== Training Configuration ==========
    parser.add_argument('--epochs', default=150, type=int,
                       help='number of training epochs')
    parser.add_argument('--batch_size', default=16, type=int,
                       help='batch size for training (RTX 4090 can handle 16-24)')
    parser.add_argument('--num_workers', default=4, type=int,
                       help='number of workers for data loading')
    parser.add_argument('--print_rate', default=100, type=int,
                       help='print loss frequency (iterations)')
    parser.add_argument('--save_rate', default=10, type=int,
                       help='model checkpoint save frequency (epochs)')
    
    # ========== Optimizer Configuration ==========
    parser.add_argument('--optim', default='adam', type=str, choices=['adam', 'sgd'],
                       help='optimizer type')
    parser.add_argument('--lr', default=1e-3, type=float,
                       help='initial learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                       help='weight decay for optimizer')
    
    # ========== Learning Rate Scheduler ==========
    parser.add_argument('--lr_decay_step', default=50, type=int,
                       help='learning rate decay step size (epochs)')
    parser.add_argument('--gamma', default=0.5, type=float,
                       help='learning rate decay factor')
    
    # ========== Model Configuration ==========
    parser.add_argument('--k', default=16, type=int,
                       help='number of nearest neighbors for feature extraction')
    parser.add_argument('--block_num', default=3, type=int,
                       help='number of dense blocks in feature extractor')
    parser.add_argument('--layer_num', default=3, type=int,
                       help='number of dense layers in each dense block')
    parser.add_argument('--feat_dim', default=32, type=int,
                       help='input/output feature dimension in each dense block')
    parser.add_argument('--bn_size', default=1, type=int,
                       help='bottleneck size factor in dense layers')
    parser.add_argument('--growth_rate', default=32, type=int,
                       help='growth rate in dense layers')
    
    # ========== Query Points Configuration ==========
    parser.add_argument('--local_sigma', default=0.02, type=float,
                       help='sigma for sampling query points around interpolated points')
    
    # ========== Output Configuration ==========
    parser.add_argument('--out_path', default='./output', type=str,
                       help='output path for checkpoints and logs')
    
    # ========== Test Configuration ==========
    parser.add_argument('--num_iterations', default=10, type=int,
                       help='number of refinement iterations during testing')
    parser.add_argument('--test_step_size', default=50, type=float,
                       help='step size for gradient-based refinement during testing')
    parser.add_argument('--ckpt_path', default='', type=str,
                       help='path to pretrained checkpoint for testing')
    parser.add_argument('--test_input_path', default='./data/test_64_patch_2048.h5', type=str,
                       help='path to test input data')
    parser.add_argument('--save_dir', default='results', type=str,
                       help='directory to save test results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Automatically set num_points based on up_rate
    if args.up_rate == 2:
        args.num_points = 2048
    elif args.up_rate == 4:
        args.num_points = 1024
    else:
        raise ValueError(f"Unsupported up_rate: {args.up_rate}")
    
    # Print configuration summary
    print("\n" + "="*60)
    print("KITTI Point Cloud Upsampling Configuration")
    print("="*60)
    print(f"Task: {args.num_points} -> {args.num_points * args.up_rate} points ({args.up_rate}x upsampling)")
    print(f"Batch size: {args.batch_size} | Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr} | LR decay: every {args.lr_decay_step} epochs (×{args.gamma})")
    print(f"Optimizer: {args.optim} | Loss: Chamfer Distance")
    print(f"Output: {args.out_path}")
    print("="*60 + "\n")
    
    return args


if __name__ == '__main__':
    """Test argument parsing"""
    args = parse_kitti_args()
    
    print("\n" + "="*60)
    print("Full Configuration (for debugging)")
    print("="*60)
    for arg, value in sorted(vars(args).items()):
        print(f"{arg:25s}: {value}")
    print("="*60 + "\n")