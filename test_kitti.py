import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())

from dataset.kitti_dataset import KITTIPatchDataset
from models.P2PNet import P2PNet
from args.kitti_args import parse_kitti_args

# Always use PyTorch-native implementations for compatibility
print("="*60)
print("Using PyTorch-native implementations (no CUDA compilation needed)")
print("="*60)

from models.utils import (
    set_seed, get_logger, get_knn_pts, index_points,
    midpoint_interpolate, get_query_points, chamfer_distance,
    knn_point, square_distance
)

# Import rearrange
from einops import rearrange


def compute_metrics(pred_pts, gt_pts):
    """
    Compute evaluation metrics
    
    Args:
        pred_pts: (B, 3, N) predicted points
        gt_pts: (B, 3, N) ground truth points
    
    Returns:
        metrics: dict of metric values
    """
    metrics = {}
    
    # Chamfer Distance
    cd = chamfer_distance(pred_pts, gt_pts)
    metrics['CD'] = cd.item()
    
    # F-Score at different thresholds
    for threshold in [0.01, 0.02, 0.05, 0.10]:
        f_score = compute_f_score(pred_pts, gt_pts, threshold=threshold)
        metrics[f'F-Score@{threshold}'] = f_score
    
    return metrics


def compute_f_score(pred_pts, gt_pts, threshold=0.01):
    """
    Compute F-Score
    
    Args:
        pred_pts: (B, 3, N) predicted points
        gt_pts: (B, 3, M) ground truth points
        threshold: distance threshold
    
    Returns:
        f_score: F-Score value
    """
    # pred_pts: (B, 3, N), gt_pts: (B, 3, M)
    pred_pts = pred_pts.permute(0, 2, 1)  # (B, N, 3)
    gt_pts = gt_pts.permute(0, 2, 1)      # (B, M, 3)
    
    # Compute pairwise distances
    # dist: (B, N, M)
    dist = torch.cdist(pred_pts, gt_pts, p=2)
    
    # Precision: percentage of predicted points close to GT
    min_dist_pred = dist.min(dim=2)[0]  # (B, N)
    precision = (min_dist_pred < threshold).float().mean()
    
    # Recall: percentage of GT points close to predictions
    min_dist_gt = dist.min(dim=1)[0]  # (B, M)
    recall = (min_dist_gt < threshold).float().mean()
    
    # F-Score
    if precision + recall > 0:
        f_score = 2 * precision * recall / (precision + recall)
    else:
        f_score = 0.0
    
    return f_score.item()


def test(args):
    """
    Test function for KITTI point cloud upsampling
    
    Args:
        args: parsed arguments
    """
    print("\n" + "="*60)
    print("KITTI Point Cloud Upsampling - Testing")
    print("="*60)
    
    # ========== Load Model ==========
    print("\nLoading model...")
    model = P2PNet(args)
    model = model.cuda()
    
    # Load checkpoint
    if not os.path.exists(args.ckpt_path):
        print(f"ERROR: Checkpoint not found: {args.ckpt_path}")
        sys.exit(1)
    
    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from: {args.ckpt_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint val loss: {checkpoint.get('val_loss', 'N/A')}")
    
    # ========== Load Test Data ==========
    print("\nLoading test data...")
    
    # Determine file paths
    if args.up_rate == 2:
        sparse_num = 2048
    elif args.up_rate == 4:
        sparse_num = 1024
    else:
        raise ValueError(f"Unsupported up_rate: {args.up_rate}")
    
    dense_num = 4096
    
    sparse_h5_path = f'./data/test_64_patch_{sparse_num}.h5'
    dense_h5_path = f'./data/test_64_patch_{dense_num}.h5'
    
    test_dataset = KITTIPatchDataset(
        sparse_h5_path=sparse_h5_path,
        dense_h5_path=dense_h5_path,
        up_rate=args.up_rate,
        normalize=True,
        augment=False  # No augmentation for testing
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,  # Test one by one
        shuffle=False,
        num_workers=0
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # ========== Testing Loop ==========
    print("\nTesting...")
    
    all_metrics = {
        'CD': [],
        'F-Score@0.01': [],
        'F-Score@0.02': [],
        'F-Score@0.05': [],
        'F-Score@0.10': []
    }
    
    # Create output directory
    output_dir = os.path.join(args.save_dir, f'test_results_{args.up_rate}x')
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, (input_pts, gt_pts, radius) in enumerate(tqdm(test_loader)):
            # Move to GPU
            input_pts = rearrange(input_pts, 'b n c -> b c n').contiguous().float().cuda()
            gt_pts = rearrange(gt_pts, 'b n c -> b c n').contiguous().float().cuda()
            
            # Interpolation and query
            interpolate_pts = midpoint_interpolate(args, input_pts)
            query_pts = get_query_points(interpolate_pts, args)
            
            # Predict
            p2p_dist = model(input_pts, query_pts)
            
            # Refine points (same as training)
            knn_pts = get_knn_pts(1, input_pts, query_pts)
            direction = knn_pts.squeeze(-1) - query_pts
            direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-8)
            refined_pts = query_pts + p2p_dist * direction
            
            # Compute metrics
            metrics = compute_metrics(refined_pts, gt_pts)
            
            for key, value in metrics.items():
                all_metrics[key].append(value)
            
            # Optionally save point clouds
            if idx < 10:  # Save first 10 samples
                # Convert to numpy
                pred_np = refined_pts.squeeze(0).permute(1, 0).cpu().numpy()  # (N, 3)
                gt_np = gt_pts.squeeze(0).permute(1, 0).cpu().numpy()          # (M, 3)
                input_np = input_pts.squeeze(0).permute(1, 0).cpu().numpy()    # (K, 3)
                
                # Save as .npy files
                np.save(os.path.join(output_dir, f'sample_{idx:03d}_pred.npy'), pred_np)
                np.save(os.path.join(output_dir, f'sample_{idx:03d}_gt.npy'), gt_np)
                np.save(os.path.join(output_dir, f'sample_{idx:03d}_input.npy'), input_np)
    
    # ========== Print Results ==========
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    
    for key in all_metrics.keys():
        values = all_metrics[key]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{key:20s}: {mean_val:.6f} ± {std_val:.6f}")
    
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Test Results\n")
        f.write("="*60 + "\n")
        for key in all_metrics.keys():
            values = all_metrics[key]
            mean_val = np.mean(values)
            std_val = np.std(values)
            f.write(f"{key:20s}: {mean_val:.6f} ± {std_val:.6f}\n")
        f.write("="*60 + "\n")
    
    print(f"Metrics saved to: {metrics_file}")


if __name__ == '__main__':
    # Parse arguments
    args = parse_kitti_args()
    
    # Check for checkpoint path
    if not args.ckpt_path:
        print("ERROR: Please specify --ckpt_path for testing")
        print("Example: python test_kitti.py --ckpt_path ./output/xxx/ckpt/ckpt-best.pth --up_rate 2")
        sys.exit(1)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU (will be slow)")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Run testing
    try:
        test(args)
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR occurred during testing!")
        print(f"{'='*60}")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        sys.exit(1)