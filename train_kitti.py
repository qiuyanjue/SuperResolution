import os
import sys
import time
import torch
import torch.optim as optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Add current directory to path
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


def create_dataloader(args, split='train'):
    """Create dataloader for given split"""
    if args.up_rate == 2:
        sparse_num = 2048
    elif args.up_rate == 4:
        sparse_num = 1024
    else:
        raise ValueError(f"Unsupported up_rate: {args.up_rate}")
    
    dense_num = 4096
    
    # Construct file paths
    sparse_h5 = f'./data/{split}_64_patch_{sparse_num}.h5'
    dense_h5 = f'./data/{split}_64_patch_{dense_num}.h5'
    
    # Check if files exist
    if not os.path.exists(sparse_h5):
        raise FileNotFoundError(f"Sparse h5 file not found: {sparse_h5}")
    if not os.path.exists(dense_h5):
        raise FileNotFoundError(f"Dense h5 file not found: {dense_h5}")
    
    # Determine augmentation
    augment = (split == 'train')
    
    # Create dataset
    dataset = KITTIPatchDataset(
        sparse_h5_path=sparse_h5,
        dense_h5_path=dense_h5,
        up_rate=args.up_rate,
        normalize=True,
        augment=augment,
        use_random_input=args.use_random_input,
        jitter_sigma=args.jitter_sigma,
        jitter_max=args.jitter_max
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=(split == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


def train(args):
    """
    Training function for KITTI point cloud upsampling
    
    Args:
        args: parsed arguments from kitti_args.py
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    start_time = time.time()
    
    # ========== Load Data ==========
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    
    train_loader = create_dataloader(args, split='train')
    val_loader = create_dataloader(args, split='val')
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # ========== Setup Output Directories ==========
    str_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(args.out_path, f'kitti_{args.up_rate}x_{str_time}')
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    log_dir = os.path.join(output_dir, 'log')
    
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Logger
    logger = get_logger('train', log_dir)
    logger.info(f'Experiment ID: {str_time}')
    logger.info(f'Output directory: {output_dir}')
    
    # ========== Build Model ==========
    print("\n" + "="*60)
    print("Building Model")
    print("="*60)
    
    model = P2PNet(args)
    model = model.cuda()
    
    # Count parameters
    para_num = sum([p.numel() for p in model.parameters()])
    logger.info(f"Model parameters: {para_num/1e6:.2f}M")
    
    # Log configuration
    logger.info("\n" + "="*60)
    logger.info("Configuration")
    logger.info("="*60)
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg:25s}: {value}")
    logger.info("="*60 + "\n")
    
    # ========== Setup Optimizer ==========
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)
    
    # ========== Training Loop ==========
    logger.info("\n" + "="*60)
    logger.info("Start Training")
    logger.info("="*60)
    
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # ========== Train One Epoch ==========
        model.train()
        train_loss = 0.0
        
        for i, (input_pts, gt_pts, radius) in enumerate(train_loader):
            # Move to GPU
            # (B, N, 3) -> (B, 3, N)
            input_pts = rearrange(input_pts, 'b n c -> b c n').contiguous().float().cuda()
            gt_pts = rearrange(gt_pts, 'b n c -> b c n').contiguous().float().cuda()
            radius = radius.float().cuda()
            
            # Midpoint interpolation (generate candidate points)
            interpolate_pts = midpoint_interpolate(args, input_pts)
            
            # Generate query points (add local perturbation)
            query_pts = get_query_points(interpolate_pts, args)
            
            # Forward pass: predict point-to-surface distance
            p2p_dist = model(input_pts, query_pts)  # (B, 1, M)
            
            # Simplified approach: refine query points along the direction to input centroid
            # This is a simplified version - the original Grad-PU uses gradient descent
            
            # Get direction vectors from query points to nearest input points
            # For simplicity, we'll just use the predicted points directly
            # and let the model learn to predict the correct displacement
            
            # Method 1: Direct displacement (simplified)
            # query_pts: (B, 3, M)
            # p2p_dist: (B, 1, M)
            
            # Compute direction from query to nearest input point
            knn_pts = get_knn_pts(1, input_pts, query_pts)  # (B, 3, M, 1)
            direction = knn_pts.squeeze(-1) - query_pts  # (B, 3, M)
            direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-8)
            
            # Move query points along the direction
            refined_pts = query_pts + p2p_dist * direction  # (B, 3, M)
            
            # Compute loss (Chamfer Distance)
            optimizer.zero_grad()
            loss = chamfer_distance(refined_pts, gt_pts)
            
            # Optional: Add repulsion loss to avoid point clustering
            # (Commented out for simplicity, can be enabled if needed)
            # repulsion = compute_repulsion_loss(refined_pts)
            # loss = loss + 0.01 * repulsion
            
            # Backward and optimize
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item()
            global_step += 1
            
            # Print progress
            if (i + 1) % args.print_rate == 0:
                avg_loss = train_loss / (i + 1)
                logger.info(f"Epoch [{epoch}/{args.epochs}] Iter [{i+1}/{len(train_loader)}] "
                          f"Loss: {loss.item():.6f} Avg Loss: {avg_loss:.6f}")
                writer.add_scalar('Train/loss_iter', loss.item(), global_step)
        
        # Average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # ========== Validation ==========
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for input_pts, gt_pts, radius in val_loader:
                # Move to GPU
                input_pts = rearrange(input_pts, 'b n c -> b c n').contiguous().float().cuda()
                gt_pts = rearrange(gt_pts, 'b n c -> b c n').contiguous().float().cuda()
                
                # Interpolation and query
                interpolate_pts = midpoint_interpolate(args, input_pts)
                query_pts = get_query_points(interpolate_pts, args)
                
                # Predict
                p2p_dist = model(input_pts, query_pts)
                
                # Refine
                knn_pts = get_knn_pts(1, input_pts, query_pts)
                direction = knn_pts.squeeze(-1) - query_pts
                direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-8)
                refined_pts = query_pts + p2p_dist * direction
                
                # Loss
                loss = chamfer_distance(refined_pts, gt_pts)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # ========== Logging ==========
        epoch_time = time.time() - epoch_start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch [{epoch}/{args.epochs}] Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Train Loss: {avg_train_loss:.6f}")
        logger.info(f"Val Loss:   {avg_val_loss:.6f}")
        logger.info(f"LR:         {scheduler.get_last_lr()[0]:.6f}")
        logger.info(f"Time:       {epoch_time:.2f}s")
        logger.info(f"{'='*60}\n")
        
        # Tensorboard
        writer.add_scalar('Train/loss_epoch', avg_train_loss, epoch)
        writer.add_scalar('Val/loss_epoch', avg_val_loss, epoch)
        writer.add_scalar('Train/lr', scheduler.get_last_lr()[0], epoch)
        
        # ========== Save Checkpoint ==========
        if epoch % args.save_rate == 0:
            ckpt_path = os.path.join(ckpt_dir, f'ckpt-epoch-{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'args': args
            }, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = os.path.join(ckpt_dir, 'ckpt-best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'args': args
            }, best_ckpt_path)
            logger.info(f"Best model saved: {best_ckpt_path} (Val Loss: {best_val_loss:.6f})")
        
        # Learning rate decay
        scheduler.step()
    
    # ========== Training Complete ==========
    total_time = time.time() - start_time
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Best val loss: {best_val_loss:.6f}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60 + "\n")
    
    writer.close()


if __name__ == '__main__':
    # Parse arguments
    args = parse_kitti_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This training script requires GPU.")
        sys.exit(1)
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    # Start training
    try:
        train(args)
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR occurred during training!")
        print(f"{'='*60}")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        sys.exit(1)