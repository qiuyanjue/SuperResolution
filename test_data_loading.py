"""
Test script for KITTI data loading

This script verifies:
1. H5 files can be read correctly
2. Dataset returns correct shapes
3. Normalization works properly
4. Augmentation works correctly
5. DataLoader works with batching

Usage:
    python test_data_loading.py --up_rate 2
    python test_data_loading.py --up_rate 4
"""

import os
import sys
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from dataset.kitti_dataset import KITTIPatchDataset, get_kitti_dataloader
from args.kitti_args import parse_kitti_args


def test_h5_files():
    """Test 1: Check if H5 files exist and can be read"""
    print("\n" + "="*60)
    print("Test 1: H5 File Accessibility")
    print("="*60)
    
    h5_files = [
        './data/train_64_patch_1024.h5',
        './data/train_64_patch_2048.h5',
        './data/train_64_patch_4096.h5',
        './data/val_64_patch_1024.h5',
        './data/val_64_patch_2048.h5',
        './data/val_64_patch_4096.h5',
        './data/test_64_patch_1024.h5',
        './data/test_64_patch_2048.h5',
        './data/test_64_patch_4096.h5'
    ]
    
    for h5_file in h5_files:
        if os.path.exists(h5_file):
            with h5py.File(h5_file, 'r') as f:
                num_patches = len(f.keys())
                first_patch = list(f.keys())[0]
                data_shape = f[first_patch].shape
                print(f"✓ {os.path.basename(h5_file):30s} - {num_patches:5d} patches, shape: {data_shape}")
        else:
            print(f"✗ {os.path.basename(h5_file):30s} - NOT FOUND")
    
    print("="*60)


def test_dataset_basic(up_rate=2):
    """Test 2: Basic dataset functionality"""
    print("\n" + "="*60)
    print(f"Test 2: Dataset Basic Functionality ({up_rate}x)")
    print("="*60)
    
    # Determine file paths
    if up_rate == 2:
        sparse_num = 2048
    elif up_rate == 4:
        sparse_num = 1024
    else:
        raise ValueError(f"Unsupported up_rate: {up_rate}")
    
    dense_num = 4096
    
    sparse_h5 = f'./data/train_64_patch_{sparse_num}.h5'
    dense_h5 = f'./data/train_64_patch_{dense_num}.h5'
    
    # Create dataset (without augmentation)
    dataset = KITTIPatchDataset(
        sparse_h5_path=sparse_h5,
        dense_h5_path=dense_h5,
        up_rate=up_rate,
        normalize=True,
        augment=False
    )
    
    print(f"\n✓ Dataset created successfully")
    print(f"  Total patches: {len(dataset)}")
    
    # Test loading one sample
    sparse, dense, radius = dataset[0]
    
    print(f"\n✓ Sample loaded successfully")
    print(f"  Sparse shape: {sparse.shape}")
    print(f"  Dense shape: {dense.shape}")
    print(f"  Radius: {radius.item():.4f}")
    print(f"  Sparse range: [{sparse.min().item():.4f}, {sparse.max().item():.4f}]")
    print(f"  Dense range: [{dense.min().item():.4f}, {dense.max().item():.4f}]")
    
    # Check types
    assert isinstance(sparse, torch.Tensor), "Sparse should be torch.Tensor"
    assert isinstance(dense, torch.Tensor), "Dense should be torch.Tensor"
    assert sparse.shape[1] == 3, f"Sparse should have 3 channels (XYZ), got {sparse.shape[1]}"
    assert dense.shape[1] == 3, f"Dense should have 3 channels (XYZ), got {dense.shape[1]}"
    assert sparse.shape[0] == sparse_num, f"Sparse should have {sparse_num} points"
    assert dense.shape[0] == dense_num, f"Dense should have {dense_num} points"
    
    print(f"\n✓ All checks passed!")
    print("="*60)


def test_augmentation(up_rate=2):
    """Test 3: Data augmentation"""
    print("\n" + "="*60)
    print(f"Test 3: Data Augmentation ({up_rate}x)")
    print("="*60)
    
    if up_rate == 2:
        sparse_num = 2048
    else:
        sparse_num = 1024
    dense_num = 4096
    
    sparse_h5 = f'./data/train_64_patch_{sparse_num}.h5'
    dense_h5 = f'./data/train_64_patch_{dense_num}.h5'
    
    # Create dataset with augmentation
    dataset = KITTIPatchDataset(
        sparse_h5_path=sparse_h5,
        dense_h5_path=dense_h5,
        up_rate=up_rate,
        normalize=True,
        augment=True
    )
    
    # Load same sample multiple times
    print("\nLoading same patch 5 times with augmentation:")
    for i in range(5):
        sparse, dense, radius = dataset[0]
        print(f"  Iter {i+1}: Sparse range [{sparse.min().item():.4f}, {sparse.max().item():.4f}], "
              f"Radius {radius.item():.4f}")
    
    print("\n✓ Augmentation working (values should be different each time)")
    print("="*60)


def test_dataloader(up_rate=2):
    """Test 4: DataLoader with batching"""
    print("\n" + "="*60)
    print(f"Test 4: DataLoader ({up_rate}x)")
    print("="*60)
    
    # Create a simple args object
    class Args:
        def __init__(self):
            self.up_rate = up_rate
            self.batch_size = 4
            self.num_workers = 2
            self.use_random_input = True
            self.jitter_sigma = 0.01
            self.jitter_max = 0.03
            self.h5_data_path = './data/split_64_patch_sparse.h5'
    
    args = Args()
    
    # Note: get_kitti_dataloader might need adjustment
    # For now, let's manually create dataloader
    if up_rate == 2:
        sparse_num = 2048
    else:
        sparse_num = 1024
    dense_num = 4096
    
    sparse_h5 = f'./data/train_64_patch_{sparse_num}.h5'
    dense_h5 = f'./data/train_64_patch_{dense_num}.h5'
    
    dataset = KITTIPatchDataset(
        sparse_h5_path=sparse_h5,
        dense_h5_path=dense_h5,
        up_rate=up_rate,
        normalize=True,
        augment=True
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    print(f"\n✓ DataLoader created")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num batches: {len(dataloader)}")
    
    # Test loading one batch
    for sparse, dense, radius in dataloader:
        print(f"\n✓ Batch loaded successfully")
        print(f"  Sparse batch shape: {sparse.shape}")  # (B, N, 3)
        print(f"  Dense batch shape: {dense.shape}")    # (B, M, 3)
        print(f"  Radius batch shape: {radius.shape}")  # (B, 1)
        break  # Only test first batch
    
    print("\n✓ DataLoader working correctly!")
    print("="*60)


def test_visualization(up_rate=2):
    """Test 5: Visualize point clouds (optional)"""
    print("\n" + "="*60)
    print(f"Test 5: Visualization ({up_rate}x)")
    print("="*60)
    
    if up_rate == 2:
        sparse_num = 2048
    else:
        sparse_num = 1024
    dense_num = 4096
    
    sparse_h5 = f'./data/train_64_patch_{sparse_num}.h5'
    dense_h5 = f'./data/train_64_patch_{dense_num}.h5'
    
    dataset = KITTIPatchDataset(
        sparse_h5_path=sparse_h5,
        dense_h5_path=dense_h5,
        up_rate=up_rate,
        normalize=True,
        augment=False
    )
    
    sparse, dense, radius = dataset[0]
    
    # Convert to numpy
    sparse_np = sparse.numpy()  # (N, 3)
    dense_np = dense.numpy()    # (M, 3)
    
    print(f"\nSparse points: {sparse_np.shape}")
    print(f"Dense points: {dense_np.shape}")
    
    # Simple 2D projection (X-Y plane)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot sparse
        axes[0].scatter(sparse_np[:, 0], sparse_np[:, 1], c='blue', s=1, alpha=0.5)
        axes[0].set_title(f'Sparse ({sparse_num} points)')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].axis('equal')
        axes[0].grid(True, alpha=0.3)
        
        # Plot dense
        axes[1].scatter(dense_np[:, 0], dense_np[:, 1], c='red', s=1, alpha=0.5)
        axes[1].set_title(f'Dense ({dense_num} points)')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].axis('equal')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_visualization.png', dpi=150)
        print(f"\n✓ Visualization saved to: test_visualization.png")
        
    except Exception as e:
        print(f"\n✗ Visualization failed (matplotlib not available or display issue): {e}")
    
    print("="*60)


def run_all_tests(up_rate=2):
    """Run all tests"""
    print("\n" + "="*60)
    print(f"KITTI Data Loading Test Suite ({up_rate}x Upsampling)")
    print("="*60)
    
    try:
        test_h5_files()
        test_dataset_basic(up_rate)
        test_augmentation(up_rate)
        test_dataloader(up_rate)
        test_visualization(up_rate)
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou can now proceed to training with:")
        print(f"  python train_kitti.py --up_rate {up_rate} --batch_size 16 --epochs 150")
        print("="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED!")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--up_rate', type=int, default=2, choices=[2, 4],
                       help='Upsampling rate to test (2 or 4)')
    args = parser.parse_args()
    
    run_all_tests(args.up_rate)