import torch
import torch.utils.data as data
import h5py
import numpy as np
from dataset.utils import *


class KITTIPatchDataset(data.Dataset):
    """
    KITTI LiDAR Patch Dataset for Point Cloud Upsampling
    
    Data Structure:
        - sparse_h5: train_64_patch_2048.h5 (input sparse point cloud)
        - dense_h5: train_64_patch_4096.h5 (ground truth dense point cloud)
        - Each h5 file contains multiple patches as separate datasets
        - Each patch: (N, 4) - [x, y, z, intensity]
        - Only XYZ is used for training (intensity discarded)
    
    Args:
        sparse_h5_path: Path to sparse point cloud h5 file
        dense_h5_path: Path to dense point cloud h5 file
        up_rate: Upsampling rate (2 or 4)
        normalize: Whether to normalize coordinates
        augment: Whether to apply data augmentation (only for training)
        use_random_input: Whether to randomly sample input points
        jitter_sigma: Sigma for jitter augmentation
        jitter_max: Max clip value for jitter
    """
    
    def __init__(self, sparse_h5_path, dense_h5_path, up_rate=2,
                 normalize=True, augment=False,
                 use_random_input=True, jitter_sigma=0.01, jitter_max=0.03):
        super(KITTIPatchDataset, self).__init__()
        
        self.sparse_h5_path = sparse_h5_path
        self.dense_h5_path = dense_h5_path
        self.up_rate = up_rate
        self.normalize = normalize
        self.augment = augment
        self.use_random_input = use_random_input
        self.jitter_sigma = jitter_sigma
        self.jitter_max = jitter_max
        
        # Fixed radius for 7m x 7m patches
        self.base_radius = 5.0
        
        # Load patch names from sparse h5 file
        with h5py.File(sparse_h5_path, 'r') as f:
            self.patch_names = list(f.keys())
        
        # Verify that dense h5 has the same patches
        with h5py.File(dense_h5_path, 'r') as f:
            dense_patch_names = set(f.keys())
        
        # Keep only patches that exist in both files
        self.patch_names = [name for name in self.patch_names if name in dense_patch_names]
        
        if len(self.patch_names) == 0:
            raise ValueError(f"No matching patches found between {sparse_h5_path} and {dense_h5_path}")
        
        print(f"[KITTIPatchDataset] Loaded {len(self.patch_names)} patches")
        print(f"  - Sparse h5: {sparse_h5_path}")
        print(f"  - Dense h5: {dense_h5_path}")
        print(f"  - Upsampling rate: {up_rate}x")
        print(f"  - Normalize: {normalize}")
        print(f"  - Augment: {augment}")
    
    def __len__(self):
        return len(self.patch_names)
    
    def __getitem__(self, index):
        patch_name = self.patch_names[index]
        
        # Load sparse point cloud (input)
        with h5py.File(self.sparse_h5_path, 'r') as f:
            sparse_points = f[patch_name][:]  # (N_sparse, 4)
        
        # Load dense point cloud (ground truth)
        with h5py.File(self.dense_h5_path, 'r') as f:
            dense_points = f[patch_name][:]  # (N_dense, 4)
        
        # Convert to float32
        sparse_points = sparse_points.astype(np.float32)
        dense_points = dense_points.astype(np.float32)
        
        # Extract only XYZ (discard intensity for now)
        sparse_xyz = sparse_points[:, :3]  # (N_sparse, 3)
        dense_xyz = dense_points[:, :3]    # (N_dense, 3)
        
        # Initialize radius
        radius = self.base_radius
        
        # Normalize coordinates (center and scale)
        if self.normalize:
            # Use dense point cloud's centroid for normalization
            centroid = dense_xyz.mean(axis=0)  # (3,)
            sparse_xyz = sparse_xyz - centroid
            dense_xyz = dense_xyz - centroid
            
            # Scale by fixed radius
            sparse_xyz = sparse_xyz / self.base_radius
            dense_xyz = dense_xyz / self.base_radius
        
        # Data augmentation (applied to both sparse and dense with same parameters)
        if self.augment:
            # Generate random augmentation parameters
            angle = np.random.uniform(-45, 45) * np.pi / 180  # Z-axis rotation
            scale = np.random.uniform(0.85, 1.15)
            
            # Apply same transformations to both
            sparse_xyz = self._apply_augmentation(sparse_xyz, angle, scale)
            dense_xyz = self._apply_augmentation(dense_xyz, angle, scale)
            
            # Update radius
            radius = radius * scale
        
        # Random sampling of input points (if enabled)
        if self.use_random_input and self.augment:
            num_sparse = sparse_xyz.shape[0]
            sample_idx = nonuniform_sampling(num_sparse, sample_num=num_sparse)
            sparse_xyz = sparse_xyz[sample_idx, :]
        
        # Jitter augmentation (only on input, only if augment is True)
        if self.augment and self.use_random_input:
            sparse_xyz = jitter_perturbation_point_cloud(
                sparse_xyz, sigma=self.jitter_sigma, clip=self.jitter_max
            )
        
        # Convert to torch tensors
        sparse_xyz = torch.from_numpy(sparse_xyz).float()  # (N_sparse, 3)
        dense_xyz = torch.from_numpy(dense_xyz).float()    # (N_dense, 3)
        radius = torch.FloatTensor([radius])               # (1,)
        
        return sparse_xyz, dense_xyz, radius
    
    def _apply_augmentation(self, points, angle, scale):
        """
        Apply augmentation transformations
        
        Args:
            points: (N, 3) point cloud
            angle: rotation angle in radians (Z-axis)
            scale: scaling factor
        
        Returns:
            augmented_points: (N, 3)
        """
        # Z-axis rotation
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ], dtype=np.float32)
        points = points @ rotation_matrix.T
        
        # Scaling
        points = points * scale
        
        # Jitter (small noise)
        noise = np.random.randn(*points.shape).astype(np.float32) * 0.01
        noise = np.clip(noise, -0.03, 0.03)
        points = points + noise
        
        return points


def get_kitti_dataloader(args, split='train'):
    """
    Create KITTI dataloader
    
    Args:
        args: arguments containing dataset configuration
        split: 'train', 'val', or 'test'
    
    Returns:
        dataloader: PyTorch DataLoader
    """
    # Determine h5 file paths based on split and up_rate
    if args.up_rate == 2:
        sparse_num = 2048
    elif args.up_rate == 4:
        sparse_num = 1024
    else:
        raise ValueError(f"Unsupported up_rate: {args.up_rate}")
    
    dense_num = 4096
    
    # Construct file paths
    sparse_h5_path = args.h5_data_path.replace('split', split).replace('sparse', str(sparse_num))
    dense_h5_path = args.h5_data_path.replace('split', split).replace('sparse', str(dense_num))
    
    # Alternatively, if you want explicit paths:
    # sparse_h5_path = f'./data/{split}_64_patch_{sparse_num}.h5'
    # dense_h5_path = f'./data/{split}_64_patch_{dense_num}.h5'
    
    # Determine augmentation
    augment = (split == 'train')
    
    # Create dataset
    dataset = KITTIPatchDataset(
        sparse_h5_path=sparse_h5_path,
        dense_h5_path=dense_h5_path,
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


if __name__ == '__main__':
    """Test data loading"""
    import argparse
    
    # Test arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparse_h5', default='./data/train_64_patch_2048.h5')
    parser.add_argument('--dense_h5', default='./data/train_64_patch_4096.h5')
    parser.add_argument('--up_rate', default=2, type=int)
    args = parser.parse_args()
    
    # Create dataset
    dataset = KITTIPatchDataset(
        sparse_h5_path=args.sparse_h5,
        dense_h5_path=args.dense_h5,
        up_rate=args.up_rate,
        normalize=True,
        augment=True
    )
    
    print(f"\n=== Dataset Info ===")
    print(f"Total patches: {len(dataset)}")
    
    # Test loading one sample
    sparse, dense, radius = dataset[0]
    print(f"\n=== Sample 0 ===")
    print(f"Sparse shape: {sparse.shape}")
    print(f"Dense shape: {dense.shape}")
    print(f"Radius: {radius.item():.4f}")
    print(f"Sparse XYZ range: [{sparse.min().item():.4f}, {sparse.max().item():.4f}]")
    print(f"Dense XYZ range: [{dense.min().item():.4f}, {dense.max().item():.4f}]")
    
    print("\n=== Data loading test passed! ===")