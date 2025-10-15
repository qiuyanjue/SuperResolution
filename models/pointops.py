"""
Complete patched pointops.py with fallback to PyTorch-native implementations
Supports automatic offset generation and proper batch indexing
Version: 2024-10-10 - Cleaned
"""

import torch

# Always use PyTorch fallback (CUDA version not compiled)
USE_CUDA = False

# ==================== PyTorch Fallback Implementations ====================

def knnquery_pytorch(nsample, xyz, new_xyz, offset, new_offset, return_global_idx=True):
    """
    PyTorch fallback for knnquery
    
    Args:
        nsample: number of neighbors
        xyz: (M, 3) all points
        new_xyz: (N, 3) query points
        offset: (B,) cumulative point counts for xyz
        new_offset: (B,) cumulative point counts for new_xyz
        return_global_idx: if True, return global indices; if False, return local indices per batch
    
    Returns:
        idx: (N, nsample) neighbor indices
        dist2: (N, nsample) squared distances
    """
    B = len(offset)
    N = new_xyz.shape[0]
    M = xyz.shape[0]
    
    idx = torch.zeros(N, nsample, dtype=torch.int32, device=xyz.device)
    dist2 = torch.zeros(N, nsample, dtype=torch.float32, device=xyz.device)
    
    start_xyz = 0
    start_new = 0
    
    for b in range(B):
        end_xyz = offset[b].item()
        end_new = new_offset[b].item()
        
        xyz_batch = xyz[start_xyz:end_xyz]  # (n, 3)
        new_xyz_batch = new_xyz[start_new:end_new]  # (m, 3)
        
        if xyz_batch.shape[0] == 0 or new_xyz_batch.shape[0] == 0:
            start_xyz = end_xyz
            start_new = end_new
            continue
        
        # Compute distances
        dist = torch.cdist(new_xyz_batch, xyz_batch, p=2)  # (m, n)
        
        # Find k nearest
        k = min(nsample, xyz_batch.shape[0])
        dist2_batch, idx_batch = torch.topk(dist, k, dim=1, largest=False)  # (m, k)
        
        # Store results with appropriate indexing
        if return_global_idx:
            # Global indices (for original pointops interface)
            idx[start_new:end_new, :k] = idx_batch.int() + start_xyz
        else:
            # Local indices (for batch processing)
            idx[start_new:end_new, :k] = idx_batch.int()
        
        dist2[start_new:end_new, :k] = dist2_batch ** 2  # squared distance
        
        # Fill remaining with first neighbor if k < nsample
        if k < nsample:
            idx[start_new:end_new, k:] = idx[start_new:end_new, 0:1].expand(-1, nsample - k)
            dist2[start_new:end_new, k:] = dist2[start_new:end_new, 0:1].expand(-1, nsample - k)
        
        start_xyz = end_xyz
        start_new = end_new
    
    return idx, dist2


def grouping_pytorch(input, idx):
    """
    PyTorch fallback for grouping
    
    Args:
        input: (M, C) features
        idx: (N, nsample) indices
    
    Returns:
        output: (N, nsample, C) grouped features
    """
    N, nsample = idx.shape
    C = input.shape[1]
    
    # Clamp indices to valid range
    idx_clamped = torch.clamp(idx.long(), 0, input.shape[0] - 1)
    
    # Flatten idx and use advanced indexing
    idx_flat = idx_clamped.view(-1)  # (N*nsample,)
    gathered = input[idx_flat]  # (N*nsample, C)
    output = gathered.view(N, nsample, C)  # (N, nsample, C)
    
    return output


def gathering_pytorch(input, idx):
    """
    PyTorch fallback for gathering
    
    Args:
        input: (M, C) features
        idx: (N,) indices
    
    Returns:
        output: (N, C) gathered features
    """
    idx_clamped = torch.clamp(idx.long(), 0, input.shape[0] - 1)
    output = input[idx_clamped]
    return output


def interpolation_pytorch(xyz, new_xyz, input, offset, new_offset, k=3):
    """
    PyTorch fallback for interpolation
    
    Args:
        xyz: (M, 3) known points
        new_xyz: (N, 3) query points
        input: (M, C) known features
        offset: (B,) cumulative counts for xyz
        new_offset: (B,) cumulative counts for new_xyz
        k: number of neighbors for interpolation
    
    Returns:
        output: (N, C) interpolated features
    """
    N, C = new_xyz.shape[0], input.shape[1]
    output = torch.zeros(N, C, dtype=input.dtype, device=input.device)
    
    B = len(offset)
    start_xyz = 0
    start_new = 0
    
    for b in range(B):
        end_xyz = offset[b].item()
        end_new = new_offset[b].item()
        
        xyz_batch = xyz[start_xyz:end_xyz]
        new_xyz_batch = new_xyz[start_new:end_new]
        input_batch = input[start_xyz:end_xyz]
        
        if xyz_batch.shape[0] == 0 or new_xyz_batch.shape[0] == 0:
            start_xyz = end_xyz
            start_new = end_new
            continue
        
        # Compute distances
        dist = torch.cdist(new_xyz_batch, xyz_batch, p=2)  # (m, n)
        
        # Find k nearest
        k_actual = min(k, xyz_batch.shape[0])
        dist_k, idx_k = torch.topk(dist, k_actual, dim=1, largest=False)  # (m, k)
        
        # Inverse distance weighting
        eps = 1e-10
        weights = 1.0 / (dist_k + eps)  # (m, k)
        weights = weights / weights.sum(dim=1, keepdim=True)  # normalize
        
        # Gather features
        idx_k_long = idx_k.long()
        neighbor_features = input_batch[idx_k_long]  # (m, k, C)
        
        # Weighted sum
        output_batch = (neighbor_features * weights.unsqueeze(-1)).sum(dim=1)  # (m, C)
        
        output[start_new:end_new] = output_batch
        
        start_xyz = end_xyz
        start_new = end_new
    
    return output


def sampling_pytorch(input, idx):
    """PyTorch fallback for sampling"""
    return gathering_pytorch(input, idx)


# ==================== Autograd Function Wrappers ====================

class KNNQuery(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nsample, xyz, new_xyz, offset=None, new_offset=None, return_global_idx=True):
        # Auto-generate offsets if not provided
        if offset is None:
            M = xyz.shape[0]
            offset = torch.tensor([M], dtype=torch.int32, device=xyz.device)
        
        if new_offset is None:
            N = new_xyz.shape[0]
            new_offset = torch.tensor([N], dtype=torch.int32, device=new_xyz.device)
        
        if USE_CUDA:
            try:
                return pointops_cuda.knnquery_cuda(nsample, xyz, new_xyz, offset, new_offset)
            except:
                pass
        
        return knnquery_pytorch(nsample, xyz, new_xyz, offset, new_offset, return_global_idx)
    
    @staticmethod
    def backward(ctx, grad_idx, grad_dist):
        return None, None, None, None, None, None


class Grouping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, idx):
        if USE_CUDA:
            try:
                return pointops_cuda.grouping_forward_cuda(input, idx)
            except:
                pass
        return grouping_pytorch(input, idx)
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None


class Gathering(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, idx):
        if USE_CUDA:
            try:
                return pointops_cuda.gathering_forward_cuda(input, idx)
            except:
                pass
        return gathering_pytorch(input, idx)
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None


class Interpolation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, new_xyz, input, offset=None, new_offset=None, k=3):
        # Auto-generate offsets if not provided
        if offset is None:
            M = xyz.shape[0]
            offset = torch.tensor([M], dtype=torch.int32, device=xyz.device)
        
        if new_offset is None:
            N = new_xyz.shape[0]
            new_offset = torch.tensor([N], dtype=torch.int32, device=new_xyz.device)
        
        if USE_CUDA:
            try:
                return pointops_cuda.interpolation_forward_cuda(xyz, new_xyz, input, offset, new_offset, k)
            except:
                pass
        
        return interpolation_pytorch(xyz, new_xyz, input, offset, new_offset, k)
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None, None


class Sampling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, idx):
        if USE_CUDA:
            try:
                return pointops_cuda.sampling_forward_cuda(input, idx)
            except:
                pass
        return sampling_pytorch(input, idx)
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None


# ==================== Exported API Functions ====================

def knnquery(nsample, xyz, new_xyz, offset=None, new_offset=None, return_global_idx=True):
    """
    K-Nearest Neighbors query
    
    Args:
        nsample: number of neighbors
        xyz: (M, 3) reference points
        new_xyz: (N, 3) query points
        offset: optional (B,) cumulative counts for xyz
        new_offset: optional (B,) cumulative counts for new_xyz
        return_global_idx: whether to return global or local indices
    
    Returns:
        idx: (N, nsample) neighbor indices
        dist2: (N, nsample) squared distances
    """
    return KNNQuery.apply(nsample, xyz, new_xyz, offset, new_offset, return_global_idx)


def knnquery_heap(nsample, xyz, new_xyz, offset=None, new_offset=None):
    """
    K-Nearest Neighbors query (heap version for compatibility)
    Returns only indices with proper batch-local indexing
    
    Args:
        nsample: number of neighbors
        xyz: (M, 3) or (B, M, 3) reference points
        new_xyz: (N, 3) or (B, N, 3) query points
        offset: optional (B,) cumulative counts
        new_offset: optional (B,) cumulative counts
    
    Returns:
        idx: (N, nsample) or (B, N, nsample) indices (local per batch)
    """
    original_dim = xyz.dim()
    
    if original_dim == 3:
        # (B, M, 3) format - batch processing
        B, M, _ = xyz.shape
        _, N, _ = new_xyz.shape
        
        xyz_flat = xyz.reshape(-1, 3)
        new_xyz_flat = new_xyz.reshape(-1, 3)
        
        if offset is None:
            offset = torch.arange(1, B+1, dtype=torch.int32, device=xyz.device) * M
        if new_offset is None:
            new_offset = torch.arange(1, B+1, dtype=torch.int32, device=new_xyz.device) * N
        
        # Get indices with LOCAL indexing per batch
        idx, dist2 = knnquery_pytorch(nsample, xyz_flat, new_xyz_flat, offset, new_offset, return_global_idx=False)
        
        # Reshape: idx is (B*N, nsample) with local indices
        idx = idx.reshape(B, N, nsample)
        
        return idx
    
    else:
        # (M, 3) format - single batch
        M = xyz.shape[0]
        N = new_xyz.shape[0]
        
        if offset is None:
            offset = torch.tensor([M], dtype=torch.int32, device=xyz.device)
        if new_offset is None:
            new_offset = torch.tensor([N], dtype=torch.int32, device=new_xyz.device)
        
        # Get indices with local indexing
        idx, dist2 = knnquery_pytorch(nsample, xyz, new_xyz, offset, new_offset, return_global_idx=False)
        
        return idx


# Export other functions
grouping = Grouping.apply
gathering = Gathering.apply
interpolation = Interpolation.apply
sampling = Sampling.apply


# ==================== Additional Utility Functions ====================

def subtraction(input, idx, output):
    """
    Compute feature differences (for compatibility)
    
    Args:
        input: (M, C) features
        idx: (N, nsample) indices
        output: (N, nsample, C) output tensor
    
    Returns:
        output: (N, nsample, C) feature differences
    """
    N, nsample, C = output.shape
    grouped = grouping(input, idx)  # (N, nsample, C)
    
    # Compute differences (grouped - center)
    center = input[idx[:, 0:1].long()]  # (N, 1, C)
    output = grouped - center
    
    return output


def aggregation(input, position, weight, idx, output):
    """
    Feature aggregation (for compatibility)
    
    Args:
        input: (M, C) features
        position: (M, 3) positions
        weight: (N, nsample, C) weights
        idx: (N, nsample) indices
        output: (N, C) output tensor
    
    Returns:
        output: (N, C) aggregated features
    """
    N, nsample = idx.shape
    C = input.shape[1]
    
    grouped = grouping(input, idx)  # (N, nsample, C)
    weighted = grouped * weight  # (N, nsample, C)
    output = weighted.sum(dim=1)  # (N, C)
    
    return output


# ==================== Module Information ====================

__all__ = [
    'knnquery',
    'knnquery_heap',
    'grouping',
    'gathering',
    'interpolation',
    'sampling',
    'subtraction',
    'aggregation',
]

__version__ = '1.0.0-patched'
__author__ = 'Patched for PyTorch fallback compatibility'