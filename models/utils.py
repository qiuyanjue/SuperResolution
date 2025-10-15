"""
Patched utils.py with fallback to PyTorch-native implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import random
import numpy as np
from einops import rearrange

# Try to import pointops, if failed use PyTorch native implementation
try:
    from models import pointops
    USE_POINTOPS = True
except Exception as e:
    USE_POINTOPS = False

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name, log_dir):
    """Create logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = os.path.join(log_dir, f'{name}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def get_knn_pts(k, pts, query_pts, return_idx=False):
    """
    Find k nearest neighbors for query points
    
    Args:
        k: number of neighbors
        pts: (B, 3, N) reference points
        query_pts: (B, 3, M) query points
        return_idx: whether to return indices
    
    Returns:
        knn_pts: (B, 3, M, k) k nearest neighbor points
        knn_idx: (B, M, k) indices (if return_idx=True)
    """
    # PyTorch native implementation (always use this for compatibility)
    B, _, N = pts.shape
    _, _, M = query_pts.shape
    
    # Compute distances
    pts_t = pts.permute(0, 2, 1)          # (B, N, 3)
    query_t = query_pts.permute(0, 2, 1)  # (B, M, 3)
    
    # (B, M, N)
    dist = torch.cdist(query_t, pts_t, p=2)
    
    # Find k nearest neighbors
    knn_dist, knn_idx = torch.topk(dist, k, dim=2, largest=False)  # (B, M, k)
    
    # Gather neighbor points
    knn_idx_expanded = knn_idx.unsqueeze(1).expand(-1, 3, -1, -1)  # (B, 3, M, k)
    pts_expanded = pts.unsqueeze(2).expand(-1, -1, M, -1)           # (B, 3, M, N)
    
    knn_pts = torch.gather(pts_expanded, 3, knn_idx_expanded)       # (B, 3, M, k)
    
    if return_idx:
        return knn_pts, knn_idx
    else:
        return knn_pts


def index_points(points, idx):
    """
    Index points using given indices
    
    Args:
        points: (B, C, N) point features
        idx: (B, M, K) or (B, M) indices
    
    Returns:
        indexed_points: (B, C, M, K) or (B, C, M)
    """
    device = points.device
    B, C, N = points.shape
    
    if idx.dim() == 2:
        # (B, M)
        M = idx.shape[1]
        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)  # (B, C, M)
        indexed_points = torch.gather(points, 2, idx_expanded)  # (B, C, M)
        return indexed_points
    else:
        # (B, M, K)
        M, K = idx.shape[1], idx.shape[2]
        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, M, K)
        points_expanded = points.unsqueeze(2).expand(-1, -1, M, -1)  # (B, C, M, N)
        indexed_points = torch.gather(points_expanded, 3, idx_expanded)  # (B, C, M, K)
        return indexed_points


def midpoint_interpolate(args, input_pts):
    """
    Midpoint interpolation for point cloud upsampling
    
    Args:
        args: arguments containing up_rate
        input_pts: (B, 3, N) input points
    
    Returns:
        interpolate_pts: (B, 3, N*up_rate) interpolated points
    """
    B, C, N = input_pts.shape
    up_rate = args.up_rate
    
    # Method 1: Simple duplication with perturbation
    interpolate_pts = input_pts.unsqueeze(3).expand(-1, -1, -1, up_rate)  # (B, 3, N, up_rate)
    interpolate_pts = interpolate_pts.reshape(B, C, N * up_rate)          # (B, 3, N*up_rate)
    
    # Add small perturbations
    noise = torch.randn_like(interpolate_pts) * 0.03
    interpolate_pts = interpolate_pts + noise
    
    return interpolate_pts


def get_query_points(interpolate_pts, args):
    """
    Generate query points by adding local perturbations
    
    Args:
        interpolate_pts: (B, 3, M) interpolated points
        args: arguments containing local_sigma
    
    Returns:
        query_pts: (B, 3, M) query points
    """
    # Add local Gaussian noise
    noise = torch.randn_like(interpolate_pts) * args.local_sigma
    query_pts = interpolate_pts + noise
    
    return query_pts


def chamfer_distance(pred, gt):
    """
    Compute Chamfer Distance between two point clouds
    
    Args:
        pred: (B, 3, N) predicted points
        gt: (B, 3, M) ground truth points
    
    Returns:
        cd: scalar Chamfer Distance
    """
    # pred: (B, 3, N), gt: (B, 3, M)
    pred = pred.permute(0, 2, 1)  # (B, N, 3)
    gt = gt.permute(0, 2, 1)      # (B, M, 3)
    
    # Compute pairwise distances: (B, N, M)
    dist = torch.cdist(pred, gt, p=2)
    
    # pred -> gt: for each predicted point, find nearest GT point
    min_dist_pred_to_gt = dist.min(dim=2)[0]  # (B, N)
    
    # gt -> pred: for each GT point, find nearest predicted point
    min_dist_gt_to_pred = dist.min(dim=1)[0]  # (B, M)
    
    # Chamfer Distance (bidirectional)
    cd = min_dist_pred_to_gt.mean() + min_dist_gt_to_pred.mean()
    
    return cd


def knn_point(k, xyz, query_xyz):
    """
    KNN search (for compatibility with pointops interface)
    
    Args:
        k: number of neighbors
        xyz: (B, N, 3) reference points
        query_xyz: (B, M, 3) query points
    
    Returns:
        dist: (B, M, k) distances
        idx: (B, M, k) indices
    """
    # xyz: (B, N, 3), query_xyz: (B, M, 3)
    dist = torch.cdist(query_xyz, xyz, p=2)  # (B, M, N)
    knn_dist, knn_idx = torch.topk(dist, k, dim=2, largest=False)  # (B, M, k)
    
    return knn_dist, knn_idx


def square_distance(src, dst):
    """
    Calculate squared Euclidean distance between every two points
    
    Args:
        src: (B, N, 3)
        dst: (B, M, 3)
    
    Returns:
        dist: (B, N, M)
    """
    dist = torch.cdist(src, dst, p=2) ** 2
    return dist


# Additional utility functions that might be needed

def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling
    
    Args:
        xyz: (B, N, 3) point cloud
        npoint: number of points to sample
    
    Returns:
        centroids: (B, npoint) sampled point indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query
    
    Args:
        radius: local region radius
        nsample: max sample number in local region
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) query points
    
    Returns:
        group_idx: (B, S, nsample) grouped points index
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    
    return group_idx