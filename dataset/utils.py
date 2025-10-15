"""
Data utilities for KITTI point cloud dataset

This module provides utility functions for data augmentation and sampling.
Only includes functions that are actually used in the codebase.
"""

import numpy as np


def nonuniform_sampling(num, sample_num):
    """
    Non-uniform sampling of point cloud indices
    
    Samples points with higher probability around a randomly selected location.
    Uses Gaussian distribution centered at a random location (0.1 to 0.9).
    
    Args:
        num: Total number of points
        sample_num: Number of points to sample
    
    Returns:
        list: Sampled indices
    
    Example:
        # Sample 2048 points from 2048 points with non-uniform distribution
        sample_idx = nonuniform_sampling(2048, sample_num=2048)
        sampled_points = points[sample_idx, :]
    
    Note:
        This creates a biased sampling where points near the randomly chosen
        center location have higher probability of being selected.
    """
    sample = set()
    loc = np.random.rand() * 0.8 + 0.1  # Random center in [0.1, 0.9]
    
    while len(sample) < sample_num:
        # Sample using Gaussian centered at loc
        a = int(np.random.normal(loc=loc, scale=0.3) * num)
        if a < 0 or a >= num:
            continue
        sample.add(a)
    
    return list(sample)


def jitter_perturbation_point_cloud(input, sigma=0.005, clip=0.02):
    """
    Add random jitter to point cloud coordinates
    
    Jittering adds per-point Gaussian noise, useful for data augmentation.
    The noise is clipped to prevent extreme perturbations.
    
    Args:
        input: (N, C) numpy array, original point cloud
        sigma: Standard deviation of Gaussian noise (default: 0.005)
        clip: Maximum absolute value of noise (default: 0.02)
    
    Returns:
        (N, C) numpy array, jittered point cloud
    
    Example:
        # Add small random perturbations to point cloud
        jittered_points = jitter_perturbation_point_cloud(
            points, sigma=0.01, clip=0.03
        )
    
    Note:
        The noise is applied independently to each point and each dimension.
        Larger sigma values create stronger perturbations.
    """
    N, C = input.shape
    assert clip > 0, "Clip value must be positive"
    
    # Generate Gaussian noise and clip to range [-clip, clip]
    jittered_data = np.clip(sigma * np.random.randn(N, C), -clip, clip)
    jittered_data += input
    
    return jittered_data


# Module exports
__all__ = ['nonuniform_sampling', 'jitter_perturbation_point_cloud']