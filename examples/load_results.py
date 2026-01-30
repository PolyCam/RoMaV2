#!/usr/bin/env python3
"""
Example script demonstrating how to load and use batch matching results.

This script shows how to:
1. Load NPZ output files from batch_match.py
2. Extract dense warp fields and confidence
3. Use sampled keypoint matches
4. Visualize results
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_results(npz_path: Path) -> dict:
    """Load batch matching results from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    
    print(f"\nLoaded results from: {npz_path.name}")
    print(f"  Pair ID: {data['pair_id']}")
    print(f"  Setting: {data['setting']}")
    print(f"  Output shape: {data['output_shape']}")
    print(f"  Image A: {data['image_A_path']} (original size: {data['image_A_shape']})")
    print(f"  Image B: {data['image_B_path']} (original size: {data['image_B_shape']})")
    
    # Extract warp and confidence
    warp_AB = data['warp_AB']  # (H, W, 2) in normalized coords [-1, 1]
    confidence_AB = data['confidence_AB']  # (H, W, 4)
    
    # Compute overlap probability from confidence
    overlap_logit = confidence_AB[..., 0]
    overlap_prob = 1.0 / (1.0 + np.exp(-overlap_logit))
    
    print(f"\nWarp field shape: {warp_AB.shape}")
    print(f"Overlap range: [{overlap_prob.min():.3f}, {overlap_prob.max():.3f}]")
    print(f"Mean overlap: {overlap_prob.mean():.3f}")
    
    # Check for sampled matches
    if 'keypoints_A' in data:
        kpts_A = data['keypoints_A']
        kpts_B = data['keypoints_B']
        matches_conf = data['matches_confidence']
        print(f"\nSampled matches: {len(kpts_A)}")
        print(f"Match confidence range: [{matches_conf.min():.3f}, {matches_conf.max():.3f}]")
    
    return {
        'warp_AB': warp_AB,
        'confidence_AB': confidence_AB,
        'overlap_prob': overlap_prob,
        'image_A_path': str(data['image_A_path']),
        'image_B_path': str(data['image_B_path']),
        'image_A_shape': data['image_A_shape'],
        'image_B_shape': data['image_B_shape'],
        'output_shape': data['output_shape'],
        'keypoints_A': data.get('keypoints_A'),
        'keypoints_B': data.get('keypoints_B'),
        'matches_confidence': data.get('matches_confidence'),
    }


def warp_image(results: dict, save_path: Path):
    """Warp image B to image A and save visualization."""
    # Load images
    img_A = Image.open(results['image_A_path']).convert('RGB')
    img_B = Image.open(results['image_B_path']).convert('RGB')
    
    # Get output resolution
    H_out, W_out = results['output_shape']
    
    # Resize images to output resolution
    img_A_resized = img_A.resize((W_out, H_out))
    img_B_resized = img_B.resize((W_out, H_out))
    
    # Convert to tensors
    img_A_tensor = torch.from_numpy(np.array(img_A_resized)).float() / 255.0
    img_B_tensor = torch.from_numpy(np.array(img_B_resized)).float() / 255.0
    
    # Permute to (C, H, W)
    img_A_tensor = img_A_tensor.permute(2, 0, 1)
    img_B_tensor = img_B_tensor.permute(2, 0, 1)
    
    # Convert warp to tensor
    warp_AB = torch.from_numpy(results['warp_AB']).float()
    
    # Warp image B to A
    warped_B = F.grid_sample(
        img_B_tensor[None],
        warp_AB[None],
        mode='bilinear',
        align_corners=False
    )[0]
    
    # Get overlap mask
    overlap = torch.from_numpy(results['overlap_prob']).float()
    
    # Create visualization: overlap-masked warp
    # White background for non-overlapping regions
    white = torch.ones_like(warped_B)
    vis = overlap[..., None] * warped_B.permute(1, 2, 0) + (1 - overlap[..., None]) * white.permute(1, 2, 0)
    
    # Convert to PIL and save
    vis_np = (vis.numpy() * 255).astype(np.uint8)
    vis_img = Image.fromarray(vis_np)
    
    # Create side-by-side comparison
    combined = Image.new('RGB', (3 * W_out, H_out))
    combined.paste(img_A_resized, (0, 0))
    combined.paste(vis_img, (W_out, 0))
    combined.paste(img_B_resized, (2 * W_out, 0))
    
    combined.save(save_path)
    print(f"\nSaved warp visualization to: {save_path}")
    print("  Layout: [Image A | Warped B (masked) | Image B]")


def print_match_stats(results: dict):
    """Print statistics about sampled matches."""
    if results['keypoints_A'] is None:
        print("\nNo sampled matches available (use --num-samples when running batch_match.py)")
        return
    
    kpts_A = results['keypoints_A']
    kpts_B = results['keypoints_B']
    conf = results['matches_confidence']
    
    print(f"\nSampled Match Statistics:")
    print(f"  Total matches: {len(kpts_A)}")
    print(f"  Confidence: min={conf.min():.3f}, max={conf.max():.3f}, mean={conf.mean():.3f}")
    
    # Compute match distances (normalized)
    H_A, W_A = results['image_A_shape']
    H_B, W_B = results['image_B_shape']
    
    # Normalize to [0, 1]
    kpts_A_norm = kpts_A / np.array([W_A, H_A])
    kpts_B_norm = kpts_B / np.array([W_B, H_B])
    
    distances = np.linalg.norm(kpts_A_norm - kpts_B_norm, axis=1)
    print(f"  Match distances (normalized): mean={distances.mean():.3f}, median={np.median(distances):.3f}")
    
    # Filter by confidence threshold
    for threshold in [0.5, 0.7, 0.9]:
        high_conf = conf > threshold
        print(f"  Matches with conf > {threshold}: {high_conf.sum()} ({100*high_conf.sum()/len(conf):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Load and analyze batch matching results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        'npz_file',
        type=Path,
        help='Path to NPZ result file from batch_match.py'
    )
    parser.add_argument(
        '--visualize',
        type=Path,
        default=None,
        help='Save warp visualization to this path'
    )
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.npz_file)
    
    # Print match statistics
    print_match_stats(results)
    
    # Visualize if requested
    if args.visualize:
        warp_image(results, args.visualize)
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
