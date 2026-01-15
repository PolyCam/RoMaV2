#!/usr/bin/env python3
"""
Export RoMaV2 matches as a 3D point cloud visualization.

This script loads matched image pairs and their dense correspondences,
then creates a 3D point cloud where:
- Points from image A are positioned at z=0
- Points from image B are positioned at z=1
- Corresponding points are connected with lines
- Points are colored by their RGB values from the images

Usage:
    python examples/export_point_cloud.py \
        --match outputs/matches/toronto_A-toronto_B.npz \
        --output visualizations/toronto_match.ply \
        --subsample 5000
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    """Load image as RGB numpy array.
    
    Returns:
        np.ndarray: (H, W, 3) array with values in [0, 255]
    """
    img = Image.open(path).convert('RGB')
    return np.array(img)


def bilinear_sample(image: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Sample image at floating-point coordinates using bilinear interpolation.
    
    Args:
        image: (H, W, C) image array
        coords: (N, 2) array of (x, y) coordinates
    
    Returns:
        (N, C) sampled values
    """
    H, W = image.shape[:2]
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Clamp to valid range
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)
    
    # Get integer parts
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    
    # Get fractional parts
    fx = x - x0
    fy = y - y0
    
    # Bilinear interpolation
    fx = fx[:, None]
    fy = fy[:, None]
    
    values = (
        image[y0, x0] * (1 - fx) * (1 - fy) +
        image[y0, x1] * fx * (1 - fy) +
        image[y1, x0] * (1 - fx) * fy +
        image[y1, x1] * fx * fy
    )
    
    return values


def warp_to_correspondences(
    warp_AB: np.ndarray,
    confidence_AB: np.ndarray,
    H_A: int,
    W_A: int,
    H_B: int,
    W_B: int,
    subsample: int | None = None,
    min_confidence: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert warp field to point correspondences.
    
    Args:
        warp_AB: (H, W, 2) warp field in normalized [-1, 1] coordinates
        confidence_AB: (H, W, 4) confidence (overlap logit + precision)
        H_A, W_A: Original dimensions of image A
        H_B, W_B: Original dimensions of image B
        subsample: Maximum number of points to sample (None for all)
        min_confidence: Minimum overlap confidence threshold
    
    Returns:
        tuple: (points_A, points_B, confidences)
            - points_A: (N, 2) pixel coordinates in image A
            - points_B: (N, 2) pixel coordinates in image B
            - confidences: (N,) overlap confidence values
    """
    H_warp, W_warp = warp_AB.shape[:2]
    
    # Create grid of coordinates in image A
    y_grid, x_grid = np.meshgrid(
        np.arange(H_warp),
        np.arange(W_warp),
        indexing='ij'
    )
    
    # Convert to pixel coordinates in original image A
    points_A = np.stack([
        x_grid.flatten() / W_warp * W_A,
        y_grid.flatten() / H_warp * H_A
    ], axis=1)
    
    # Get corresponding points in B from warp field
    warp_coords = warp_AB.reshape(-1, 2)  # (N, 2) in [-1, 1]
    
    # Convert from [-1, 1] to pixel coordinates in image B
    points_B = np.stack([
        (warp_coords[:, 0] + 1) * W_B / 2,
        (warp_coords[:, 1] + 1) * H_B / 2
    ], axis=1)
    
    # Get confidence (sigmoid of overlap logit)
    overlap_logits = confidence_AB[..., 0].flatten()
    confidences = 1 / (1 + np.exp(-overlap_logits))
    
    # Filter by confidence
    valid_mask = confidences > min_confidence
    points_A = points_A[valid_mask]
    points_B = points_B[valid_mask]
    confidences = confidences[valid_mask]
    
    logger.info(f"Found {len(points_A)} valid correspondences (confidence > {min_confidence})")
    
    # Subsample if requested
    if subsample is not None and len(points_A) > subsample:
        # Sample with probability proportional to confidence
        probs = confidences / confidences.sum()
        indices = np.random.choice(len(points_A), size=subsample, replace=False, p=probs)
        points_A = points_A[indices]
        points_B = points_B[indices]
        confidences = confidences[indices]
        logger.info(f"Subsampled to {subsample} points")
    
    return points_A, points_B, confidences


def export_ply(
    output_path: Path,
    points_A: np.ndarray,
    points_B: np.ndarray,
    colors_A: np.ndarray,
    colors_B: np.ndarray,
    scale: float = 1.0,
    spacing: float = 1.0,
    include_lines: bool = True
):
    """Export point cloud to PLY format.
    
    Args:
        output_path: Output PLY file path
        points_A: (N, 2) pixel coordinates in image A
        points_B: (N, 2) pixel coordinates in image B
        colors_A: (N, 3) RGB colors for points A
        colors_B: (N, 3) RGB colors for points B
        scale: Scale factor for coordinates
        spacing: Z-spacing between the two image planes
        include_lines: Whether to include line elements connecting correspondences
    """
    N = len(points_A)
    
    # Create 3D coordinates
    # Image A at z=0, Image B at z=spacing
    vertices_A = np.column_stack([
        points_A[:, 0] * scale,
        points_A[:, 1] * scale,
        np.zeros(N)
    ])
    
    vertices_B = np.column_stack([
        points_B[:, 0] * scale,
        points_B[:, 1] * scale,
        np.full(N, spacing)
    ])
    
    # Combine vertices
    vertices = np.vstack([vertices_A, vertices_B])
    colors = np.vstack([colors_A, colors_B]).astype(np.uint8)
    
    total_vertices = len(vertices)
    
    # Write PLY file
    with open(output_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {total_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        
        if include_lines:
            f.write(f"element edge {N}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
        
        f.write("end_header\n")
        
        # Vertices
        for i in range(total_vertices):
            f.write(f"{vertices[i, 0]:.6f} {vertices[i, 1]:.6f} {vertices[i, 2]:.6f} ")
            f.write(f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")
        
        # Edges (correspondences)
        if include_lines:
            for i in range(N):
                f.write(f"{i} {i + N}\n")
    
    logger.info(f"Saved PLY with {total_vertices} vertices and {N if include_lines else 0} edges to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export RoMaV2 matches as 3D point cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export
  python examples/export_point_cloud.py \\
      --match outputs/matches/toronto_A-toronto_B.npz \\
      --output visualizations/toronto_match.ply
  
  # High-density point cloud
  python examples/export_point_cloud.py \\
      --match outputs/matches/toronto_A-toronto_B.npz \\
      --output visualizations/toronto_dense.ply \\
      --subsample 20000 --min-confidence 0.3
  
  # Points only (no correspondence lines)
  python examples/export_point_cloud.py \\
      --match outputs/matches/toronto_A-toronto_B.npz \\
      --output visualizations/toronto_points.ply \\
      --no-lines
        """
    )
    
    parser.add_argument(
        '--match',
        type=Path,
        required=True,
        help='Path to NPZ file with matching results'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output PLY file path'
    )
    parser.add_argument(
        '--subsample',
        type=int,
        default=5000,
        help='Maximum number of correspondences to export (default: 5000)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.5,
        help='Minimum overlap confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=0.01,
        help='Coordinate scale factor (default: 0.01)'
    )
    parser.add_argument(
        '--spacing',
        type=float,
        default=100.0,
        help='Z-spacing between image planes (default: 100.0)'
    )
    parser.add_argument(
        '--no-lines',
        action='store_true',
        help='Disable correspondence lines (points only)'
    )
    
    args = parser.parse_args()
    
    # Load match data
    logger.info(f"Loading match data from {args.match}")
    data = np.load(args.match, allow_pickle=True)
    
    warp_AB = data['warp_AB']
    confidence_AB = data['confidence_AB']
    H_A, W_A = data['image_A_shape']
    H_B, W_B = data['image_B_shape']
    path_A = str(data['image_A_path'])
    path_B = str(data['image_B_path'])
    
    logger.info(f"Image A: {path_A} ({W_A}x{H_A})")
    logger.info(f"Image B: {path_B} ({W_B}x{H_B})")
    logger.info(f"Warp resolution: {warp_AB.shape[1]}x{warp_AB.shape[0]}")
    
    # Load images
    logger.info("Loading images...")
    img_A = load_image(path_A)
    img_B = load_image(path_B)
    
    # Convert warp to correspondences
    logger.info("Extracting correspondences from warp field...")
    points_A, points_B, confidences = warp_to_correspondences(
        warp_AB=warp_AB,
        confidence_AB=confidence_AB,
        H_A=H_A,
        W_A=W_A,
        H_B=H_B,
        W_B=W_B,
        subsample=args.subsample,
        min_confidence=args.min_confidence
    )
    
    # Sample colors from images
    logger.info("Sampling colors from images...")
    colors_A = bilinear_sample(img_A, points_A)
    colors_B = bilinear_sample(img_B, points_B)
    
    # Export to PLY
    logger.info("Exporting point cloud...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_ply(
        output_path=args.output,
        points_A=points_A,
        points_B=points_B,
        colors_A=colors_A,
        colors_B=colors_B,
        scale=args.scale,
        spacing=args.spacing,
        include_lines=not args.no_lines
    )
    
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
