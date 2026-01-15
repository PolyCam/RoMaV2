#!/usr/bin/env python3
"""
Batch matching script for RoMaV2.

Efficiently matches a list of image pairs from a CSV file and saves results to disk.
Supports intelligent caching and pair ordering to maximize throughput.

Usage:
    python scripts/batch_match.py --pairs-csv pairs.csv --output-dir outputs/

CSV Format:
    image_A_path,image_B_path
    path/to/img1.jpg,path/to/img2.jpg
    ...
"""

import argparse
import csv
import logging
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from romav2 import RoMaV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class LRUImageCache:
    """LRU cache for loaded images with original dimension tracking."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: OrderedDict[str, tuple[torch.Tensor, int, int]] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, path: str, loader_fn) -> tuple[torch.Tensor, int, int]:
        """Get image from cache or load it.
        
        Returns:
            tuple: (image_tensor, H_original, W_original)
        """
        if path in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(path)
            return self.cache[path]
        
        self.misses += 1
        # Load image
        img_tensor = loader_fn(path)
        # Extract original dimensions (before any model resizing)
        # img_tensor shape: (1, 3, H, W) or (3, H, W)
        if img_tensor.dim() == 4:
            _, _, H_orig, W_orig = img_tensor.shape
        else:
            _, H_orig, W_orig = img_tensor.shape
        
        # Add to cache
        self.cache[path] = (img_tensor, H_orig, W_orig)
        
        # Evict oldest if over capacity
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        
        return self.cache[path]
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache)
        }


def load_pairs_csv(csv_path: Path) -> list[tuple[str, str, str]]:
    """Load image pairs from CSV file.
    
    Args:
        csv_path: Path to CSV file with format: image_A_path,image_B_path
    
    Returns:
        List of tuples: (pair_id, path_A, path_B)
    """
    pairs = []
    csv_dir = csv_path.parent
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, 1):
            if len(row) < 2:
                logger.warning(f"Line {i}: Expected 2 columns, got {len(row)}. Skipping.")
                continue
            
            path_A, path_B, *_ = row
            path_A = path_A.strip()
            path_B = path_B.strip()
            
            # Skip empty lines or headers
            if not path_A or not path_B or path_A.lower() == 'image_a_path' or path_A.lower() == 'imagea':
                continue
            
            # Resolve relative paths
            if not Path(path_A).is_absolute():
                path_A = str(csv_dir / path_A)
            if not Path(path_B).is_absolute():
                path_B = str(csv_dir / path_B)
            
            # Generate pair_id from filenames
            stem_A = Path(path_A).stem
            stem_B = Path(path_B).stem
            pair_id = f"{stem_A}-{stem_B}"
            
            pairs.append((pair_id, path_A, path_B))
    
    logger.info(f"Loaded {len(pairs)} image pairs from {csv_path}")
    return pairs


def optimize_pair_order(pairs: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Reorder pairs to maximize cache hits by clustering shared images.
    
    Uses a simple greedy algorithm to group pairs that share images.
    
    Args:
        pairs: List of (pair_id, path_A, path_B)
    
    Returns:
        Reordered list of pairs
    """
    if len(pairs) <= 1:
        return pairs
    
    # Build adjacency graph: pairs that share at least one image
    pair_indices = list(range(len(pairs)))
    adjacency = defaultdict(list)
    
    for i in range(len(pairs)):
        paths_i = {pairs[i][1], pairs[i][2]}
        for j in range(i + 1, len(pairs)):
            paths_j = {pairs[j][1], pairs[j][2]}
            if paths_i & paths_j:  # Share at least one image
                adjacency[i].append(j)
                adjacency[j].append(i)
    
    # Greedy traversal: start from pair with most connections
    visited = set()
    ordered = []
    
    # Find starting point (highest degree node)
    if adjacency:
        start = max(adjacency.keys(), key=lambda k: len(adjacency[k]))
    else:
        # No connections, return original order
        return pairs
    
    # BFS-like traversal favoring connected pairs
    queue = [start]
    
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        
        visited.add(current)
        ordered.append(pairs[current])
        
        # Add neighbors, sorted by degree (prefer high-degree nodes)
        neighbors = [n for n in adjacency[current] if n not in visited]
        neighbors.sort(key=lambda n: len(adjacency[n]), reverse=True)
        queue.extend(neighbors)
    
    # Add any remaining unvisited pairs
    for i in pair_indices:
        if i not in visited:
            ordered.append(pairs[i])
    
    logger.info(f"Optimized pair order: {len(ordered)} pairs reordered for better caching")
    return ordered


def save_results(
    output_path: Path,
    pair_id: str,
    warp_AB: torch.Tensor,
    overlap_AB: torch.Tensor,
    precision_AB: torch.Tensor,
    path_A: str,
    path_B: str,
    warp_BA: Optional[torch.Tensor] = None,
    overlap_BA: Optional[torch.Tensor] = None,
    precision_BA: Optional[torch.Tensor] = None,
    matches: Optional[torch.Tensor] = None,
    matches_confidence: Optional[torch.Tensor] = None,
    keypoints_A: Optional[torch.Tensor] = None,
    keypoints_B: Optional[torch.Tensor] = None,
    calibration: Optional[dict] = None,
):
    """Save matching results to NPZ file.
    
    Args:
        output_path: Output file path
        pair_id: Pair identifier
        warp_AB: (H, W, 2) warp field in normalized coordinates
        overlap_AB: (H, W, 1) overlap logit
        precision_AB: (H, W, 3) precision parameters [sigma_x, sigma_y, correlation]
        path_A, path_B: Image paths
        warp_BA, overlap_BA, precision_BA: Reverse direction (optional)
        matches: Optional (N, 4) sampled matches in normalized coords
        matches_confidence: Optional (N,) confidence for sampled matches
        keypoints_A: Optional (N, 2) keypoints in image A (pixel coords)
        keypoints_B: Optional (N, 2) keypoints in image B (pixel coords)
        calibration: Optional calibration dict. If provided, converts precision to normalized weights.
    """
    # Convert to numpy
    data = {
        'pair_id': pair_id,
        'warp_AB': warp_AB.cpu().numpy(),
        'overlap_AB': overlap_AB.cpu().numpy(),
        'image_A_path': path_A,
        'image_B_path': path_B,
    }

    # Process precision_AB
    if calibration is not None:
        # Convert to normalized weights and remove raw precision
        precision_AB_np = precision_AB.cpu().numpy()
        precision_weight_AB = precision_to_weight(precision_AB_np, calibration)
        data['precision_weight_AB'] = precision_weight_AB
    else:
        # Store raw precision
        data['precision_AB'] = precision_AB.cpu().numpy()

    # Add reverse direction if available
    if warp_BA is not None:
        data['warp_BA'] = warp_BA.cpu().numpy()
    if overlap_BA is not None:
        data['overlap_BA'] = overlap_BA.cpu().numpy()
    
    if precision_BA is not None:
        if calibration is not None:
            precision_BA_np = precision_BA.cpu().numpy()
            precision_weight_BA = precision_to_weight(precision_BA_np, calibration)
            data['precision_weight_BA'] = precision_weight_BA
        else:
            data['precision_BA'] = precision_BA.cpu().numpy()
    
    # Add optional sampled matches
    if matches is not None:
        data['matches'] = matches.cpu().numpy()
    if matches_confidence is not None:
        data['matches_confidence'] = matches_confidence.cpu().numpy()
    if keypoints_A is not None:
        data['keypoints_A'] = keypoints_A.cpu().numpy()
    if keypoints_B is not None:
        data['keypoints_B'] = keypoints_B.cpu().numpy()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    np.savez_compressed(output_path, **data)


def calibrate_precision_params(precision_samples: np.ndarray) -> dict:
    """Calibrate precision parameters from collected samples.
    
    Computes threshold and temperature for sigmoid normalization of precision
    eigenvalues to [0, 1] weights.
    
    Args:
        precision_samples: Array of shape (num_samples, 3) containing
                          [sigma_x, sigma_y, correlation] representing 2x2 covariance matrices
    
    Returns:
        Dictionary with calibration parameters:
            - 'threshold': eigenvalue threshold for sigmoid
            - 'temperature': temperature parameter for sigmoid smoothness
            - 'mean_eigenvalue': mean of minimum eigenvalues (for reference)
            - 'num_samples': number of samples used
    """
    logger.info(f"Calibrating precision parameters from {len(precision_samples)} samples...")
    
    # Convert list to numpy array if needed
    if isinstance(precision_samples, list):
        precision_samples = np.array(precision_samples)
    
    # precision_samples shape: (num_samples, 3) with [sigma_x, sigma_y, correlation]
    sigma_x = precision_samples[:, 0]  # (num_samples,)
    sigma_y = precision_samples[:, 1]  # (num_samples,)
    corr = precision_samples[:, 2]     # (num_samples,)
    
    # Compute eigenvalues of 2x2 symmetric matrix
    # For [[a, b], [b, c]]: eigenvalues = (a+c)/2 ± sqrt((a-c)^2/4 + b^2)
    a = sigma_x ** 2
    c = sigma_y ** 2
    b = corr * sigma_x * sigma_y
    
    trace = a + c
    det = a * c - b ** 2
    discriminant = (trace / 2) ** 2 - det
    discriminant = np.maximum(discriminant, 0)  # Ensure non-negative
    
    sqrt_disc = np.sqrt(discriminant)
    lambda_min = trace / 2 - sqrt_disc
    
    # Use minimum eigenvalue (smaller = more precise)
    all_eigenvalues = lambda_min
    all_eigenvalues = all_eigenvalues[np.isfinite(all_eigenvalues)]  # Remove NaNs/Infs
    
    # Compute threshold as median eigenvalue
    threshold = float(np.median(all_eigenvalues))
    
    # Compute temperature to make sigmoid transition smooth
    # Use standard deviation scaled appropriately
    temperature = float(np.std(all_eigenvalues) / 2.0)
    
    # Ensure temperature is positive and reasonable
    temperature = max(temperature, 1e-6)
    
    calibration = {
        'threshold': threshold,
        'temperature': temperature,
        'mean_eigenvalue': float(np.mean(all_eigenvalues)),
        'median_eigenvalue': threshold,
        'std_eigenvalue': float(np.std(all_eigenvalues)),
        'num_samples': len(precision_samples),
    }
    
    logger.info("Calibration parameters:")
    logger.info(f"  threshold:  {calibration['threshold']:.6f}")
    logger.info(f"  temperature: {calibration['temperature']:.6f}")
    logger.info(f"  mean eigenvalue: {calibration['mean_eigenvalue']:.6f}")
    logger.info(f"  std eigenvalue:  {calibration['std_eigenvalue']:.6f}")
    
    return calibration


def precision_matrix_to_params(precision_matrix: np.ndarray) -> np.ndarray:
    """Convert 2×2 precision matrix to 3-parameter representation.
    
    Args:
        precision_matrix: (H, W, 2, 2) array of 2×2 covariance matrices
    
    Returns:
        (H, W, 3) array with [sigma_x, sigma_y, correlation]
    """
    # Extract matrix elements
    # Matrix is [[sigma_x^2, rho*sigma_x*sigma_y], [rho*sigma_x*sigma_y, sigma_y^2]]
    a = precision_matrix[..., 0, 0]  # sigma_x^2
    b = precision_matrix[..., 0, 1]  # rho*sigma_x*sigma_y
    c = precision_matrix[..., 1, 1]  # sigma_y^2
    
    # Compute parameters
    sigma_x = np.sqrt(np.maximum(a, 1e-10))
    sigma_y = np.sqrt(np.maximum(c, 1e-10))
    corr = b / (sigma_x * sigma_y + 1e-10)
    corr = np.clip(corr, -1.0, 1.0)
    
    # Stack into (H, W, 3)
    params = np.stack([sigma_x, sigma_y, corr], axis=-1)
    return params.astype(np.float32)


def precision_to_weight(precision_params: np.ndarray, calibration: dict) -> np.ndarray:
    """Convert 3-channel precision parameters to normalized weight in [0, 1].
    
    Computes minimum eigenvalue of 2x2 covariance matrix and applies sigmoid
    normalization with learned threshold and temperature.
    
    Args:
        precision_params: (H, W, 3) or (H, W, 2, 2) array with precision parameters
        calibration: Calibration dictionary from calibrate_precision_params()
    
    Returns:
        (H, W, 1) array with normalized weights in [0, 1]
    """
    # Convert from 2×2 matrix format if needed
    if precision_params.ndim == 4 and precision_params.shape[2:] == (2, 2):
        precision_params = precision_matrix_to_params(precision_params)
    
    sigma_x = precision_params[..., 0]
    sigma_y = precision_params[..., 1]
    corr = precision_params[..., 2]
    
    # Reconstruct 2x2 covariance matrices
    a = sigma_x ** 2
    c = sigma_y ** 2
    b = corr * sigma_x * sigma_y
    
    # Compute eigenvalues
    trace = a + c
    det = a * c - b ** 2
    discriminant = (trace / 2) ** 2 - det
    discriminant = np.maximum(discriminant, 0)
    
    sqrt_disc = np.sqrt(discriminant)
    lambda_min = trace / 2 - sqrt_disc
    
    # Apply sigmoid normalization
    threshold = calibration['threshold']
    temperature = calibration['temperature']
    
    z = (lambda_min - threshold) / temperature
    z = np.clip(z, -100, 100)
    weight = 1.0 / (1.0 + np.exp(-z))
    
    return weight[..., np.newaxis].astype(np.float32)


def sample_precision_from_center(precision: np.ndarray, num_samples: int = 1000) -> np.ndarray:
    """Sample precision values from center region of image.
    
    Randomly samples points from the center 70% of the image to avoid edge artifacts.
    
    Args:
        precision: (H, W, 3) or (H, W, 2, 2) precision array
        num_samples: Number of samples to draw
    
    Returns:
        (num_samples, 3) array of sampled precision values
    """
    # Convert from 2×2 matrix format if needed
    if precision.ndim == 4 and precision.shape[2:] == (2, 2):
        precision = precision_matrix_to_params(precision)
    
    # Validate shape
    if precision.ndim != 3 or precision.shape[2] != 3:
        raise ValueError(f"Expected precision shape (H, W, 3), got {precision.shape}")
    
    H, W, _ = precision.shape
    
    # Define center region: 70% of image (15% margin on each side)
    h_start = int(H * 0.15)
    h_end = int(H * 0.85)
    w_start = int(W * 0.15)
    w_end = int(W * 0.85)
    
    center_region = precision[h_start:h_end, w_start:w_end, :]
    center_h, center_w, _ = center_region.shape
    
    # Randomly sample points
    num_pixels = center_h * center_w
    num_samples = min(num_samples, num_pixels)
    
    # Flatten and sample
    flat_precision = center_region.reshape(-1, 3)
    indices = np.random.choice(flat_precision.shape[0], size=num_samples, replace=False)
    sampled = flat_precision[indices, :]
    
    return sampled


def update_npz_with_calibrated_precision(npz_path: Path, calibration: dict):
    """Update an NPZ file to add calibrated precision weights and remove raw precision.
    
    Args:
        npz_path: Path to NPZ file to update
        calibration: Calibration parameters
    """
    # Load existing data
    data = np.load(npz_path, allow_pickle=True)
    updated_data = dict(data)
    
    # Process precision_AB
    if 'precision_AB' in data:
        precision_AB = data['precision_AB']  # (H, W, 3) or (H, W, 2, 2)
        precision_weight = precision_to_weight(precision_AB, calibration)  # (H, W, 1)
        updated_data['precision_weight_AB'] = precision_weight
        # Remove raw precision to save space
        del updated_data['precision_AB']
    
    # Process precision_BA if present
    if 'precision_BA' in data:
        precision_BA = data['precision_BA']  # (H, W, 3) or (H, W, 2, 2)
        precision_weight = precision_to_weight(precision_BA, calibration)  # (H, W, 1)
        updated_data['precision_weight_BA'] = precision_weight
        # Remove raw precision to save space
        del updated_data['precision_BA']
    
    # Save updated data
    np.savez_compressed(npz_path, **updated_data)


def process_batch(
    model: RoMaV2,
    batch_pairs: list[tuple[str, str, str]],
    output_dir: Path,
    cache: Optional[LRUImageCache],
    num_samples: Optional[int],
    stats: dict,
    precision_samples: Optional[np.ndarray] = None,
    max_calibration_samples: int = 0,
    calibration_sample_size: int = 500,
    calibration: Optional[dict] = None,
) -> tuple[list[Path], Optional[np.ndarray]]:
    """Process a batch of image pairs.
    
    Args:
        model: RoMaV2 model
        batch_pairs: List of (pair_id, path_A, path_B)
        output_dir: Output directory
        cache: Image cache (or None if caching disabled)
        num_samples: Number of matches to sample (or None)
        stats: Statistics dictionary to update
        precision_samples: Optional list to collect precision parameters for calibration
        max_calibration_samples: Maximum number of samples to collect for calibration
        calibration_sample_size: Number of points to sample per pair for calibration
        calibration: Optional calibration dict. If provided, saves with normalized precision.
    
    Returns:
        List of output file paths that were successfully created
    """
    output_paths = []
    
    for pair_id, path_A, path_B in batch_pairs:
        output_path = output_dir / f"{pair_id}.npz"
        
        # Skip if already processed
        if output_path.exists():
            stats['skipped'] += 1
            continue
        
        try:
            # Load images
            if cache is not None:
                img_A, H_A, W_A = cache.get(path_A, model._load_image)
                img_B, H_B, W_B = cache.get(path_B, model._load_image)
            else:
                img_A = model._load_image(path_A)
                img_B = model._load_image(path_B)
                # Extract original dimensions
                if img_A.dim() == 4:
                    _, _, H_A, W_A = img_A.shape
                    _, _, H_B, W_B = img_B.shape
                else:
                    _, H_A, W_A = img_A.shape
                    _, H_B, W_B = img_B.shape
            
            # Match
            preds = model.match(img_A, img_B)
            
            # Get outputs (remove batch dimension)
            warp_AB = preds['warp_AB'][0]  # (H, W, 2)
            overlap_AB = preds['overlap_AB'][0]  # (H, W, 1)
            precision_AB = preds['precision_AB'][0]  # (H, W, 2, 2)
            warp_BA = preds['warp_BA'][0] if preds['warp_BA'] is not None else None
            overlap_BA = preds['overlap_BA'][0] if preds['overlap_BA'] is not None else None
            precision_BA = preds['precision_BA'][0] if preds['precision_BA'] is not None else None

            # Collect precision samples for calibration (sample from center)
            if (precision_samples is not None and 
                len(precision_samples) < max_calibration_samples):
                # Sample from center region
                precision_AB_np = precision_AB.cpu().numpy()
                sampled_AB = sample_precision_from_center(precision_AB_np, calibration_sample_size)
                precision_samples = np.vstack([precision_samples, sampled_AB])
                if precision_BA is not None:
                    precision_BA_np = precision_BA.cpu().numpy()
                    sampled_BA = sample_precision_from_center(precision_BA_np, calibration_sample_size)
                    precision_samples = np.vstack([precision_samples, sampled_BA])

            # Optionally sample matches
            matches = None
            matches_confidence = None
            keypoints_A = None
            keypoints_B = None
            if num_samples is not None and num_samples > 0:
                try:
                    preds_for_sampling = {
                        'warp_AB': preds['warp_AB'],
                        'overlap_AB': preds['overlap_AB'],
                        'precision_AB': preds['precision_AB'],
                    }
                    if preds['warp_BA'] is not None:
                        preds_for_sampling.update({
                            'warp_BA': preds['warp_BA'],
                            'overlap_BA': preds['overlap_BA'],
                            'precision_BA': preds['precision_BA'],
                        })

                    matches, matches_confidence, _, _ = model.sample(
                        preds_for_sampling, num_samples
                    )
                    
                    # Convert to pixel coordinates
                    keypoints_A, keypoints_B = model.to_pixel_coordinates(
                        matches, H_A, W_A, H_B, W_B
                    )
                except Exception as e:
                    logger.warning(f"Failed to sample matches for {pair_id}: {e}")
            
            # Save results
            save_results(
                output_path=output_path,
                pair_id=pair_id,
                warp_AB=warp_AB,
                overlap_AB=overlap_AB,
                precision_AB=precision_AB,
                path_A=path_A,
                path_B=path_B,
                warp_BA=warp_BA,
                overlap_BA=overlap_BA,
                precision_BA=precision_BA,
                matches=matches,
                matches_confidence=matches_confidence,
                keypoints_A=keypoints_A,
                keypoints_B=keypoints_B,
                calibration=calibration,
            )
            
            stats['processed'] += 1
            output_paths.append(output_path)
            
        except FileNotFoundError as e:
            logger.warning(f"Image not found for {pair_id}: {e}")
            stats['failed'] += 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"OOM error on {pair_id}. Try reducing batch size or image resolution.")
                stats['failed'] += 1
                # Clear cache to free memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            else:
                logger.error(f"Runtime error on {pair_id}: {e}")
                stats['failed'] += 1
        except Exception as e:
            import traceback
            logger.error(f"Unexpected error on {pair_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            stats['failed'] += 1
    
    return output_paths, precision_samples


def main():
    parser = argparse.ArgumentParser(
        description="Batch matching for RoMaV2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Basic usage
            python scripts/batch_match.py --pairs-csv pairs.csv --output-dir outputs/
            
            # With image caching and match sampling
            python scripts/batch_match.py --pairs-csv pairs.csv --output-dir outputs/ \\
                --cache-images --cache-size 200 --num-samples 5000
            
            # Fast processing with lower quality
            python scripts/batch_match.py --pairs-csv pairs.csv --output-dir outputs/ \\
                --setting fast --batch-size 4
            
            # With precision calibration (100 pairs × 1000 samples per pair = 100k total)
            python scripts/batch_match.py --pairs-csv pairs.csv --output-dir outputs/ \\
                --calibration-pairs 100 --calibration-sample-size 1000
        """
    )
    
    parser.add_argument(
        '--pairs-csv',
        type=Path,
        required=True,
        help='Path to CSV file with image pairs (format: image_A_path,image_B_path)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for NPZ files'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for processing (default: 1)'
    )
    parser.add_argument(
        '--setting',
        type=str,
        choices=['turbo', 'fast', 'base', 'precise'],
        default='base',
        help='Model setting (default: base)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of matches to sample per pair (optional)'
    )
    parser.add_argument(
        '--cache-images',
        action='store_true',
        help='Enable LRU image caching'
    )
    parser.add_argument(
        '--cache-size',
        type=int,
        default=100,
        help='Maximum number of images to cache (default: 100)'
    )
    parser.add_argument(
        '--no-optimize-order',
        action='store_true',
        help='Disable pair order optimization (process in original order)'
    )
    parser.add_argument(
        '--no-bidirectional',
        dest='bidirectional',
        action='store_false',
        default=True,
        help='Disable bidirectional matching (only A->B; default is bidirectional)'
    )
    parser.add_argument(
        '--calibration-pairs',
        type=int,
        default=15,
        help='Number of pairs to use for precision calibration (optional). '
             'Collects precision statistics from first N pairs, then '
             'updates all NPZ files with normalized precision weights.'
    )
    parser.add_argument(
        '--calibration-sample-size',
        type=int,
        default=1000,
        help='Number of random points to sample from center of each pair for calibration (default: 1000). '
             'Total calibration samples = calibration-pairs × calibration-sample-size'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.pairs_csv.exists():
        logger.error(f"Pairs CSV file not found: {args.pairs_csv}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pairs
    pairs = load_pairs_csv(args.pairs_csv)
    if not pairs:
        logger.error("No valid pairs found in CSV file")
        return 1
    
    # Optimize pair order for caching
    if args.cache_images and not args.no_optimize_order:
        pairs = optimize_pair_order(pairs)
    
    # Initialize model
    logger.info(f"Initializing RoMaV2 with setting: {args.setting}")
    model = RoMaV2()
    model.apply_setting(args.setting)
    model.bidirectional = args.bidirectional
    model.eval()
    logger.info(f"Bidirectional matching is {'enabled' if model.bidirectional else 'disabled'}")
    
    # Initialize cache if enabled
    cache = LRUImageCache(max_size=args.cache_size) if args.cache_images else None
    if cache:
        logger.info(f"Image caching enabled (max size: {args.cache_size})")
    
    # Initialize calibration if requested
    precision_samples = np.empty((0, 3), dtype=np.float32) if args.calibration_pairs is not None else None
    calibration = None
    max_calibration_samples = 0
    
    if precision_samples is not None:
        max_calibration_samples = args.calibration_pairs * args.calibration_sample_size
        if model.bidirectional:
            max_calibration_samples *= 2  # Account for both directions per pair
        logger.info(f"Precision calibration enabled:")
        logger.info(f"  Pairs to calibrate:     {args.calibration_pairs}")
        logger.info(f"  Sample size per pair:   {args.calibration_sample_size}")
        logger.info(f"  Total calibration samples: {max_calibration_samples}")
    
    # Statistics
    stats = {
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'start_time': time.time(),
    }
    
    # Process pairs
    logger.info(f"Processing {len(pairs)} pairs...")
    
    # Note: Batching is per-pair for now since images can have different resolutions
    # Could be extended to batch pairs with same resolution
    if args.batch_size > 1:
        logger.warning("Batch size > 1 not fully implemented. Processing one pair at a time.")
    
    processed_files = []
    
    # Determine effective batch size for first batch (limited by calibration_pairs if enabled)
    first_batch_size = args.batch_size
    if precision_samples is not None and args.calibration_pairs is not None:
        first_batch_size = min(args.batch_size, args.calibration_pairs)
        if first_batch_size < args.batch_size:
            logger.info(f"First batch limited to {first_batch_size} pairs (calibration requirement)")
    
    calibration_complete = False
    
    with tqdm(total=len(pairs), desc="Matching pairs", unit="pair") as pbar:
        idx = 0
        while idx < len(pairs):
            # Determine batch size
            current_batch_size = args.batch_size if calibration_complete else first_batch_size
            batch_end = min(idx + current_batch_size, len(pairs))
            batch = pairs[idx:batch_end]
            
            for pair_id, path_A, path_B in batch:
                # Check if we need to calibrate (only before calibration_complete)
                if (not calibration_complete and
                    precision_samples is not None and 
                    calibration is None and 
                    len(precision_samples) >= max_calibration_samples):
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Reached {len(precision_samples)} calibration samples")
                    logger.info(f"{'='*60}")
                    
                    # Calibrate precision parameters
                    calibration = calibrate_precision_params(precision_samples)
                    
                    # Update all previously processed files
                    logger.info(f"Updating {len(processed_files)} NPZ files with calibrated precision...")
                    for npz_path in tqdm(processed_files, desc="Updating NPZ files", unit="file"):
                        update_npz_with_calibrated_precision(npz_path, calibration)
                    
                    # Clear samples to free memory
                    precision_samples = None
                    calibration_complete = True
                    logger.info("Calibration complete. Continuing with remaining pairs...\n")
                
                # Process pair and get output paths (and updated precision samples)
                new_outputs, precision_samples = process_batch(
                    model=model,
                    batch_pairs=[(pair_id, path_A, path_B)],
                    output_dir=args.output_dir,
                    cache=cache,
                    num_samples=args.num_samples,
                    stats=stats,
                    precision_samples=precision_samples,
                    max_calibration_samples=max_calibration_samples,
                    calibration_sample_size=args.calibration_sample_size,
                    calibration=calibration,
                )
                
                # Track processed files for calibration update (only before calibration)
                if not calibration_complete and precision_samples is not None:
                    processed_files.extend(new_outputs)
                
                pbar.update(1)
                
                # Update progress bar description
                if stats['processed'] > 0:
                    elapsed = time.time() - stats['start_time']
                    rate = stats['processed'] / elapsed
                    postfix = {
                        'rate': f'{rate:.2f} pairs/s',
                        'skipped': stats['skipped'],
                        'failed': stats['failed']
                    }
                    if precision_samples is not None:
                        postfix['cal_samples'] = len(precision_samples)
                        postfix['cal_target'] = max_calibration_samples
                    pbar.set_postfix(postfix)
            
            idx = batch_end
    
    # Final statistics
    elapsed = time.time() - stats['start_time']
    logger.info("\n" + "="*60)
    logger.info("BATCH MATCHING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total pairs:     {len(pairs)}")
    logger.info(f"Processed:       {stats['processed']}")
    logger.info(f"Skipped:         {stats['skipped']}")
    logger.info(f"Failed:          {stats['failed']}")
    logger.info(f"Elapsed time:    {elapsed:.1f}s")
    logger.info(f"Throughput:      {stats['processed']/elapsed:.2f} pairs/s")
    
    if calibration is not None:
        logger.info(f"\nPrecision Calibration:")
        logger.info(f"  Calibration pairs:      {args.calibration_pairs}")
        logger.info(f"  Sample size per pair:   {args.calibration_sample_size}")
        logger.info(f"  Total samples collected: {max_calibration_samples}")
        logger.info(f"  Applied to all {len(pairs)} pairs")
    
    if cache:
        cache_stats = cache.get_stats()
        logger.info(f"\nCache Statistics:")
        logger.info(f"  Hit rate:      {cache_stats['hit_rate']:.1%}")
        logger.info(f"  Hits:          {cache_stats['hits']}")
        logger.info(f"  Misses:        {cache_stats['misses']}")
        logger.info(f"  Final size:    {cache_stats['size']}")
    
    logger.info(f"\nResults saved to: {args.output_dir}")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())
