#!/usr/bin/env python3
"""
Batch matching script for RoMaV2.

Efficiently matches a list of image pairs from a CSV file and saves results to disk.
Supports intelligent caching and pair ordering to maximize throughput.

Usage:
    python scripts/batch_match_min.py --pairs-csv pairs.csv --output-dir outputs/

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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

from romav2 import RoMaV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

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

def save_results(
    output_path: Path,
    pair_id: str,
    warp_AB: torch.Tensor,
    overlap_AB: torch.Tensor,
    path_A: str,
    path_B: str,
    save_format: str = 'pt',
):
    """Save matching results to pt or npy file.
    
    Args:
        output_path: Output file path
        pair_id: Pair identifier
        warp_AB: (H, W, 2) warp field in normalized coordinates
        overlap_AB: (H, W, 1) overlap logit
        path_A, path_B: Image paths
        save_format: Format to save ('pt' or 'npy')
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    with torch.profiler.record_function("save_to_disk"):
        if save_format == 'npy':
            # Convert tensors to numpy and save as npz
            np.savez(
                output_path,
                pair_id=pair_id,
                warp_AB=warp_AB.cpu().numpy(),
                overlap_AB=overlap_AB.cpu().numpy(),
                image_A_path=path_A,
                image_B_path=path_B,
            )
        else:
            # Convert to dict
            data = {
                'pair_id': pair_id,
                'warp_AB': warp_AB,
                'overlap_AB': overlap_AB,
                'image_A_path': path_A,
                'image_B_path': path_B,
            }
            torch.save(data, output_path)

@torch.profiler.record_function("load_im")
def load_im(path):
    img_pil = Image.open(path)
    img_pil = img_pil.convert("RGB")
    img = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)

    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    return img

class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return idx, load_im(self.paths[idx])

@torch.profiler.record_function("load_images")
def load_images(pairs):
    iset = set()
    for idx in range(len(pairs)):
        iset.add(pairs[idx][1])
        iset.add(pairs[idx][2])
    iset = list(iset)
    print(f"Loading {len(iset)} images...")
    dataloader = DataLoader(ImageDataset(iset),
                            batch_size=1,
                            prefetch_factor=2,
                            pin_memory=True,
                            num_workers=20)
    icache = {}
    for idx, im in dataloader: 
        icache[iset[idx]] = im
    return icache


enroll_queue = []
enroll_batch = 16
@torch.profiler.record_function("enroll")
def enroll(
        cache,
        model,
        image_path:str,
        icache=None
        ):
    if(image_path in cache):
        return

    if(icache):
        img = icache[image_path].to("cuda:0")
    else:
        img = model._load_image(image_path)

    global enroll_queue
    enroll_queue.append([image_path,img])
    if(len(enroll_queue) >= enroll_batch):
        enroll_flush(cache, model)

@torch.profiler.record_function("enroll_flush")
def enroll_flush(cache, model):
    global enroll_queue
    if(len(enroll_queue) != 0):
        bt = model.cache_batch_features(enroll_queue)
        for x in bt:
            cache[x["path"]] = x
    enroll_queue = []
    pass
       
def streaming_cache(model, pairs):
    iset = set()
    for idx in range(len(pairs)):
        iset.add(pairs[idx][1])
        iset.add(pairs[idx][2])
    iset = list(iset)
    print(f"Loading {len(iset)} images...")
    dataloader = DataLoader(ImageDataset(iset),
                            batch_size=1,
                            prefetch_factor=2,
                            pin_memory=True,
                            num_workers=20)
    icache = {}
    cache  = {}
    for idx, im in dataloader: 
        icache[iset[idx]] = im
        enroll(cache, model, iset[idx], icache)
    enroll_flush(cache, model)

    return cache

def process_real_batch(
    model: RoMaV2,
    batch_pairs: list[tuple[str, str, str]],
    output_dir: Path,
    stats: dict,
    cache,
    save_format: str = 'pt',
) -> None:

    output_paths = []
    feat1_A  = []
    feat1_B  = []
    feat2_A  = []
    feat2_B  = []
    img_A   = []
    img_B   = []

    # Determine file extension based on save format
    file_ext = '.npz' if save_format == 'npy' else '.pt'

    #Assemble batch
    for pair_id, path_A, path_B in batch_pairs:
        output_path = output_dir / f"{pair_id}{file_ext}"
        output_paths.append(output_path)
        frame_A = cache[path_A]
        frame_B = cache[path_B]
        feat1_A.append(frame_A["features"][0])
        feat1_B.append(frame_B["features"][0])
        feat2_A.append(frame_A["features"][1])
        feat2_B.append(frame_B["features"][1])
        img_A.append(frame_A["rescaled"])
        img_B.append(frame_B["rescaled"])

    batch_A = {"features": [torch.cat(feat1_A),
                            torch.cat(feat2_A)],
               "rescaled": torch.cat(img_A)}
    batch_B = {"features": [torch.cat(feat1_B),
                            torch.cat(feat2_B)],
               "rescaled": torch.cat(img_B)}

    #Perform Inference
    with torch.profiler.record_function("DNN_exec"):
        preds = model.coarse_cached_match(batch_A, batch_B)

    #Dissassemble batch
    with torch.profiler.record_function("save_batch"):
      if(False):
        idx = 0
        for pair_id, path_A, path_B in batch_pairs:
            # Get outputs (remove batch dimension)
            warp_AB = preds['warp_AB'][idx]  # (H, W, 2)
            overlap_AB = preds['overlap_AB'][idx]  # (H, W, 1)
            
            # Save results
            with torch.profiler.record_function("save_warps"):
              save_results(
                output_path=output_paths[idx],
                pair_id=pair_id,
                warp_AB=warp_AB,
                overlap_AB=overlap_AB,
                path_A=path_A,
                path_B=path_B,
                save_format=save_format,
            )
            idx += 1
            stats['processed'] += 1
      else:
        stats['processed'] += len(output_paths)
        ids = []
        pA  = []
        pB  = []
        for pair_id, path_A, path_B in batch_pairs:
            ids.append(pair_id)
            pA.append(path_A)
            pB.append(path_B)
        
        # Use first output path for batch saving
        output_path = output_paths[0]
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save
        with torch.profiler.record_function("save_to_disk"):
            if save_format == 'npy':
                # Save as npz with arrays
                np.savez(
                    output_path,
                    pair_id=ids,
                    warp_AB=preds["warp_AB"].cpu().numpy(),
                    overlap_AB=preds["overlap_AB"].cpu().numpy(),
                    image_A_path=pA,
                    image_B_path=pB,
                )
            else:
                # Convert to dict
                data = {
                    'pair_id':      ids,
                    'warp_AB':      preds["warp_AB"],
                    'overlap_AB':   preds["overlap_AB"],
                    'image_A_path': pA,
                    'image_B_path': pB,
                }
                torch.save(data, output_path)

def main():
    parser = argparse.ArgumentParser(
        description="Batch matching for RoMaV2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Basic usage
            python scripts/batch_match_min.py --pairs-csv pairs.csv --output-dir outputs/
            
            # Save results in NumPy format
            python scripts/batch_match_min.py --pairs-csv pairs.csv --output-dir outputs/ \\
                --save-format npy
            
            # With image caching and match sampling
            python scripts/batch_match_min.py --pairs-csv pairs.csv --output-dir outputs/ \\
            
            # Fast processing with lower quality
            python scripts/batch_match_min.py --pairs-csv pairs.csv --output-dir outputs/ \\
                --setting fast --batch-size 4
            
            # With precision calibration (100 pairs × 1000 samples per pair = 100k total)
            python scripts/batch_match_min.py --pairs-csv pairs.csv --output-dir outputs/ \\
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
        '--save-format',
        type=str,
        choices=['pt', 'npy'],
        default='pt',
        help='Save format for output files: pt (PyTorch) or npy (NumPy) (default: pt)'
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
    
    # Initialize model
    logger.info(f"Initializing RoMaV2 with setting: {args.setting}")
    model = RoMaV2(RoMaV2.Cfg(compile=False))
    model.apply_setting(args.setting)
    model.eval()
    
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
    
    # Adjust file extensions based on save format
    file_ext = '.npz' if args.save_format == 'npy' else '.pt'
    logger.info(f"Saving results in {args.save_format.upper()} format (extension: {file_ext})")
    
    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]
    #if(True):
    with torch.profiler.profile(activities=activities,
                                record_shapes=True,
                                profile_memory=False,) as prof:
     with torch.profiler.record_function("streaming_feature_cache"):
         cache = streaming_cache(model, pairs)

     with torch.profiler.record_function("matching_pairs"):
      with tqdm(total=len(pairs), desc="Matching pairs", unit="pair") as pbar:
        idx = 0
        while idx < len(pairs):
            # Determine batch size
            current_batch_size = args.batch_size
            batch_end = min(idx + current_batch_size, len(pairs))
            batch = pairs[idx:batch_end]
            
            with torch.profiler.record_function("process_batch"):
                new_outputs = process_real_batch(
                        model=model,
                        batch_pairs=batch,
                        output_dir=args.output_dir,
                        stats=stats,
                        cache=cache,
                        save_format=args.save_format,
                    )
             
            pbar.update(len(batch))
                
            # Update progress bar description
            if stats['processed'] > 0:
                elapsed = time.time() - stats['start_time']
                rate = stats['processed'] / elapsed
                postfix = {
                    'rate': f'{rate:.2f} pairs/s',
                    'skipped': stats['skipped'],
                    'failed': stats['failed']
                }
                pbar.set_postfix(postfix)
            
            idx = batch_end
    
    prof.export_chrome_trace("trace.json")
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
    
    logger.info(f"\nResults saved to: {args.output_dir}")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())
