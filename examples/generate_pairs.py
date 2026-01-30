#!/usr/bin/env python3
"""
Helper script to generate a pairs CSV file from image directories.

This script helps create the input CSV file needed by batch_match.py.
"""

import argparse
from pathlib import Path
from itertools import combinations
from typing import Optional


def generate_all_pairs(image_dir: Path, extensions=None) -> list[tuple[Path, Path]]:
    """Generate all possible pairs from images in a directory.
    
    Args:
        image_dir: Directory containing images
        extensions: List of valid extensions (default: ['.jpg', '.jpeg', '.png'])
    
    Returns:
        List of (path_A, path_B) tuples
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    # Find all images
    images = []
    for ext in extensions:
        images.extend(image_dir.glob(f'**/*{ext}'))
    
    images = sorted(set(images))
    print(f"Found {len(images)} images in {image_dir}")
    
    # Generate all pairs
    pairs = list(combinations(images, 2))
    print(f"Generated {len(pairs)} pairs")
    
    return pairs


def generate_sequential_pairs(image_dir: Path, extensions=None, overlap=1) -> list[tuple[Path, Path]]:
    """Generate sequential pairs (e.g., for video frames or ordered images).
    
    Args:
        image_dir: Directory containing images
        extensions: List of valid extensions
        overlap: Number of following images to pair with each image (default: 1)
    
    Returns:
        List of (path_A, path_B) tuples
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    # Find all images
    images = []
    for ext in extensions:
        images.extend(image_dir.glob(f'**/*{ext}'))
    
    images = sorted(set(images))
    print(f"Found {len(images)} images in {image_dir}")
    
    # Generate sequential pairs
    pairs = []
    for i in range(len(images)):
        for j in range(1, overlap + 1):
            if i + j < len(images):
                pairs.append((images[i], images[i + j]))
    
    print(f"Generated {len(pairs)} sequential pairs (overlap={overlap})")
    
    return pairs


def generate_pairs_from_list(list_file: Path, base_dir: Optional[Path] = None) -> list[tuple[Path, Path]]:
    """Generate pairs from a text file listing image paths (one per line).
    
    Creates all combinations of listed images.
    
    Args:
        list_file: Text file with one image path per line
        base_dir: Optional base directory to resolve relative paths
    
    Returns:
        List of (path_A, path_B) tuples
    """
    with open(list_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    images = []
    for line in lines:
        if base_dir:
            img_path = base_dir / line
        else:
            img_path = Path(line)
        
        if img_path.exists():
            images.append(img_path)
        else:
            print(f"Warning: Image not found: {img_path}")
    
    print(f"Found {len(images)} valid images from {list_file}")
    
    # Generate all pairs
    pairs = list(combinations(images, 2))
    print(f"Generated {len(pairs)} pairs")
    
    return pairs


def save_pairs_csv(pairs: list[tuple[Path, Path]], output_file: Path, relative_to: Optional[Path] = None):
    """Save pairs to CSV file.
    
    Args:
        pairs: List of (path_A, path_B) tuples
        output_file: Output CSV file path
        relative_to: Optional directory to make paths relative to
    """
    with open(output_file, 'w') as f:
        # Write header
        f.write("image_A_path,image_B_path\n")
        
        # Write pairs
        for path_A, path_B in pairs:
            if relative_to:
                try:
                    path_A = path_A.relative_to(relative_to)
                    path_B = path_B.relative_to(relative_to)
                except ValueError:
                    pass  # Keep absolute if can't make relative
            
            f.write(f"{path_A},{path_B}\n")
    
    print(f"\nSaved {len(pairs)} pairs to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate pairs CSV for batch matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # All pairs from a directory
            python examples/generate_pairs.py --image-dir data/images/ --output pairs.csv --mode all
            
            # Sequential pairs (for video frames)
            python examples/generate_pairs.py --image-dir data/frames/ --output pairs.csv --mode sequential --overlap 5
            
            # Pairs from a list file
            python examples/generate_pairs.py --list-file images.txt --output pairs.csv --mode list
        """
    )
    
    parser.add_argument(
        '--image-dir',
        type=Path,
        help='Directory containing images'
    )
    parser.add_argument(
        '--list-file',
        type=Path,
        help='Text file with image paths (one per line)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--mode',
        choices=['all', 'sequential', 'list'],
        default='all',
        help='Pairing mode (default: all)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=1,
        help='For sequential mode: number of following images to pair with each image (default: 1)'
    )
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'],
        help='Valid image extensions (default: .jpg .jpeg .png)'
    )
    parser.add_argument(
        '--relative',
        action='store_true',
        help='Use relative paths in output CSV'
    )
    parser.add_argument(
        '--max-pairs',
        type=int,
        default=None,
        help='Maximum number of pairs to generate (useful for large directories)'
    )
    
    args = parser.parse_args()
    
    # Generate pairs based on mode
    if args.mode == 'list':
        if not args.list_file:
            parser.error("--list-file required for mode 'list'")
        base_dir = args.image_dir if args.image_dir else Path.cwd()
        pairs = generate_pairs_from_list(args.list_file, base_dir)
    elif args.mode == 'sequential':
        if not args.image_dir:
            parser.error("--image-dir required for mode 'sequential'")
        pairs = generate_sequential_pairs(args.image_dir, args.extensions, args.overlap)
    else:  # mode == 'all'
        if not args.image_dir:
            parser.error("--image-dir required for mode 'all'")
        pairs = generate_all_pairs(args.image_dir, args.extensions)
    
    # Limit pairs if requested
    if args.max_pairs and len(pairs) > args.max_pairs:
        print(f"Limiting to {args.max_pairs} pairs (from {len(pairs)})")
        pairs = pairs[:args.max_pairs]
    
    # Determine relative path base
    relative_to = None
    if args.relative:
        if args.output.parent.exists():
            relative_to = args.output.parent
        else:
            relative_to = Path.cwd()
    
    # Save to CSV
    save_pairs_csv(pairs, args.output, relative_to)


if __name__ == "__main__":
    main()
