#!/usr/bin/env python3
"""
Script to convert a local dataset saved with save_to_disk to Parquet format that can be loaded with load_dataset.
Based on HuggingFace documentation: datasets saved with save_to_disk can be converted to Parquet format
and then loaded with load_dataset.
"""

import os
import sys
from pathlib import Path

from datasets import load_from_disk


def convert_to_parquet(input_path: str, output_path: str, shard_size: int = 100000):
    """
    Convert a dataset saved with save_to_disk to Parquet format that can be loaded with load_dataset.
    
    Args:
        input_path: Path to the dataset saved with save_to_disk
        output_path: Path where to save the Parquet files
        shard_size: Number of examples per Parquet file
    """
    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(input_path)
    
    print(f"Converting to Parquet format...")
    print(f"Dataset has {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Shard the dataset and save as Parquet files
    num_shards = (len(dataset) + shard_size - 1) // shard_size
    
    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, len(dataset))
        
        # Get the shard
        shard = dataset.select(range(start_idx, end_idx))
        
        # Save as Parquet
        parquet_path = os.path.join(output_path, f"data_{i:06d}.parquet")
        shard.to_parquet(parquet_path)
        
        if (i + 1) % 10 == 0:
            print(f"Saved shard {i + 1}/{num_shards}")
    
    print(f"Successfully converted dataset to Parquet format at {output_path}")
    print(f"Created {num_shards} Parquet files")
    print(f"Each file contains approximately {shard_size} examples")
    
    # Create a simple dataset_info.json for better compatibility
    dataset_info = {
        "builder_name": "parquet",
        "citation": "",
        "config_name": "default",
        "dataset_name": os.path.basename(output_path),
        "dataset_size": len(dataset),
        "description": f"Parquet dataset converted from {input_path}",
        "download_checksums": {},
        "download_size": 0,
        "features": {
            "text": {"dtype": "string", "_type": "Value"}
        },
        "homepage": "",
        "license": "",
        "post_processed": None,
        "post_processing_size": None,
        "size_in_bytes": 0,
        "splits": {
            "train": {
                "name": "train",
                "num_bytes": 0,
                "num_examples": len(dataset),
                "shard_lengths": None,
                "dataset_name": os.path.basename(output_path)
            }
        },
        "supervised_keys": None,
        "task_templates": [],
        "version": {"version_str": "0.0.0", "description": None, "major": 0, "minor": 0, "patch": 0}
    }
    
    # Save dataset_info.json
    info_path = os.path.join(output_path, "dataset_info.json")
    with open(info_path, 'w') as f:
        import json
        json.dump(dataset_info, f, indent=2)
    
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_to_parquet.py <input_path> <output_path> [shard_size]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    shard_size = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
    
    if not os.path.exists(input_path):
        print(f"Error: Input path {input_path} does not exist")
        sys.exit(1)
    
    try:
        convert_to_parquet(input_path, output_path, shard_size)
        print("Success! You can now use the output path in FLAME training scripts.")
        print("Example usage:")
        print(f"  --training.dataset {output_path}")
        print("  or")
        print(f"  --training.dataset {output_path} --training.data_files '*.parquet'")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 