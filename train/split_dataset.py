#!/usr/bin/env python3
"""
Dataset splitting script for cybersecurity dataset.
Splits the preprocessed dataset into train/val/test (8:1:1).
"""

import json
import random
from pathlib import Path

def split_dataset(input_file, output_dir):
    """
    Split dataset into train/val/test with 8:1:1 ratio.
    
    Args:
        input_file: Path to the input JSON file
        output_dir: Directory to save split datasets
    """
    print(f"Loading dataset from {input_file}...")
    
    # Load the dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Shuffle the data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split indices
    total_samples = len(shuffled_data)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size
    
    print(f"Split sizes:")
    print(f"  Train: {train_size} samples ({train_size/total_samples*100:.1f}%)")
    print(f"  Val:   {val_size} samples ({val_size/total_samples*100:.1f}%)")
    print(f"  Test:  {test_size} samples ({test_size/total_samples*100:.1f}%)")
    
    # Split the data
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        output_file = output_path / f"{split_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {split_name} split: {output_file} ({len(split_data)} samples)")
    
    print("Dataset splitting completed successfully!")
    
    return {
        'train': len(train_data),
        'val': len(val_data),
        'test': len(test_data)
    }

if __name__ == "__main__":
    # Input file path
    input_file = "/data/cyber-llm-instruct-py/dataset_creation/final_dataset_preprocessed/final_cybersecurity_dataset_preprocessed_20251013_044718.json"
    
    # Output directory
    output_dir = "/data/cyber-llm-instruct-py/train/data"
    
    # Run the splitting
    split_stats = split_dataset(input_file, output_dir)
    
    print("\nSplit statistics:")
    for split, count in split_stats.items():
        print(f"  {split}: {count} samples")
