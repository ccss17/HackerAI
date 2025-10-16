#!/usr/bin/env python3
"""
Analyze instruction lengths in all datasets (test.json, train.json, val.json) 
to determine optimal sequence length for evaluation.
"""

import json
import statistics
import os
from pathlib import Path

def analyze_instruction_lengths():
    """Analyze instruction lengths in all datasets."""
    
    datasets = ['data/test.json', 'data/train.json', 'data/val.json']
    all_instruction_lengths = []
    dataset_stats = {}
    
    print("🔍 Analyzing Instruction Lengths in All Datasets")
    print("="*80)
    
    for dataset_path in datasets:
        if not os.path.exists(dataset_path):
            print(f"⚠️  Dataset not found: {dataset_path}")
            continue
            
        print(f"\n📊 Analyzing: {dataset_path}")
        print("-" * 50)
        
        # Load the dataset
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Extract instruction lengths
        instruction_lengths = []
        for item in data:
            instruction = item.get('instruction', '')
            instruction_lengths.append(len(instruction))
        
        # Calculate statistics
        min_length = min(instruction_lengths)
        max_length = max(instruction_lengths)
        mean_length = statistics.mean(instruction_lengths)
        median_length = statistics.median(instruction_lengths)
        
        # Calculate percentiles
        sorted_lengths = sorted(instruction_lengths)
        p90 = sorted_lengths[int(len(sorted_lengths) * 0.9)]
        p95 = sorted_lengths[int(len(sorted_lengths) * 0.95)]
        p99 = sorted_lengths[int(len(sorted_lengths) * 0.99)]
        
        print(f"  📈 Statistics for {len(data)} samples:")
        print(f"    - Min: {min_length} characters")
        print(f"    - Max: {max_length} characters")
        print(f"    - Mean: {mean_length:.1f} characters")
        print(f"    - Median: {median_length:.1f} characters")
        print(f"    - 90th percentile: {p90} characters")
        print(f"    - 95th percentile: {p95} characters")
        print(f"    - 99th percentile: {p99} characters")
        
        # Token estimates (1 token ≈ 4 characters)
        print(f"  🎯 Token Count Estimates (1 token ≈ 4 characters):")
        print(f"    - Min: {min_length // 4} tokens")
        print(f"    - Max: {max_length // 4} tokens")
        print(f"    - Mean: {mean_length // 4} tokens")
        print(f"    - Median: {median_length // 4} tokens")
        print(f"    - 90th percentile: {p90 // 4} tokens")
        print(f"    - 95th percentile: {p95 // 4} tokens")
        print(f"    - 99th percentile: {p99 // 4} tokens")
        
        # Store for combined analysis
        all_instruction_lengths.extend(instruction_lengths)
        dataset_stats[dataset_path] = {
            'count': len(data),
            'min': min_length,
            'max': max_length,
            'mean': mean_length,
            'median': median_length,
            'p90': p90,
            'p95': p95,
            'p99': p99
        }
    
    # Combined analysis
    if all_instruction_lengths:
        print(f"\n🎯 COMBINED ANALYSIS (All Datasets)")
        print("="*80)
        
        min_length = min(all_instruction_lengths)
        max_length = max(all_instruction_lengths)
        mean_length = statistics.mean(all_instruction_lengths)
        median_length = statistics.median(all_instruction_lengths)
        
        sorted_lengths = sorted(all_instruction_lengths)
        p90 = sorted_lengths[int(len(sorted_lengths) * 0.9)]
        p95 = sorted_lengths[int(len(sorted_lengths) * 0.95)]
        p99 = sorted_lengths[int(len(sorted_lengths) * 0.99)]
        
        print(f"📊 Combined Statistics ({len(all_instruction_lengths)} total samples):")
        print(f"  - Min: {min_length} characters")
        print(f"  - Max: {max_length} characters")
        print(f"  - Mean: {mean_length:.1f} characters")
        print(f"  - Median: {median_length:.1f} characters")
        print(f"  - 90th percentile: {p90} characters")
        print(f"  - 95th percentile: {p95} characters")
        print(f"  - 99th percentile: {p99} characters")
        
        print(f"\n🎯 Token Count Estimates (1 token ≈ 4 characters):")
        print(f"  - Min: {min_length // 4} tokens")
        print(f"  - Max: {max_length // 4} tokens")
        print(f"  - Mean: {mean_length // 4} tokens")
        print(f"  - Median: {median_length // 4} tokens")
        print(f"  - 90th percentile: {p90 // 4} tokens")
        print(f"  - 95th percentile: {p95 // 4} tokens")
        print(f"  - 99th percentile: {p99 // 4} tokens")
        
        # Sequence length recommendations
        print(f"\n🚀 SEQUENCE LENGTH RECOMMENDATIONS")
        print("="*80)
        
        # For 99% coverage
        recommended_seq_length_99 = p99 // 4 + 200  # Add buffer for response
        print(f"  📏 For 99% coverage: {recommended_seq_length_99} tokens")
        
        # For 95% coverage  
        recommended_seq_length_95 = p95 // 4 + 200
        print(f"  📏 For 95% coverage: {recommended_seq_length_95} tokens")
        
        # For 90% coverage
        recommended_seq_length_90 = p90 // 4 + 200
        print(f"  📏 For 90% coverage: {recommended_seq_length_90} tokens")
        
        print(f"\n⚡ SPEED vs COVERAGE ANALYSIS")
        print("="*80)
        
        # Coverage analysis for different sequence lengths (including lower than 1024)
        seq_lengths = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
        for seq_len in seq_lengths:
            coverage = 100 * sum(1 for x in all_instruction_lengths if x // 4 <= seq_len - 200) / len(all_instruction_lengths)
            
            # More granular batch size estimates
            if seq_len <= 256:
                batch_size_estimate = 128
                speed_estimate = "8x faster"
            elif seq_len <= 384:
                batch_size_estimate = 112
                speed_estimate = "7x faster"
            elif seq_len <= 512:
                batch_size_estimate = 96
                speed_estimate = "6x faster"
            elif seq_len <= 768:
                batch_size_estimate = 80
                speed_estimate = "5x faster"
            elif seq_len <= 1024:
                batch_size_estimate = 64
                speed_estimate = "4x faster"
            elif seq_len <= 1536:
                batch_size_estimate = 48
                speed_estimate = "3x faster"
            elif seq_len <= 2048:
                batch_size_estimate = 32
                speed_estimate = "2x faster"
            elif seq_len <= 3072:
                batch_size_estimate = 24
                speed_estimate = "1.5x faster"
            elif seq_len <= 4096:
                batch_size_estimate = 16
                speed_estimate = "1x baseline"
            elif seq_len <= 6144:
                batch_size_estimate = 12
                speed_estimate = "0.75x slower"
            else:
                batch_size_estimate = 8
                speed_estimate = "0.5x slower"
            
            print(f"  📊 {seq_len:4d} tokens: ~{coverage:5.1f}% coverage, batch_size~{batch_size_estimate:2d}, {speed_estimate}")
        
        print(f"\n🎯 RECOMMENDED CONFIGURATIONS")
        print("="*80)
        
        # Find the optimal sequence length based on coverage
        optimal_configs = []
        
        # Check different sequence lengths for optimal configurations
        for seq_len in [256, 384, 512, 768, 1024, 1536, 2048]:
            coverage = 100 * sum(1 for x in all_instruction_lengths if x // 4 <= seq_len - 200) / len(all_instruction_lengths)
            if coverage >= 99.0:  # 99%+ coverage
                if seq_len <= 256:
                    batch_size = 128
                    speed = "8x faster"
                elif seq_len <= 384:
                    batch_size = 112
                    speed = "7x faster"
                elif seq_len <= 512:
                    batch_size = 96
                    speed = "6x faster"
                elif seq_len <= 768:
                    batch_size = 80
                    speed = "5x faster"
                elif seq_len <= 1024:
                    batch_size = 64
                    speed = "4x faster"
                elif seq_len <= 1536:
                    batch_size = 48
                    speed = "3x faster"
                else:
                    batch_size = 32
                    speed = "2x faster"
                
                optimal_configs.append((seq_len, batch_size, speed, coverage))
        
        if optimal_configs:
            print("  🚀 SPEED-OPTIMIZED CONFIGURATIONS (99%+ coverage):")
            for seq_len, batch_size, speed, coverage in optimal_configs:
                print(f"    📊 {seq_len} tokens: batch_size={batch_size}, {speed} ({coverage:.1f}% coverage)")
                print(f"       Command: pixi run python train/evaluate_baseline.py --batch_size {batch_size} --max_seq_length {seq_len}")
                print()
        
        # Show the fastest option
        if optimal_configs:
            fastest = min(optimal_configs, key=lambda x: x[0])
            print(f"  🏆 FASTEST OPTION: {fastest[0]} tokens, batch_size={fastest[1]}, {fastest[2]}")
            print(f"     Command: pixi run python train/evaluate_baseline.py --batch_size {fastest[1]} --max_seq_length {fastest[0]}")
        
        # Show some examples of long instructions
        print(f"\n📝 LONGEST INSTRUCTIONS (First 3)")
        print("="*80)
        
        # Find longest instructions from all datasets
        all_data = []
        for dataset_path in datasets:
            if os.path.exists(dataset_path):
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                    all_data.extend(data)
        
        sorted_data = sorted(all_data, key=lambda x: len(x.get('instruction', '')), reverse=True)
        for i, item in enumerate(sorted_data[:3]):
            instruction = item.get('instruction', '')
            print(f"{i+1}. Length: {len(instruction)} characters ({len(instruction) // 4} tokens)")
            print(f"   Preview: {instruction[:150]}...")
            print()

if __name__ == "__main__":
    analyze_instruction_lengths()
