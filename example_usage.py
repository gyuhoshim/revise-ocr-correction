#!/usr/bin/env python3
"""
Example Usage Script for REVISE OCR Error Correction Framework

This script demonstrates how to use the REVISE framework for:
1. Data contamination (simulating OCR errors)
2. Training a model for OCR error correction
3. Basic inference with the trained model

Author: Gyuho Shim
"""

import json
import jsonlines
from contaminate import (
    deletion, insertion, segmentation, transposition, substitution,
    apply_contamination, process_jsonl_file
)


def create_sample_data():
    """Create sample clean text data for demonstration."""
    sample_texts = [
        {
            "id": "1",
            "text": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet."
        },
        {
            "id": "2", 
            "text": "Machine learning and artificial intelligence are transforming how we process and understand text data."
        },
        {
            "id": "3",
            "text": "Optical Character Recognition (OCR) technology converts images of text into machine-readable text formats."
        },
        {
            "id": "4",
            "text": "Data preprocessing is a crucial step in any machine learning pipeline, especially for natural language processing tasks."
        },
        {
            "id": "5",
            "text": "The REVISE framework addresses OCR error correction through synthetic data contamination and instruction-following approaches."
        }
    ]
    
    # Save sample data
    with jsonlines.open('sample_clean_data.jsonl', 'w') as writer:
        for item in sample_texts:
            writer.write(item)
    
    print("✅ Created sample_clean_data.jsonl with 5 sample texts")


def demonstrate_contamination():
    """Demonstrate different types of OCR error contamination."""
    sample_text = "The quick brown fox jumps over the lazy dog."
    
    print("\n🔧 OCR Error Contamination Examples:")
    print(f"Original: {sample_text}")
    print("-" * 60)
    
    # Character-level errors
    print(f"Deletion:      {deletion(sample_text)}")
    print(f"Insertion:     {insertion(sample_text)}")
    print(f"Segmentation:  {segmentation(sample_text)}")
    print(f"Transposition: {transposition(sample_text)}")
    
    # Note: substitution requires mapping.json
    try:
        print(f"Substitution:  {substitution(sample_text)}")
    except FileNotFoundError:
        print("Substitution:  (requires mapping.json file)")
    
    # Layout contamination
    print(f"Layout (Type II): {apply_contamination(sample_text, 'Type II')}")


def create_contaminated_dataset():
    """Create a contaminated dataset from clean data."""
    print("\n📊 Creating contaminated dataset...")
    
    # First create sample data if it doesn't exist
    try:
        with open('sample_clean_data.jsonl', 'r') as f:
            pass
    except FileNotFoundError:
        create_sample_data()
    
    # Process the data to add contamination
    process_jsonl_file(
        input_file='sample_clean_data.jsonl',
        output_file='sample_contaminated_data.jsonl',
        total_data_lines=5
    )
    
    print("✅ Created sample_contaminated_data.jsonl")
    
    # Show examples
    print("\n📋 Contaminated Data Examples:")
    with jsonlines.open('sample_contaminated_data.jsonl') as reader:
        for i, item in enumerate(reader):
            if i < 2:  # Show first 2 examples
                print(f"\nExample {i+1}:")
                print(f"Original:     {item['text'][:80]}...")
                print(f"Contaminated: {item['contaminated_text'][:80]}...")


def show_training_command():
    """Show example training commands."""
    print("\n🚀 Training Commands:")
    print("-" * 40)
    
    print("Basic training:")
    print("python train.py \\")
    print("    --base_model='meta-llama/Llama-3.2-1B-Instruct' \\")
    print("    --data_path='sample_contaminated_data.jsonl' \\")
    print("    --output_dir='./sample_checkpoint' \\")
    print("    --per_device_train_batch_size=1 \\")
    print("    --num_epochs=1 \\")
    print("    --learning_rate=3e-4")
    
    print("\nMulti-GPU training:")
    print("bash train.sh")


def main():
    """Main demonstration function."""
    print("🎯 REVISE Framework - Example Usage")
    print("=" * 50)
    
    # Step 1: Create sample data
    create_sample_data()
    
    # Step 2: Demonstrate contamination
    demonstrate_contamination()
    
    # Step 3: Create contaminated dataset
    create_contaminated_dataset()
    
    # Step 4: Show training commands
    show_training_command()
    
    print("\n✨ Example usage complete!")
    print("\nNext steps:")
    print("1. Review the generated sample files")
    print("2. Modify the contamination parameters as needed")
    print("3. Run training with your own data")
    print("4. Evaluate the trained model on your OCR correction tasks")


if __name__ == "__main__":
    main()
