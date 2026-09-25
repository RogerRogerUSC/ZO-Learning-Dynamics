"""
Simple script to print test sample data without running experiments.
"""

import argparse
from pathlib import Path

import torch

from datasets import load_dataset as huggingface_load_dataset

from zo_llm.util import config_parser, data_utils
from zo_llm.util.language_utils import LM_DATASET_MAP, get_hf_tokenizer
from llm_dynamics_main import generate_balanced_test_samples


def print_test_samples(
    config_path: str = "text_classification/gaussian_opt.yaml",
    num_test_samples: int = 5,
):
    """
    Print test sample data without running experiments.
    
    Args:
        config_path: Path to config file
        num_test_samples: Number of test samples to print
    """
    # Parse config
    config = config_parser.parse_config(config_path)
    
    # Load raw HuggingFace dataset directly to see actual raw data
    print("Loading raw HuggingFace dataset directly...")
    dataset_name = LM_DATASET_MAP[config.dataset.value]
    if config.dataset.value == "sst2":
        raw_hf_dataset = huggingface_load_dataset(dataset_name, config.dataset.value)
    else:
        raw_hf_dataset = huggingface_load_dataset(dataset_name)
    
    raw_test_dataset = raw_hf_dataset["validation"]
    print(f"Raw HuggingFace dataset loaded. Total test samples: {len(raw_test_dataset)}")
    
    # Show first few raw samples directly from HuggingFace
    print("\n" + "=" * 80)
    print("RAW HUGGINGFACE DATASET SAMPLES (First 3)")
    print("=" * 80)
    for i in range(min(3, len(raw_test_dataset))):
        print(f"\nRaw Sample {i} from HuggingFace:")
        raw_entry = raw_test_dataset[i]
        print(f"  Type: {type(raw_entry)}")
        print(f"  Keys: {raw_entry.keys() if isinstance(raw_entry, dict) else 'N/A'}")
        print(f"  Full entry: {raw_entry}")
    
    # Load test dataset through the normal pipeline
    print("\nLoading test dataset through normal pipeline...")
    _, test_loader = data_utils.get_dataloaders(
        config, config.seed, config.get_hf_model_name()
    )
    test_dataset = test_loader.dataset
    
    # Generate balanced test samples
    print(f"Generating {num_test_samples} balanced test samples...")
    test_samples, test_raw_sentences, test_encoded_texts = generate_balanced_test_samples(
        test_dataset=test_dataset,
        num_test_samples=num_test_samples,
        seed=config.seed,
    )
    
    # Get tokenizer for decoding if needed
    hf_model_name = config.get_hf_model_name()
    tokenizer = get_hf_tokenizer(hf_model_name)
    
    # Print test samples
    print("\n" + "=" * 80)
    print("TEST SAMPLE DATA")
    print("=" * 80)
    
    for i in range(num_test_samples):
        print(f"\n{'='*80}")
        print(f"Test Sample {i + 1}/{num_test_samples}")
        print(f"{'='*80}")
        
        # Print raw sentence from stored raw_samples
        if i < len(test_raw_sentences) and test_raw_sentences[i] is not None:
            raw_sample = test_raw_sentences[i]
            print("\nRaw Sample (from test_dataset.get_raw_sample):")
            print(f"  Type: {type(raw_sample)}")
            if isinstance(raw_sample, dict):
                print(f"  Keys: {list(raw_sample.keys())}")
                # Extract raw sentence text based on common fields
                if "sentence" in raw_sample:
                    print(f"  Sentence: {raw_sample['sentence']}")
                elif "text" in raw_sample:
                    print(f"  Text: {raw_sample['text']}")
                elif "question1" in raw_sample and "question2" in raw_sample:
                    print(f"  Question 1: {raw_sample['question1']}")
                    print(f"  Question 2: {raw_sample['question2']}")
                elif "passage" in raw_sample and "question" in raw_sample:
                    print(f"  Passage: {raw_sample['passage']}")
                    print(f"  Question: {raw_sample['question']}")
                elif "premise" in raw_sample and "hypothesis" in raw_sample:
                    print(f"  Premise: {raw_sample['premise']}")
                    print(f"  Hypothesis: {raw_sample['hypothesis']}")
                else:
                    print(f"  Full raw sample: {raw_sample}")
                
                if "label" in raw_sample:
                    print(f"  Label: {raw_sample['label']}")
            else:
                print(f"  Full raw sample: {raw_sample}")
        
        # Print encoded/verbalized text if available
        if i < len(test_encoded_texts) and test_encoded_texts[i] is not None:
            print(f"\nEncoded/Verbalized Text:")
            print(f"  {test_encoded_texts[i]}")
        
        # Print input_ids (decoded back to text for readability)
        if i < len(test_samples):
            batch_input, label = test_samples[i]
            if hasattr(batch_input, 'input_ids'):
                input_ids = batch_input.input_ids[0]  # First (and only) sample in batch
                decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
                print(f"\nDecoded Input IDs:")
                print(f"  {decoded_text}")
                
                # Print token IDs for reference
                print(f"\nInput IDs (first 20 tokens):")
                print(f"  {input_ids[:20].tolist()}")
            
            if label is not None:
                if isinstance(label, torch.Tensor):
                    label_value = label.item() if label.numel() == 1 else label.tolist()
                else:
                    label_value = label
                print(f"\nLabel Tensor:")
                print(f"  {label_value}")
    
    print("\n" + "=" * 80)
    print("Done printing test samples")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print test sample data without running experiments")
    parser.add_argument(
        "--config-path",
        type=str,
        default="text_classification/gaussian_opt.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--num-test-samples",
        type=int,
        default=5,
        help="Number of test samples to print",
    )
    
    args = parser.parse_args()
    
    print_test_samples(
        config_path=args.config_path,
        num_test_samples=args.num_test_samples,
    )
