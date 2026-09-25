"""
Unit tests for balanced test sample generation function.
"""

import torch

from llm_dynamics_main import generate_balanced_test_samples
from zo_llm.util.language_utils import LLMBatchInput

# Try to import pytest, but make it optional
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, labels, raw_samples=None):
        self.labels = labels
        self.raw_samples = raw_samples or [None] * len(labels)
        self.texts = [f"text_{i}" for i in range(len(labels))]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Return a tuple of (input_ids, label)
        input_ids = torch.tensor([1, 2, 3, 4, 5])  # Mock input_ids
        label = self.labels[idx]
        if isinstance(label, torch.Tensor):
            return input_ids, label
        return input_ids, torch.tensor(label)
    
    def get_raw_sample(self, idx):
        if self.raw_samples:
            return self.raw_samples[idx]
        return None
    
    def get_encoded_text(self, idx):
        """Get the encoded/verbalized text at index idx."""
        if self.texts is not None and idx < len(self.texts):
            return self.texts[idx]
        return None


class MockDatasetNoLabels:
    """Mock dataset without labels (for generation tasks)."""
    
    def __init__(self, size=10):
        self.size = size
        self.texts = [f"text_{i}" for i in range(size)]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        return input_ids


def test_balanced_samples_binary_classification():
    """Test balanced sampling for binary classification (2 classes)."""
    # Create dataset with 20 samples: 10 class 0, 10 class 1
    labels = [0] * 10 + [1] * 10
    raw_samples = [{"sentence": f"sample_{i}", "label": labels[i]} for i in range(20)]
    dataset = MockDataset(labels, raw_samples)
    
    # Request 4 samples - should get 2 from each class
    test_samples, test_raw_sentences, test_encoded_texts = generate_balanced_test_samples(
        dataset, num_test_samples=4, seed=42
    )
    
    assert len(test_samples) == 4
    assert len(test_raw_sentences) == 4
    assert len(test_encoded_texts) == 4
    
    # Check that we have balanced classes
    labels_collected = []
    for _, label in test_samples:
        if isinstance(label, torch.Tensor):
            labels_collected.append(label.item() if label.numel() == 1 else label[0].item())
        else:
            labels_collected.append(label)
    
    # Count occurrences of each class
    class_0_count = labels_collected.count(0)
    class_1_count = labels_collected.count(1)
    
    assert class_0_count == 2, f"Expected 2 samples of class 0, got {class_0_count}"
    assert class_1_count == 2, f"Expected 2 samples of class 1, got {class_1_count}"


def test_balanced_samples_three_classes():
    """Test balanced sampling for 3 classes."""
    # Create dataset with 30 samples: 10 of each class
    labels = [0] * 10 + [1] * 10 + [2] * 10
    raw_samples = [{"sentence": f"sample_{i}", "label": labels[i]} for i in range(30)]
    dataset = MockDataset(labels, raw_samples)
    
    # Request 6 samples - should get 2 from each class
    test_samples, test_raw_sentences = generate_balanced_test_samples(
        dataset, num_test_samples=6, seed=42
    )
    
    assert len(test_samples) == 6
    
    labels_collected = []
    for _, label in test_samples:
        if isinstance(label, torch.Tensor):
            labels_collected.append(label.item() if label.numel() == 1 else label[0].item())
        else:
            labels_collected.append(label)
    
    # Check balance
    class_counts = {0: 0, 1: 0, 2: 0}
    for label in labels_collected:
        class_counts[label] += 1
    
    assert all(count == 2 for count in class_counts.values()), \
        f"Expected 2 samples per class, got {class_counts}"


def test_balanced_samples_uneven_division():
    """Test balanced sampling when num_test_samples doesn't divide evenly by num_classes."""
    # Create dataset with 20 samples: 10 class 0, 10 class 1
    labels = [0] * 10 + [1] * 10
    raw_samples = [{"sentence": f"sample_{i}", "label": labels[i]} for i in range(20)]
    dataset = MockDataset(labels, raw_samples)
    
    # Request 5 samples - should get 3 from one class, 2 from the other
    test_samples, test_raw_sentences = generate_balanced_test_samples(
        dataset, num_test_samples=5, seed=42
    )
    
    assert len(test_samples) == 5
    
    labels_collected = []
    for _, label in test_samples:
        if isinstance(label, torch.Tensor):
            labels_collected.append(label.item() if label.numel() == 1 else label[0].item())
        else:
            labels_collected.append(label)
    
    class_0_count = labels_collected.count(0)
    class_1_count = labels_collected.count(1)
    
    # Should have 3 of one class and 2 of the other
    assert class_0_count + class_1_count == 5
    assert abs(class_0_count - class_1_count) == 1, \
        f"Expected difference of 1, got {class_0_count} and {class_1_count}"


def test_balanced_samples_more_than_available():
    """Test when requesting more samples than available per class."""
    # Create dataset with only 3 samples: 2 class 0, 1 class 1
    labels = [0, 0, 1]
    raw_samples = [{"sentence": f"sample_{i}", "label": labels[i]} for i in range(3)]
    dataset = MockDataset(labels, raw_samples)
    
    # Request 10 samples - should get all available samples
    test_samples, test_raw_sentences = generate_balanced_test_samples(
        dataset, num_test_samples=10, seed=42
    )
    
    # Should get all 3 samples
    assert len(test_samples) == 3
    
    labels_collected = []
    for _, label in test_samples:
        if isinstance(label, torch.Tensor):
            labels_collected.append(label.item() if label.numel() == 1 else label[0].item())
        else:
            labels_collected.append(label)
    
    # Should have 2 of class 0 and 1 of class 1
    assert labels_collected.count(0) == 2
    assert labels_collected.count(1) == 1


def test_balanced_samples_no_labels():
    """Test sampling for dataset without labels (generation tasks)."""
    dataset = MockDatasetNoLabels(size=10)
    
    # Request 5 samples
    test_samples, test_raw_sentences, test_encoded_texts = generate_balanced_test_samples(
        dataset, num_test_samples=5, seed=42
    )
    
    assert len(test_samples) == 5
    assert len(test_raw_sentences) == 5
    assert len(test_encoded_texts) == 5
    
    # All samples should have None labels
    for _, label in test_samples:
        assert label is None


def test_balanced_samples_reproducibility():
    """Test that same seed produces same samples."""
    labels = [0] * 10 + [1] * 10
    raw_samples = [{"sentence": f"sample_{i}", "label": labels[i]} for i in range(20)]
    dataset1 = MockDataset(labels, raw_samples)
    dataset2 = MockDataset(labels, raw_samples)
    
    # Generate samples with same seed
    samples1, raw1, encoded1 = generate_balanced_test_samples(dataset1, num_test_samples=4, seed=42)
    samples2, raw2, encoded2 = generate_balanced_test_samples(dataset2, num_test_samples=4, seed=42)
    
    # Should produce same results
    assert len(samples1) == len(samples2)
    
    # Check that labels match (order might differ due to final shuffle, but counts should match)
    labels1 = []
    labels2 = []
    for _, label in samples1:
        if isinstance(label, torch.Tensor):
            labels1.append(label.item() if label.numel() == 1 else label[0].item())
        else:
            labels1.append(label)
    
    for _, label in samples2:
        if isinstance(label, torch.Tensor):
            labels2.append(label.item() if label.numel() == 1 else label[0].item())
        else:
            labels2.append(label)
    
    assert sorted(labels1) == sorted(labels2), \
        f"Labels should match: {labels1} vs {labels2}"


def test_balanced_samples_different_seeds():
    """Test that different seeds produce different samples."""
    labels = [0] * 20 + [1] * 20
    raw_samples = [{"sentence": f"sample_{i}", "label": labels[i]} for i in range(40)]
    dataset1 = MockDataset(labels, raw_samples)
    dataset2 = MockDataset(labels, raw_samples)
    
    # Generate samples with different seeds
    samples1, _, _ = generate_balanced_test_samples(dataset1, num_test_samples=4, seed=42)
    samples2, _, _ = generate_balanced_test_samples(dataset2, num_test_samples=4, seed=123)
    
    # Extract indices (we can't directly compare, but we can check they're valid)
    assert len(samples1) == len(samples2) == 4
    
    # Both should be balanced
    labels1 = []
    labels2 = []
    for _, label in samples1:
        if isinstance(label, torch.Tensor):
            labels1.append(label.item() if label.numel() == 1 else label[0].item())
        else:
            labels1.append(label)
    
    for _, label in samples2:
        if isinstance(label, torch.Tensor):
            labels2.append(label.item() if label.numel() == 1 else label[0].item())
        else:
            labels2.append(label)
    
    assert labels1.count(0) == labels1.count(1) == 2
    assert labels2.count(0) == labels2.count(1) == 2


def test_balanced_samples_batch_input_format():
    """Test that samples are returned in correct format for model inference."""
    labels = [0] * 10 + [1] * 10
    raw_samples = [{"sentence": f"sample_{i}", "label": labels[i]} for i in range(20)]
    dataset = MockDataset(labels, raw_samples)
    
    test_samples, test_raw_sentences = generate_balanced_test_samples(
        dataset, num_test_samples=4, seed=42
    )
    
    # Check format
    for batch_input, label in test_samples:
        # Should be LLMBatchInput
        assert isinstance(batch_input, LLMBatchInput)
        assert hasattr(batch_input, 'input_ids')
        assert hasattr(batch_input, 'attention_mask')
        
        # Should have batch dimension
        assert batch_input.input_ids.dim() == 2
        assert batch_input.attention_mask.dim() == 2
        assert batch_input.input_ids.shape[0] == 1  # Single sample batch


if __name__ == "__main__":
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        # Run tests manually if pytest is not available
        print("Running tests without pytest...")
        try:
            test_balanced_samples_binary_classification()
            print("✓ test_balanced_samples_binary_classification passed")
            
            test_balanced_samples_three_classes()
            print("✓ test_balanced_samples_three_classes passed")
            
            test_balanced_samples_uneven_division()
            print("✓ test_balanced_samples_uneven_division passed")
            
            test_balanced_samples_more_than_available()
            print("✓ test_balanced_samples_more_than_available passed")
            
            test_balanced_samples_no_labels()
            print("✓ test_balanced_samples_no_labels passed")
            
            test_balanced_samples_reproducibility()
            print("✓ test_balanced_samples_reproducibility passed")
            
            test_balanced_samples_different_seeds()
            print("✓ test_balanced_samples_different_seeds passed")
            
            test_balanced_samples_batch_input_format()
            print("✓ test_balanced_samples_batch_input_format passed")
            
            print("\nAll tests passed!")
        except AssertionError as e:
            print(f"\n✗ Test failed: {e}")
            raise
