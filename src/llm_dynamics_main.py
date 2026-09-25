"""
Experiment comparing SGD vs ZO optimization for LLM on SST2 dataset.
Tracks logits changes for 5 test samples over training iterations.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from zo_llm.llm_trainer import LLM_trainer
from zo_llm.util import config_parser, data_utils, model_utils, prepare_settings
from zo_llm.util.language_utils import get_hf_tokenizer, LM_TEMPLATE_MAP, LLMBatchInput
from zo_llm.util.metrics import Metric
from zo_llm.zo_optim import ZOOptimizer


def generate_balanced_test_samples(
    test_dataset: torch.utils.data.Dataset,
    num_test_samples: int,
    seed: int | None = None,
) -> tuple[list[tuple[Any, Any]], list[dict[str, Any] | None], list[str | None]]:
    """
    Generate balanced test samples from a test dataset.
    
    This function ensures that each class is represented equally (or as equally as possible)
    in the selected test samples. For example, if num_test_samples=4 and there are 2 classes,
    it will select 2 samples from each class.
    
    Args:
        test_dataset: The test dataset (should have labels attribute for classification tasks)
        num_test_samples: Total number of test samples to generate
        seed: Random seed for reproducibility (optional)
    
    Returns:
        A tuple of (test_samples, test_raw_sentences, test_encoded_texts) where:
        - test_samples: List of tuples (batch_input, label) ready for model inference
        - test_raw_sentences: List of raw sample dictionaries (or None if not available)
        - test_encoded_texts: List of encoded/verbalized text strings (or None if not available)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Check if dataset has labels (classification task)
    has_labels = hasattr(test_dataset, 'labels') and test_dataset.labels is not None
    
    if has_labels:
        # Get all unique labels
        labels = test_dataset.labels
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        unique_labels = sorted(set(labels))
        num_classes = len(unique_labels)
        
        # Group indices by label
        label_to_indices = {label: [] for label in unique_labels}
        for idx in range(len(test_dataset)):
            label = labels[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            label_to_indices[label].append(idx)
        
        # Calculate samples per class (balanced)
        samples_per_class = num_test_samples // num_classes
        remainder = num_test_samples % num_classes
        
        # Select indices ensuring balance
        selected_indices = []
        for i, label in enumerate(unique_labels):
            indices_for_label = label_to_indices[label]
            # Shuffle to get random samples
            shuffled_indices = indices_for_label.copy()
            random.shuffle(shuffled_indices)
            
            # Take samples_per_class + 1 for first 'remainder' classes
            num_to_take = samples_per_class + (1 if i < remainder else 0)
            selected_indices.extend(shuffled_indices[:num_to_take])
        
        # Shuffle the final selection to mix classes
        random.shuffle(selected_indices)
    else:
        # For generation tasks or datasets without labels, just take first N samples
        selected_indices = list(range(min(num_test_samples, len(test_dataset))))
        random.shuffle(selected_indices)
    
    # Collect samples
    test_samples = []
    test_raw_sentences = []
    test_encoded_texts = []
    
    for idx in selected_indices:
        # Get the sample from dataset
        sample_data = test_dataset[idx]
        
        # Handle different return formats
        if isinstance(sample_data, tuple):
            sample_input_ids, sample_label = sample_data
        else:
            sample_input_ids = sample_data
            sample_label = None
        
        # Create single-sample batch input
        if isinstance(sample_input_ids, torch.Tensor):
            single_input = LLMBatchInput(
                input_ids=sample_input_ids.unsqueeze(0),
                attention_mask=torch.ones_like(sample_input_ids.unsqueeze(0))
            )
        else:
            # Handle case where input_ids might already be a batch
            single_input = sample_input_ids
        
        # Handle label format
        if sample_label is not None:
            if isinstance(sample_label, torch.Tensor):
                single_label = sample_label.unsqueeze(0) if sample_label.dim() == 0 else sample_label
            else:
                single_label = sample_label
        else:
            single_label = None
        
        test_samples.append((single_input, single_label))
        
        # Get raw sentence from dataset
        if hasattr(test_dataset, 'get_raw_sample'):
            raw_sample = test_dataset.get_raw_sample(idx)
            test_raw_sentences.append(raw_sample)
        else:
            test_raw_sentences.append(None)
        
        # Get encoded/verbalized text from dataset
        if hasattr(test_dataset, 'get_encoded_text'):
            encoded_text = test_dataset.get_encoded_text(idx)
            test_encoded_texts.append(encoded_text)
        else:
            test_encoded_texts.append(None)
    
    return test_samples, test_raw_sentences, test_encoded_texts


def get_logits_for_test_samples(
    model: nn.Module,
    test_samples: list[tuple[Any, Any]],
    model_inference: Callable,
    device: torch.device,
    torch_dtype: torch.dtype,
    verbalizer_id_list: list[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Extract logits and probabilities for the output tokens (verbalizer tokens) for test samples.
    Returns a tuple of (logits_list, probabilities_list), one per sample.
    Each sample is already a single-sample batch.
    """
    model.eval()
    logits_list = []
    probabilities_list = []
    
    with torch.no_grad():
        for batch_inputs, labels in test_samples:
            if device != torch.device("cpu") or torch_dtype != torch.float32:
                batch_inputs = batch_inputs.to(device, torch_dtype)
            
            # Get model output
            output = model_inference(model, batch_inputs)
            logits = output.logits  # Shape: [1, seq_len, vocab_size] for single sample
            
            # Extract logits for last token position and verbalizer tokens
            # Shape: [1, num_verbalizer_tokens], squeeze to [num_verbalizer_tokens]
            last_token_logits = logits[:, -1, verbalizer_id_list].squeeze(0)
            
            # Compute probabilities using softmax
            probabilities = torch.nn.functional.softmax(last_token_logits, dim=0)
            
            logits_list.append(last_token_logits.cpu().clone())
            probabilities_list.append(probabilities.cpu().clone())
    
    return logits_list, probabilities_list


def setup_sgd_trainer(
    config: config_parser.MyConfig,
    device: torch.device,
    torch_dtype: torch.dtype,
    train_loader: Any,
) -> tuple[LLM_trainer, nn.Module]:
    """
    Setup trainer with SGD optimizer instead of ZO optimizer.
    Returns trainer and model (we need model separately for logits extraction).
    """
    model_inferences, metrics = prepare_settings.get_model_inferences_and_metrics(
        config.dataset, config
    )
    
    # Create model
    model = prepare_settings.get_model(
        dataset=config.dataset, model_setting=config, seed=config.seed
    ).to(device)
    
    # Create a custom trainer that uses SGD
    trainer = SGD_Trainer(device=device, dataloader=train_loader, torch_dtype=torch_dtype)
    trainer.set_model_and_criterion(
        model,
        model_inferences.test_inference,
        metrics.test_loss,
        metrics.test_acc,
        lr=config.lr,
    )
    
    return trainer, model


class SGD_Trainer(LLM_trainer):
    """
    Trainer that uses SGD optimizer instead of ZO optimizer.
    """
    
    def __init__(self, device, torch_dtype, dataloader):
        super().__init__(device, torch_dtype, dataloader)
        self.optimizer = None
    
    def set_model_and_criterion(
        self,
        model: torch.nn.Module,
        model_inference: Callable[[torch.nn.Module, Any], torch.Tensor],
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        accuracy_func,
        lr: float,
    ) -> None:
        self.model = model
        self.model_inference = model_inference
        self.criterion = criterion
        self.accuracy_func = accuracy_func
        
        # Setup SGD optimizer
        trainable_params = list(model_utils.get_trainable_model_parameters(model))
        self.optimizer = torch.optim.SGD(trainable_params, lr=lr)
    
    def train_one_step(self, iteration: int) -> tuple[float, float]:
        train_loss = Metric("Train loss")
        train_acc = Metric("Train acc")
        
        self.model.train()  # Use training mode for SGD
        batch_inputs, labels = next(self.data_iterator)
        
        if self.device != torch.device("cpu") or self.torch_dtype != torch.float32:
            batch_inputs = batch_inputs.to(self.device, self.torch_dtype)
            if isinstance(labels, torch.Tensor):
                labels = labels.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        pred = self.model_inference(self.model, batch_inputs)
        loss = self.criterion(pred, labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Calculate metrics
        train_loss.update(loss.item())
        train_acc.update(self.accuracy_func(pred, labels).item())
        
        return train_loss.avg, train_acc.avg


def save_results(results: dict, output_dir: Path, config_path: str):
    """
    Save logits and probability history to a single JSON file.
    
    Args:
        results: Dictionary containing logits and probability history
        output_dir: Directory to save results to
        config_path: Original config path for reference
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = len(results["sgd_logits_history"])
    
    # Convert torch tensors to lists for JSON serialization
    def tensor_list_to_list(tensor_list):
        """Convert list of torch tensors to list of lists."""
        return [t.tolist() if isinstance(t, torch.Tensor) else t.tolist() if isinstance(t, np.ndarray) else list(t) for t in tensor_list]
    
    # Build combined results dictionary
    combined_results = {
        "config_path": config_path,
        "num_test_samples": num_samples,
        "verbalizer_id_list": results["verbalizer_id_list"],
        "samples": []
    }
    
    # Process each sample
    for sample_idx in range(num_samples):
        sample_data = {
            "sample_idx": sample_idx,
            "sgd_logits_history": tensor_list_to_list(results["sgd_logits_history"][sample_idx]),
            "sgd_probs_history": tensor_list_to_list(results["sgd_probs_history"][sample_idx]),
            "zo_logits_history": tensor_list_to_list(results["zo_logits_history"][sample_idx]),
            "zo_probs_history": tensor_list_to_list(results["zo_probs_history"][sample_idx]),
        }
        
        # Add raw sentence if available
        if "test_raw_sentences" in results and sample_idx < len(results["test_raw_sentences"]):
            sample_data["raw_sentence"] = results["test_raw_sentences"][sample_idx]
        
        # Add encoded text if available
        if "test_encoded_texts" in results and sample_idx < len(results["test_encoded_texts"]):
            sample_data["encoded_text"] = results["test_encoded_texts"][sample_idx]
        
        combined_results["samples"].append(sample_data)
    
    # Add test accuracy trajectory
    combined_results["test_accuracy"] = {
        "eval_at_iterations": results["eval_iterations_log"],
        "sgd_acc_history": results["sgd_acc_history"],
        "zo_acc_history": results["zo_acc_history"],
        "sgd_loss_history": results["sgd_loss_history"],
        "zo_loss_history": results["zo_loss_history"],
    }

    # Generate filename with datetime (dd_mm_yyyy_hh_mm format)
    now = datetime.now()
    datetime_str = now.strftime("%d_%m_%Y_%H_%M")
    output_file = output_dir / f"results_{datetime_str}.json"
    
    # Save to single JSON file
    with open(output_file, "w") as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def run_experiment(
    config_path: str = "text_classification/gaussian_opt.yaml",
    num_test_samples: int = 5,
):
    """
    Run experiment comparing SGD vs ZO optimization.
    
    Args:
        config_path: Path to config file
        num_test_samples: Number of test samples to track
    """
    # Load raw config dict to check large_model format
    config_file_path = Path(__file__).parent / "zo_llm" / "configs" / config_path
    config_dict = config_parser.load_yaml_config(str(config_file_path))
    
    # Determine which model sizes to run
    # large_model can be a single string or a list of strings
    large_model_value = config_dict.get("large_model")
    if isinstance(large_model_value, list):
        # If it's a list, use all models in the list
        model_sizes = large_model_value
    elif isinstance(large_model_value, str):
        # If it's a single string, use just that model
        model_sizes = [large_model_value]
    else:
        # Default to all model sizes between 125m and 1.3b if not specified
        model_sizes = [config_parser.LargeModel.opt_125m.value, 
                      config_parser.LargeModel.opt_350m.value,
                      config_parser.LargeModel.opt_1p3b.value]
    
    # Determine output directory based on config path
    # Remove .yaml extension and create results folder structure
    config_name = Path(config_path).stem  # Get filename without extension
    config_dir = Path(config_path).parent  # Get directory path
    
    # Create results directory structure matching config path
    # __file__ is at src/llm_dynamics_main.py, so parent.parent is the repo root
    repo_root = Path(__file__).parent.parent
    results_base_dir = repo_root / "results" / config_dir / "01_21_2026"
    
    all_results = {}
    
    # Run experiment for each model size
    for model_size in model_sizes:
        print(f"\n{'='*80}")
        print(f"Running experiment for model: {model_size}")
        print(f"{'='*80}\n")
        
        # Parse config and override the model size
        config = config_parser.parse_config(config_path)
        config.large_model = config_parser.LargeModel(model_size)
        
        # All parameters (lr, iterations, eval_iterations, num_pert, pert_distribution) 
        # are read from config file
        
        result = run_single_experiment(config, config.iterations, num_test_samples, config.eval_iterations)
        all_results[model_size] = result
        
        # Save results for this model size
        # If multiple models, append model size to output directory name
        if len(model_sizes) > 1:
            output_dir = results_base_dir / f"{config_name}_{model_size}"
        else:
            output_dir = results_base_dir / config_name
        
        save_results(result, output_dir, config_path)
    
    return all_results


def run_single_experiment(
    config: config_parser.MyConfig,
    num_iterations: int,
    num_test_samples: int,
    eval_iterations: int,
):
    
    device = config.get_device()
    torch_dtype = config.get_torch_dtype()
    
    # Load data
    train_loader, test_loader = data_utils.get_dataloaders(
        config, config.seed, config.get_hf_model_name()
    )
    
    # Get test dataset to access raw samples
    test_dataset = test_loader.dataset
    
    # Get tokenizer for encoding
    hf_model_name = config.get_hf_model_name()
    tokenizer = get_hf_tokenizer(hf_model_name)
    
    # Generate balanced test samples
    test_samples, test_raw_sentences, test_encoded_texts = generate_balanced_test_samples(
        test_dataset=test_dataset,
        num_test_samples=num_test_samples,
        seed=config.seed,
    )
    
    # Get verbalizer tokens for logits extraction
    hf_model_name = config.get_hf_model_name()
    tokenizer = get_hf_tokenizer(hf_model_name)
    template = LM_TEMPLATE_MAP[config.dataset.value]()
    verbalizer_id_map = template.get_verbalizer_id(tokenizer)
    verbalizer_id_list = [verbalizer_id_map[i] for i in range(len(verbalizer_id_map))]
    
    # Get model inference function
    model_inferences, metrics = prepare_settings.get_model_inferences_and_metrics(
        config.dataset, config
    )
    model_inference_fn = model_inferences.test_inference
    
    # Setup SGD trainer
    sgd_trainer, sgd_model = setup_sgd_trainer(config, device, torch_dtype, train_loader)
    # Setup ZO trainer
    zo_trainer = LLM_trainer(device=device, dataloader=train_loader, torch_dtype=torch_dtype)
    zo_model = prepare_settings.get_model(
        dataset=config.dataset, model_setting=config, seed=config.seed
    ).to(device)
    zo_optimizer = ZOOptimizer.from_config(config, model=zo_model)
    zo_trainer.set_model_and_criterion(
        zo_model,
        model_inferences.test_inference,
        metrics.test_loss,
        metrics.test_acc,
        zo_optimizer,
    )
    
    # Print optimizer information
    print("=" * 80)
    print("Optimizer Information")
    print("=" * 80)
    print("SGD Optimizer:")
    sgd_optimizer = sgd_trainer.optimizer
    print(f"  Type: {type(sgd_optimizer).__name__}")
    print(f"  Learning rate: {sgd_optimizer.param_groups[0]['lr']}")
    if 'momentum' in sgd_optimizer.param_groups[0]:
        print(f"  Momentum: {sgd_optimizer.param_groups[0]['momentum']}")
    print(f"  Number of parameter groups: {len(sgd_optimizer.param_groups)}")
    print()
    print("ZO Optimizer:")
    print(f"  Type: {type(zo_optimizer).__name__}")
    print(f"  Learning rate: {zo_optimizer.lr}")
    print(f"  Number of perturbations: {zo_optimizer.num_pert}")
    print(f"  Perturbation distribution: {config.pert_distribution}")
    print(f"  Mu: {config.mu}")
    print(f"  Device: {zo_optimizer.device}")
    print("=" * 80)
    
    def extract_logits():
        """Extract logits and probabilities for both models and return them as lists."""
        sgd_logits, sgd_probs = get_logits_for_test_samples(
            sgd_model,
            test_samples,
            model_inference_fn,
            device,
            torch_dtype,
            verbalizer_id_list,
        )
        zo_logits, zo_probs = get_logits_for_test_samples(
            zo_model,
            test_samples,
            model_inference_fn,
            device,
            torch_dtype,
            verbalizer_id_list,
        )
        
        return sgd_logits, sgd_probs, zo_logits, zo_probs
    
    # Extract initial logits and probabilities (before training)
    sgd_initial_logits, sgd_initial_probs, zo_initial_logits, zo_initial_probs = extract_logits()
    
    # Initialize history storage
    sgd_logits_history = [[] for _ in range(num_test_samples)]
    sgd_probs_history = [[] for _ in range(num_test_samples)]
    zo_logits_history = [[] for _ in range(num_test_samples)]
    zo_probs_history = [[] for _ in range(num_test_samples)]
    
    # Store initial values
    for i in range(num_test_samples):
        sgd_logits_history[i].append(sgd_initial_logits[i].clone())
        sgd_probs_history[i].append(sgd_initial_probs[i].clone())
        zo_logits_history[i].append(zo_initial_logits[i].clone())
        zo_probs_history[i].append(zo_initial_probs[i].clone())
    
    # Initialize accuracy/loss history storage
    sgd_acc_history: list[float] = []
    zo_acc_history: list[float] = []
    sgd_loss_history: list[float] = []
    zo_loss_history: list[float] = []
    eval_iterations_log: list[int] = []

    # Initial evaluation (iteration 0, before training)
    sgd_eval_loss_0, sgd_eval_acc_0 = sgd_trainer.eval_model(test_loader, False)
    zo_eval_loss_0, zo_eval_acc_0 = zo_trainer.eval_model(test_loader, False)
    sgd_acc_history.append(sgd_eval_acc_0)
    zo_acc_history.append(zo_eval_acc_0)
    sgd_loss_history.append(sgd_eval_loss_0)
    zo_loss_history.append(zo_eval_loss_0)
    eval_iterations_log.append(0)

    # Create a shared data iterator so both trainers see the same batches
    # This ensures a fair comparison between SGD and ZO
    # CRITICAL: Each trainer's train_one_step() calls next(self.data_iterator) independently,
    # so we need to manually get batches and train both models with the same batch
    shared_data_iterator = iter(train_loader)
    
    def get_shared_batch():
        """Get next batch from shared iterator, resetting if exhausted."""
        nonlocal shared_data_iterator
        try:
            return next(shared_data_iterator)
        except StopIteration:
            shared_data_iterator = iter(train_loader)
            return next(shared_data_iterator)
    
    # Train both models with the SAME batches
    for iteration in tqdm(range(num_iterations), desc="Training"):
        # Get the same batch for both trainers
        batch_inputs, labels = get_shared_batch()
        
        # Train SGD model with this batch
        sgd_model.train()
        if device != torch.device("cpu") or torch_dtype != torch.float32:
            batch_inputs_sgd = batch_inputs.to(device, torch_dtype)
            if isinstance(labels, torch.Tensor):
                labels_sgd = labels.to(device)
            else:
                labels_sgd = labels
        else:
            batch_inputs_sgd = batch_inputs
            labels_sgd = labels
        
        sgd_trainer.optimizer.zero_grad()
        pred_sgd = model_inference_fn(sgd_model, batch_inputs_sgd)
        loss_sgd = metrics.test_loss(pred_sgd, labels_sgd)
        loss_sgd.backward()
        sgd_trainer.optimizer.step()
        
        # Train ZO model with the SAME batch
        zo_model.eval()  # ZO uses eval mode
        if device != torch.device("cpu") or torch_dtype != torch.float32:
            batch_inputs_zo = batch_inputs.to(device, torch_dtype)
            if isinstance(labels, torch.Tensor):
                labels_zo = labels.to(device)
            else:
                labels_zo = labels
        else:
            batch_inputs_zo = batch_inputs
            labels_zo = labels
        
        seed = random.randint(0, 1000000)
        
        def loss_fn(model):
            return metrics.test_loss(model_inference_fn(model, batch_inputs_zo), labels_zo)
        
        with torch.no_grad():
            zo_optimizer.update_model_given_seed(iteration=iteration, seed=seed, loss_fn=loss_fn)
        
        # Extract logits and probabilities at each step
        sgd_step_logits, sgd_step_probs, zo_step_logits, zo_step_probs = extract_logits()
        for i in range(num_test_samples):
            sgd_logits_history[i].append(sgd_step_logits[i].clone())
            sgd_probs_history[i].append(sgd_step_probs[i].clone())
            zo_logits_history[i].append(zo_step_logits[i].clone())
            zo_probs_history[i].append(zo_step_probs[i].clone())
        
        # Evaluate on test set
        if eval_iterations > 0 and (iteration + 1) % eval_iterations == 0:
            print(f"\nEvaluation at iteration {iteration + 1}:")
            sgd_eval_loss, sgd_eval_acc = sgd_trainer.eval_model(test_loader, False)
            zo_eval_loss, zo_eval_acc = zo_trainer.eval_model(test_loader, False)
            print(f"  SGD - Eval Loss: {sgd_eval_loss:.4f}, Eval Acc: {sgd_eval_acc:.4f}")
            print(f"  ZO  - Eval Loss: {zo_eval_loss:.4f}, Eval Acc: {zo_eval_acc:.4f}")
            sgd_acc_history.append(sgd_eval_acc)
            zo_acc_history.append(zo_eval_acc)
            sgd_loss_history.append(sgd_eval_loss)
            zo_loss_history.append(zo_eval_loss)
            eval_iterations_log.append(iteration + 1)
    
    # Extract final logits and probabilities (after training)
    sgd_final_logits, sgd_final_probs, zo_final_logits, zo_final_probs = extract_logits()
    
    # Print summary of logits changes
    print("Logits Summary (for output tokens)")
    
    for i in range(num_test_samples):
        print(f"\nTest Sample {i + 1}:")
        print("-" * 80)
        
        # Print raw sentence if available
        if i < len(test_raw_sentences) and test_raw_sentences[i] is not None:
            raw_sample = test_raw_sentences[i]
            # Extract raw sentence text based on common fields
            if "sentence" in raw_sample:
                print(f"Raw sentence: {raw_sample['sentence']}")
            elif "text" in raw_sample:
                print(f"Raw text: {raw_sample['text']}")
            elif "question1" in raw_sample and "question2" in raw_sample:
                print(f"Raw Q1: {raw_sample['question1']}")
                print(f"Raw Q2: {raw_sample['question2']}")
            elif "passage" in raw_sample and "question" in raw_sample:
                print(f"Raw passage: {raw_sample['passage']}")
                print(f"Raw question: {raw_sample['question']}")
            elif "premise" in raw_sample and "hypothesis" in raw_sample:
                print(f"Raw premise: {raw_sample['premise']}")
                print(f"Raw hypothesis: {raw_sample['hypothesis']}")
            else:
                print(f"Raw sample: {raw_sample}")
            if "label" in raw_sample:
                print(f"Label: {raw_sample['label']}")
        
        # Print encoded/verbalized text if available
        if i < len(test_encoded_texts) and test_encoded_texts[i] is not None:
            print(f"Encoded text: {test_encoded_texts[i]}")
        
        # Initial and final logits
        sgd_initial = sgd_initial_logits[i]
        zo_initial = zo_initial_logits[i]
        sgd_final = sgd_final_logits[i]
        zo_final = zo_final_logits[i]
        
        # Changes
        sgd_change = sgd_final - sgd_initial
        zo_change = zo_final - zo_initial
        
        print(f"SGD - Initial logits: {sgd_initial.tolist()}")
        print(f"SGD - Final logits:   {sgd_final.tolist()}")
        print(f"SGD - Change:          {sgd_change.tolist()}")
        print()
        print(f"ZO  - Initial logits: {zo_initial.tolist()}")
        print(f"ZO  - Final logits:   {zo_final.tolist()}")
        print(f"ZO  - Change:          {zo_change.tolist()}")
    
    return {
        "sgd_initial_logits": sgd_initial_logits,
        "sgd_final_logits": sgd_final_logits,
        "sgd_initial_probs": sgd_initial_probs,
        "sgd_final_probs": sgd_final_probs,
        "zo_initial_logits": zo_initial_logits,
        "zo_final_logits": zo_final_logits,
        "zo_initial_probs": zo_initial_probs,
        "zo_final_probs": zo_final_probs,
        "sgd_logits_history": sgd_logits_history,
        "sgd_probs_history": sgd_probs_history,
        "zo_logits_history": zo_logits_history,
        "zo_probs_history": zo_probs_history,
        "test_samples": test_samples,
        "test_raw_sentences": test_raw_sentences,
        "test_encoded_texts": test_encoded_texts,
        "verbalizer_id_list": verbalizer_id_list,
        "sgd_acc_history": sgd_acc_history,
        "zo_acc_history": zo_acc_history,
        "sgd_loss_history": sgd_loss_history,
        "zo_loss_history": zo_loss_history,
        "eval_iterations_log": eval_iterations_log,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare SGD vs ZO optimization")
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
        help="Number of test samples to track",
    )
    
    args = parser.parse_args()
    
    results = run_experiment(
        config_path=args.config_path,
        num_test_samples=args.num_test_samples,
    )
