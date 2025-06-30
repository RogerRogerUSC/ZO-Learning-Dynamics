from __future__ import annotations
import random
from typing import Any, Callable, Iterable, Iterator
import torch
from grad_estimators.abstract_grad_estimator import AbstractGradientEstimator
from grad_estimators.random_grad_estimator import RandomGradientEstimator
from util.typing import CriterionType
from util.metrics import Metric


class LLM_trainer:
    def __init__(self, device, dataloader) -> None:
        self.device = device
        self.model = None
        self.model_inference: Callable[[torch.nn.Module, Any], torch.Tensor] | None = None
        self.criterion: CriterionType | None = None
        self.accuracy_func = None
        self.optim: torch.optim.Optimizer | None = None
        self.grad_estimator: AbstractGradientEstimator | None = None
        self.dataloader = dataloader
        self.data_iterator = self._get_train_batch_iterator()

    def set_model_and_criterion(
        self,
        model: torch.nn.Module,
        model_inference: Callable[[torch.nn.Module, Any], torch.Tensor],
        criterion: CriterionType,
        accuracy_func,
        optimizer: torch.optim.Optimizer,
        grad_estimator: AbstractGradientEstimator,
    ) -> None:
        self.model = model
        self.model_inference = model_inference
        self.criterion = criterion
        self.accuracy_func = accuracy_func
        self.optim = optimizer
        self.grad_estimator = grad_estimator

    def set_lr(self, lr: float) -> None:
        if self.model and self.optim:
            for p in self.optim.param_groups:
                p["lr"] = lr

    def _loss_fn(self, batch_inputs, batch_labels):
        return self.criterion(self.model_inference(self.model, batch_inputs), batch_labels)
    
    def train_one_step(self, iteration: int) -> tuple[float, float]:
        seed = random.randint(0, 1000000)
        train_loss = Metric("Train loss")
        train_acc = Metric("Train acc")

        with torch.no_grad():
            self.optim.zero_grad()
            batch_inputs, labels = next(self.data_iterator)
            if (
                self.device != torch.device("cpu")
                or self.grad_estimator.torch_dtype != torch.float32
            ):
                batch_inputs = batch_inputs.to(self.device, self.grad_estimator.torch_dtype)
                if isinstance(labels, torch.Tensor):  # In generation mode, labels are not tensor.
                    labels = labels.to(self.device)
            grad_scalars: torch.Tensor
            if self.grad_estimator.sgd_only_no_optim and isinstance(
                self.grad_estimator, RandomGradientEstimator
            ):
                grad_scalars = self.grad_estimator._zo_grad_estimate_paramwise(
                    batch_inputs, labels, self._loss_fn, seed
                )
                self.grad_estimator.update_model_given_seed_and_grad(
                    self.optim, [seed], [grad_scalars]
                )
            else:
                # generate grads and update model's gradient
                # The length of grad_scalars is number of perturbations
                grad_scalars = self.grad_estimator.compute_grad(
                    batch_inputs, labels, self._loss_fn, seed
                )
                self.optim.step()

            pred = self.model_inference(self.model, batch_inputs)
            train_loss.update(self.criterion(pred, labels))
            train_acc.update(self.accuracy_func(pred, labels))

        return train_loss.avg, train_acc.avg

    def eval_model(self, test_loader: Iterable[Any]) -> tuple[float, float]:
        self.model.eval()
        eval_loss = Metric("Eval loss")
        eval_acc = Metric("Eval acc")
        with torch.no_grad():
            for _, (batch_inputs, batch_labels) in enumerate(test_loader):
                if (
                    self.device != torch.device("cpu")
                    or self.gradient_estimator.torch_dtype != torch.float32
                ):
                    batch_inputs = batch_inputs.to(self.device, self.grad_estimator.torch_dtype)
                    # In generation mode, labels are not tensor.
                    if isinstance(batch_labels, torch.Tensor):
                        batch_labels = batch_labels.to(self.device)
                pred = self.model_inference(self.model, batch_inputs)
                eval_loss.update(self.criterion(pred, batch_labels))
                eval_acc.update(self.accuracy_func(pred, batch_labels))
        print(
            # f"\nEvaluation(Iteration {self.seed_grad_records.current_iteration}): ",
            f"Eval Loss:{eval_loss.avg:.4f}, Eval Acc:{eval_acc.avg * 100:.2f}%",
        )
        return eval_loss.avg, eval_acc.avg

    def _get_train_batch_iterator(self) -> Iterator:
        # NOTE: used only in init, will generate an infinite iterator from dataloader
        while True:
            for v in self.dataloader:
                yield v
