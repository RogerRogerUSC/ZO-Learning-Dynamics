from __future__ import annotations
import random
from typing import Any, Callable, Iterable, Iterator, TypeAlias
import torch

from zo_llm.zo_optim import ZOOptimizer
from util.metrics import Metric

CriterionType: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class LLM_trainer:
    def __init__(self, device, torch_dtype, dataloader) -> None:
        self.device = device
        self.torch_dtype = torch_dtype
        self.dataloader = dataloader
        self.data_iterator = self._get_train_batch_iterator()
        
        self.model = None
        self.model_inference: Callable[[torch.nn.Module, Any], torch.Tensor] | None = None
        self.criterion: CriterionType | None = None
        self.accuracy_func = None

    def set_model_and_criterion(
        self,
        model: torch.nn.Module,
        model_inference: Callable[[torch.nn.Module, Any], torch.Tensor],
        criterion: CriterionType,
        accuracy_func,
        zo_optimizer: ZOOptimizer,
    ) -> None:
        self.model = model
        self.model_inference = model_inference
        self.criterion = criterion
        self.accuracy_func = accuracy_func
        self.zo_optimizer = zo_optimizer

    def set_lr(self, lr: float) -> None:
        self.zo_optimizer.lr = lr

    def train_one_step(self, iteration: int) -> tuple[float, float]:
        seed = random.randint(0, 1000000)
        train_loss = Metric("Train loss")
        train_acc = Metric("Train acc")

        with torch.no_grad():
            batch_inputs, labels = next(self.data_iterator)
            if (
                self.device != torch.device("cpu")
                or self.torch_dtype != torch.float32
            ):
                batch_inputs = batch_inputs.to(self.device, self.torch_dtype)
                if isinstance(labels, torch.Tensor):  # In generation mode, labels are not tensor.
                    labels = labels.to(self.device)

            def loss_fn(model):
                return self.criterion(self.model_inference(model, batch_inputs), labels)

            self.zo_optimizer.update_model_given_seed(seed=seed, loss_fn=loss_fn)

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
                    or self.torch_dtype != torch.float32
                ):
                    batch_inputs = batch_inputs.to(self.device, self.torch_dtype)
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
