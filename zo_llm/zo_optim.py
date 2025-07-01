from typing import Callable

import torch
import torch.nn as nn

from zo_llm.perturb import PerturbBase, GaussianPerturb


def forward_zo_estimate(
    model, loss_fn, num_pert: int, perturbor: PerturbBase
) -> list[torch.Tensor]:
    grad_scalars: list[torch.Tensor] = []
    loss_base = loss_fn(model)
    for i in range(num_pert):
        perturbor.perturb(model.parameters(), i, perturbor.mu)
        pert_plus_loss = loss_fn(model)
        perturbor.perturb(model.parameters(), i, -1 * perturbor.mu)

        grad_scalars.append((pert_plus_loss - loss_base) / perturbor.mu)
    return grad_scalars


def central_zo_estimate(
    model, loss_fn, num_pert: int, perturbor: PerturbBase
) -> list[torch.Tensor]:
    grad_scalars: list[torch.Tensor] = []
    for i in range(num_pert):
        perturbor.perturb(model.parameters(), i, perturbor.mu)
        pert_plus_loss = loss_fn(model)
        perturbor.perturb(model.parameters(), i, -2 * perturbor.mu)
        pert_minus_loss = loss_fn(model)
        perturbor.perturb(model.parameters(), i, perturbor.mu)

        grad_scalars.append((pert_plus_loss - pert_minus_loss) / perturbor.mu)
    return grad_scalars


def zo_sgd_step(
    model, lr: float, perturbor: PerturbBase, grad_scalars: list[torch.Tensor]
) -> list[torch.Tensor]:
    num_pert = len(grad_scalars)
    for i, grad_scalar in enumerate(grad_scalars):
        perturbor.perturb(model.parameters(), i, -1 * lr * grad_scalar / num_pert)
    return model


class ZOOptimizer:
    def __init__(self, model, lr, device, num_pert, estimate_func, perturbor):
        self.model = model
        self.device = device
        self.num_pert = num_pert
        self.estimate_func = estimate_func
        self.perturbor = perturbor
        self.lr = lr

    @classmethod
    def from_config(cls, config, model):
        # TODO make criterion and model_inference_fn into proper position.
        return ZOOptimizer(
            model=model,
            num_pert=config.num_pert,
            device=config.device,
            lr=config.lr,
            # TODO: make this selection according to config
            estimate_func=forward_zo_estimate,
            perturbor=GaussianPerturb(config.device),
        )

    def update_model_given_seed(self, seed: int, loss_fn: Callable[[nn.Module], torch.tensor]):
        # Loss_fn is a function takes model as input and return loss value.
        # Consider a better approach?
        self.perturbor.set_seed(seed)
        grad_scalars = self.estimate_func(self.model, loss_fn, self.num_pert, self.perturbor)
        return zo_sgd_step(
            self.model, lr=self.lr, perturbor=self.perturbor, grad_scalars=grad_scalars
        )
