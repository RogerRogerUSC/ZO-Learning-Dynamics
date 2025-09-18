from typing import Callable

import torch
import torch.nn as nn

from zo_llm.perturb import (
    BernoulliPerturb,
    GaussianPerturb,
    PerturbBase,
    RandomizedGaussianPerturb,
    UnifromPerturb,
)


def forward_zo_estimate(
    model, loss_fn, num_pert: int, perturbator: PerturbBase
) -> list[torch.Tensor]:
    grad_scalars: list[torch.Tensor] = []
    loss_base = loss_fn(model)
    for i in range(num_pert):
        perturbator.perturb(model.parameters(), i, perturbator.mu)
        pert_plus_loss = loss_fn(model)
        perturbator.perturb(model.parameters(), i, -1 * perturbator.mu)

        grad_scalars.append((pert_plus_loss - loss_base) / perturbator.mu)
    return grad_scalars


def central_zo_estimate(
    model, loss_fn, num_pert: int, perturbator: PerturbBase
) -> list[torch.Tensor]:
    grad_scalars: list[torch.Tensor] = []
    for i in range(num_pert):
        perturbator.perturb(model.parameters(), i, perturbator.mu)
        pert_plus_loss = loss_fn(model)
        perturbator.perturb(model.parameters(), i, -2 * perturbator.mu)
        pert_minus_loss = loss_fn(model)
        perturbator.perturb(model.parameters(), i, perturbator.mu)

        grad_scalars.append((pert_plus_loss - pert_minus_loss) / perturbator.mu)
    return grad_scalars


def zo_sgd_step(
    model, lr: float, perturbator: PerturbBase, grad_scalars: list[torch.Tensor]
) -> list[torch.Tensor]:
    num_pert = len(grad_scalars)
    for i, grad_scalar in enumerate(grad_scalars):
        perturbator.perturb(model.parameters(), i, -1 * lr * grad_scalar / num_pert)
    return model


class ZOOptimizer:
    def __init__(self, model, lr, device, num_pert, estimate_func, perturbator):
        self.model = model
        self.device = device
        self.num_pert = num_pert
        self.estimate_func = estimate_func
        self.perturbator = perturbator
        self.lr = lr

    @classmethod
    def from_config(cls, config, model):
        # TODO make criterion and model_inference_fn into proper position.
        if config.pert_distribution == "gaussian":
            perturbator = GaussianPerturb(config.device, config.mu)
        elif config.pert_distribution == "bernoulli":
            perturbator = BernoulliPerturb(config.device, config.mu)
        elif config.pert_distribution == "uniform":
            perturbator = UnifromPerturb(config.device, config.mu)
        elif config.pert_distribution == "random_gaussian":
            perturbator = RandomizedGaussianPerturb(config.device, config.mu)
        else:
            raise Exception(config.pert_distribution + " is not supported!")

        return ZOOptimizer(
            model=model,
            num_pert=config.num_pert,
            device=config.device,
            lr=config.lr,
            estimate_func=forward_zo_estimate,
            perturbator=perturbator,
        )

    def update_model_given_seed(
        self, iteration: int, seed: int, loss_fn: Callable[[nn.Module], torch.tensor]
    ):
        # Loss_fn is a function takes model as input and return loss value.
        # Consider a better approach?
        self.perturbator.set_seed(seed)
        self.perturbator.pre_hook(iteration, self.model.parameters())

        grad_scalars = self.estimate_func(self.model, loss_fn, self.num_pert, self.perturbator)
        return zo_sgd_step(
            self.model, lr=self.lr, perturbator=self.perturbator, grad_scalars=grad_scalars
        )
