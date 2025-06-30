import torch
from zo_llm.perturb import PerturbBase, GaussianPerturb


def forward_zo_estimate(
    model, loss_fn, num_perb: int, perturbor: PerturbBase
) -> list[torch.Tensor]:
    grad_scalars: list[torch.Tensor] = []
    loss_base = loss_fn(model)
    for i in num_perb:
        perturbor.perturb(model.parameters, i, perturbor.mu)
        pert_plus_loss = loss_fn(model)
        perturbor.perturb(model.parameters, i, -1 * perturbor.mu)

        grad_scalars.append((pert_plus_loss - loss_base) / perturbor.mu)
    return grad_scalars


def central_zo_estimate(
    model, loss_fn, num_perb: int, perturbor: PerturbBase
) -> list[torch.Tensor]:
    grad_scalars: list[torch.Tensor] = []
    for i in num_perb:
        perturbor.perturb(model.parameters, i, perturbor.mu)
        pert_plus_loss = loss_fn(model)
        perturbor.perturb(model.parameters, i, -2 * perturbor.mu)
        pert_minus_loss = loss_fn(model)
        perturbor.perturb(model.parameters, i, perturbor.mu)

        grad_scalars.append((pert_plus_loss - pert_minus_loss) / perturbor.mu)
    return grad_scalars


def zo_sgd_step(
    model, lr: float, perturbor: PerturbBase, grad_scalars: list[torch.Tensor]
) -> list[torch.Tensor]:
    num_perb = len(grad_scalars)
    for i, grad_scalar in enumerate(grad_scalars):
        perturbor.perturb(model.parameters, i, -1 * lr * grad_scalar / num_perb)
    return model


class ZOOptimizer:
    def __init__(self, device, num_perturb, step_func, estimate_func, perturbor):
        self.device = device
        self.num_perturb = num_perturb
        self.step_func = step_func
        self.estimate_func = estimate_func
        self.perturbor = perturbor

    @classmethod
    def from_config(cls, config):
        # todo
        return ZOOptimizer(
            num_perturb=config.num_perturb,
            device=config.device,
            step_func=zo_sgd_step,
            estimate_func=forward_zo_estimate,
            perturbor=GaussianPerturb(config.device),
        )

    def update_model_given_seed(self, lr, seed: int, model, batch_inputs, labels):
        # TODO construct loss function
        loss_fn = (batch_inputs, labels)
        self.perturbor.set_seed(seed)
        grad_scalars = self.estimate_func(model, loss_fn, self.num_perb, self.perturbor)
        return self.step_func(model, lr=lr, perturbor=self.perturbor, grad_scalars=grad_scalars)
