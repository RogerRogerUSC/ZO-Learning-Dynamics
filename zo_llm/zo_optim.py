import torch


class PerturbBase:
    @property
    def mu(self) -> float:
        return self._mu

    def perturb(self, param, index, alpha) -> None:
        raise NotImplementedError


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
