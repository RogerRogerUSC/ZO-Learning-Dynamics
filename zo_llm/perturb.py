import torch


class PerturbBase:
    def get_seed(self):
        if self._seed is None:
            raise ValueError("Forget to set a seed?")
        return self._seed

    def set_seed(self, seed: int):
        self._seed = seed

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    def get_rng(self, seed: int, perturb_index: int) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(
            seed * (perturb_index + 17) + perturb_index
        )

    def perturb(self, params, index: int, alpha: float) -> None:
        raise NotImplementedError


class GaussianPerturb(PerturbBase):
    def __init__(self, device, mu=1e-4):
        self._mu = mu
        self._device = device

    def perturb(self, params, index: int, alpha: float) -> None:
        seed = self.get_seed()
        rng = self.get_rng(seed, index)
        for param in params:
            perturb = torch.randn(
                *param.shape, device=self.device, dtype=param.dtype, generator=rng
            )
            param.add_(perturb, alpha=alpha)
