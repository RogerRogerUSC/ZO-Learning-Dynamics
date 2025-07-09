import torch


class PerturbBase:
    def get_seed(self):
        if self._seed is None:
            raise ValueError("Forget to set a seed?")
        return self._seed

    def set_seed(self, seed: int):
        self._seed = seed

    def pre_hook(self, iteration: int, params: list[torch.Tensor]):
        pass

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
    def __init__(self, device, mu=1e-3):
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


class BernoulliPerturb(PerturbBase):
    def __init__(self, device, mu=1e-3):
        self._mu = mu
        self._device = device

    def perturb(self, params, index: int, alpha: float) -> None:
        seed = self.get_seed()
        rng = self.get_rng(seed, index)
        for param in params:
            perturb = (
                torch.randint(
                    0, 2, param.shape, device=self._device, dtype=param.dtype, generator=rng
                )
                * 2
                - 1
            )
            param.add_(perturb, alpha=alpha)


class UnifromPerturb(PerturbBase):
    def __init__(self, device, mu=1e-3):
        self._mu = mu
        self._device = device

    def perturb(self, params, index: int, alpha: float) -> None:
        seed = self.get_seed()
        rng = self.get_rng(seed, index)
        for param in params:
            perturb = (
                torch.rand(*param.shape, device=self._device, dtype=param.dtype, generator=rng) * 2
            ) - 1
            param.add_(perturb, alpha=alpha)


class RandomizedGaussianPerturb(PerturbBase):
    def __init__(self, device, mu=1e-3, percent=0.25):
        self._mu = mu
        self._device = device
        self._percent = percent
        self._mask = None
        self.update_mark_every_iterations = 50

    def get_mask(self):
        if self._mask is None:
            raise ValueError()
        return self._mask

    def set_mask(self, params):
        self._mask = []
        seed = self.get_seed()
        rng = self.get_rng(seed, -1)
        for param in params:
            self._mask.append(
                torch.rand(*param.shape, device=self.device, generator=rng) < self._percent
            )

    def pre_hook(self, iteration: int, params: list[torch.Tensor]):
        if iteration % self.update_mark_every_iterations == 0:
            self.set_mask(params)

    def perturb(self, params, index: int, alpha: float) -> None:
        seed = self.get_seed()
        rng = self.get_rng(seed, index)
        for param, mask in zip(params, self.get_mask()):
            perturb = (
                torch.randn(*param.shape, device=self.device, dtype=param.dtype, generator=rng)
                * mask
            )
            param.add_(perturb, alpha=alpha)
