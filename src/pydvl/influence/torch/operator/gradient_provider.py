from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

import torch
from torch.func import functional_call

from ...types import PerSampleGradientProvider
from ..functional import (
    create_matrix_jacobian_product_function,
    create_per_sample_gradient_function,
    create_per_sample_mixed_derivative_function,
)
from ..util import (
    BlockMode,
    LossType,
    ModelParameterDictBuilder,
    TorchBatch,
    flatten_dimensions,
)


class TorchPerSampleGradientProvider(
    PerSampleGradientProvider[TorchBatch, torch.Tensor], ABC
):
    r"""
    Abstract base class for calculating per-sample gradients of a function defined by
    a [torch.nn.Module][torch.nn.Module] and a loss function.

    This class must be subclassed with implementations for its abstract methods tailored
    to specific gradient computation needs, e.g. using [torch.autograd][torch.autograd]
    or stochastic finite differences.

    Consider a function

    $$ \ell: \mathbb{R}^{d_1} \times \mathbb{R}^{d_2} \times \mathbb{R}^{n} \times
        \mathbb{R}^{n}, \quad \ell(\omega_1, \omega_2, x, y) =
        \operatorname{loss}(f(\omega_1, \omega_2; x), y) $$

    e.g. a two layer neural network $f$ with a loss function, then this object should
    compute the expressions:

    $$ \nabla_{\omega_{i}}\ell(\omega_1, \omega_2, x, y),
    \nabla_{\omega_{i}}\nabla_{x}\ell(\omega_1, \omega_2, x, y),
    \nabla_{\omega}\ell(\omega_1, \omega_2, x, y) \cdot v$$

    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: LossType,
        restrict_to: Optional[Dict[str, torch.nn.Parameter]],
    ):
        self.loss = loss
        self.model = model

        if restrict_to is None:
            restrict_to = ModelParameterDictBuilder(model).build(BlockMode.FULL)

        self.params_to_restrict_to = restrict_to

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        self.params_to_restrict_to = {
            k: p.detach()
            for k, p in self.model.named_parameters()
            if k in self.params_to_restrict_to
        }
        return self

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @abstractmethod
    def _per_sample_gradient_dict(self, batch: TorchBatch) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def _per_sample_mixed_gradient_dict(
        self, batch: TorchBatch
    ) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def _matrix_jacobian_product(
        self,
        batch: TorchBatch,
        g: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def _detach_dict(tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: g.detach() if g.requires_grad else g for k, g in tensor_dict.items()}

    def per_sample_gradient_dict(self, batch: TorchBatch) -> Dict[str, torch.Tensor]:
        r"""
        Computes and returns a dictionary mapping gradient names to their respective
        per-sample gradients. Given the example in the class docstring, this means

        $$ \text{result}[\omega_i] = \nabla_{\omega_{i}}\ell(\omega_1, \omega_2,
            \text{batch.x}, \text{batch.y}), $$

        where the first dimension of the resulting tensors is always considered to be
        the batch dimension, so the shape of the resulting tensors are $(N, d_i)$,
        where $N$ is the number of samples in the batch.

        Args:
            batch: The batch of data for which to compute gradients.

        Returns:
            A dictionary where keys are gradient identifiers and values are the
                gradients computed per sample.
        """
        gradient_dict = self._per_sample_gradient_dict(batch.to(self.device))
        return self._detach_dict(gradient_dict)

    def per_sample_mixed_gradient_dict(
        self, batch: TorchBatch
    ) -> Dict[str, torch.Tensor]:
        r"""
        Computes and returns a dictionary mapping gradient names to their respective
        per-sample mixed gradients. In this context, mixed gradients refer to computing
        gradients with respect to the instance definition in addition to
        compute derivatives with respect to the input batch.
        Given the example in the class docstring, this means

        $$ \text{result}[\omega_i] = \nabla_{\omega_{i}}\nabla_{x}\ell(\omega_1,
            \omega_2, \text{batch.x}, \text{batch.y}), $$

        where the first dimension of the resulting tensors is always considered to be
        the batch dimension and the last to be the non-batch input related derivatives.
        So the shape of the resulting tensors are $(N, n, d_i)$,
        where $N$ is the number of samples in the batch.

        Args:
            batch: The batch of data for which to compute mixed gradients.

        Returns:
            A dictionary where keys are gradient identifiers and values are the
                mixed gradients computed per sample.
        """
        gradient_dict = self._per_sample_mixed_gradient_dict(batch.to(self.device))
        return self._detach_dict(gradient_dict)

    def matrix_jacobian_product(
        self,
        batch: TorchBatch,
        g: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Computes the matrix-Jacobian product for the provided batch and input tensor.
        Given the example in the class docstring, this means

        $$ (\nabla_{\omega_{1}}\ell(\omega_1, \omega_2,
            \text{batch.x}, \text{batch.y}),
            \nabla_{\omega_{2}}\ell(\omega_1, \omega_2,
            \text{batch.x}, \text{batch.y})) \cdot g^T$$

        where g must be a tensor of shape $(K, d_1+d_2)$, so the resulting tensor
        is of shape $(N, K)$.

        Args:
            batch: The batch of data for which to compute the Jacobian.
            g: The tensor to be used in the matrix-Jacobian product
                calculation.

        Returns:
            The resulting tensor from the matrix-Jacobian product computation.
        """
        result = self._matrix_jacobian_product(batch.to(self.device), g.to(self.device))
        if result.requires_grad:
            result = result.detach()
        return result

    def per_sample_flat_gradient(self, batch: TorchBatch) -> torch.Tensor:
        return flatten_dimensions(
            self.per_sample_gradient_dict(batch).values(), shape=(batch.x.shape[0], -1)
        )

    def per_sample_flat_mixed_gradient(self, batch: TorchBatch) -> torch.Tensor:
        shape = (*batch.x.shape, -1)
        return flatten_dimensions(
            self.per_sample_mixed_gradient_dict(batch).values(), shape=shape
        )


class TorchPerSampleAutoGrad(TorchPerSampleGradientProvider):
    r"""
    Compute per-sample gradients of a function defined by
    a [torch.nn.Module][torch.nn.Module] and a loss function using
    [torch.func][torch.func].

    Consider a function

    $$ \ell: \mathbb{R}^{d_1} \times \mathbb{R}^{d_2} \times \mathbb{R}^{n} \times
        \mathbb{R}^{n}, \quad \ell(\omega_1, \omega_2, x, y) =
        \operatorname{loss}(f(\omega_1, \omega_2; x), y) $$

    e.g. a two layer neural network $f$ with a loss function, then this object should
    compute the expressions:

    $$ \nabla_{\omega_{i}}\ell(\omega_1, \omega_2, x, y),
    \nabla_{\omega_{i}}\nabla_{x}\ell(\omega_1, \omega_2, x, y),
    \nabla_{\omega}\ell(\omega_1, \omega_2, x, y) \cdot v$$

    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: LossType,
        restrict_to: Optional[Dict[str, torch.nn.Parameter]] = None,
    ):
        super().__init__(model, loss, restrict_to)
        self._per_sample_gradient_function = create_per_sample_gradient_function(
            model, loss
        )
        self._per_sample_mixed_gradient_func = (
            create_per_sample_mixed_derivative_function(model, loss)
        )

    def _compute_loss(
        self, params: Dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        outputs = functional_call(self.model, params, (x.unsqueeze(0).to(self.device),))
        return self.loss(outputs, y.unsqueeze(0))

    def _per_sample_gradient_dict(self, batch: TorchBatch) -> Dict[str, torch.Tensor]:
        return self._per_sample_gradient_function(
            self.params_to_restrict_to, batch.x, batch.y
        )

    def _per_sample_mixed_gradient_dict(
        self, batch: TorchBatch
    ) -> Dict[str, torch.Tensor]:
        return self._per_sample_mixed_gradient_func(
            self.params_to_restrict_to, batch.x, batch.y
        )

    def _matrix_jacobian_product(
        self,
        batch: TorchBatch,
        g: torch.Tensor,
    ) -> torch.Tensor:
        matrix_jacobian_product_func = create_matrix_jacobian_product_function(
            self.model, self.loss, g
        )
        return matrix_jacobian_product_func(
            self.params_to_restrict_to, batch.x, batch.y
        )


GradientProviderFactoryType = Callable[
    [torch.nn.Module, LossType, Optional[Dict[str, torch.nn.Parameter]]],
    TorchPerSampleGradientProvider,
]
