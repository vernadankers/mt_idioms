import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch import nn

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def hard_sigmoid(x):
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))


class _L0Norm(nn.Module):

    def __init__(self, origin, loc_mean=0, loc_sdev=0.01, beta=2 / 3, gamma=-0.1,
                 zeta=1.1, fix_temp=True):
        """
        Base class of layers using L0 Norm
        :param origin: original layer such as nn.Linear(..), nn.Conv2d(..)
        :param loc_mean: mean of the normal distribution which generates initial location parameters
        :param loc_sdev: standard deviation of the normal distribution which generates initial location parameters
        :param beta: initial temperature parameter
        :param gamma: lower bound of "stretched" s
        :param zeta: upper bound of "stretched" s
        :param fix_temp: True if temperature is fixed
        """
        super(_L0Norm, self).__init__()
        self._origin = origin
        self._size = self._origin.weight.size()
        self.loc = nn.Parameter(torch.zeros(self._size).normal_(loc_mean, loc_sdev))
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        self.register_buffer("uniform", torch.zeros(self._size))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)

    def l0_norm(self):
        penalty = torch.sigmoid(self.loc - self.temp * self.gamma_zeta_ratio).sum()
        return penalty

    def _get_mask(self):
        if self.training:
            self.uniform.uniform_()
            u = Variable(self.uniform)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
            # penalty = torch.sigmoid(self.loc - self.temp * self.gamma_zeta_ratio).sum()
        else:
            s = torch.sigmoid(self.loc) * (self.zeta - self.gamma) + self.gamma
            # penalty = 0
        return hard_sigmoid(s)# , penalty


class L0Linear(_L0Norm):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(L0Linear, self).__init__(nn.Linear(in_features, out_features, bias=bias), **kwargs)

    def forward(self, input):
        mask = self._get_mask()
        return F.linear(input, self._origin.weight * mask, self._origin.bias)


class HardConcrete(nn.Module):
    """A HarcConcrete module.
    Use this module to create a mask of size N, which you can
    then use to perform L0 regularization. Note that in general,
    we also provide utilities which introduce HardConrete modules
    in the desired places in your model. See ``utils`` for details.
    To obtain a mask, simply run a forward pass through the module
    with no input data. The mask is sampled in training mode, and
    fixed during evaluation mode:
    >>> module = HardConcrete(n_in=100)
    >>> mask = module()
    >>> norm = module.l0_norm()
    """

    def __init__(self,
                 n_in: int,
                 init_mean: float = 0.5,
                 init_std: float = 0.01,
                 temperature: float = 1.0,
                 stretch: float = 0.1,
                 eps: float = 1e-6) -> None:
        """Initialize the HardConcrete module.
        Parameters
        ----------
        n_in : int
            The number of hard concrete variables in this mask.
        init_mean : float, optional
            Initialization value for hard concrete parameter,
            by default 0.5.,
        init_std: float, optional
            Used to initialize the hard concrete parameters,
            by default 0.01.
        temperature : float, optional
            Temperature used to control the sharpness of the
            distribution, by default 1.0
        stretch : float, optional
            Stretch the sampled value from [0, 1] to the interval
            [-stretch, 1 + stretch], by default 0.1.
        """
        super().__init__()

        self.n_in = n_in
        self.limit_l = -stretch
        self.limit_r = 1.0 + stretch
        self.log_alpha = nn.Parameter(torch.zeros(n_in))  # type: ignore
        self.beta = temperature
        self.init_mean = init_mean
        self.init_std = init_std
        self.bias = -self.beta * math.log(-self.limit_l / self.limit_r)

        self.eps = eps
        self.compiled_mask = None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of this module."""
        self.compiled_mask = None
        mean = math.log(1 - self.init_mean) - math.log(self.init_mean)
        self.log_alpha.data.normal_(mean, self.init_std)

    def l0_norm(self) -> torch.Tensor:
        """Compute the expected L0 norm of this mask.
        Returns
        -------
        torch.Tensor
            The expected L0 norm.
        """
        return (self.log_alpha + self.bias).sigmoid().sum()

    def forward(self) -> torch.Tensor:  # type: ignore
        """Sample a harconcrete mask.
        Returns
        -------
        torch.Tensor
            The sampled binary mask
        """
        if self.training:
            # Reset the compiled mask
            self.compiled_mask = None
            # Sample mask dynamically
            u = self.log_alpha.new(self.n_in).uniform_(self.eps, 1 - self.eps)  # type: ignore
            s = torch.sigmoid((torch.log(u / (1 - u)) + self.log_alpha) / self.beta)
            s = s * (self.limit_r - self.limit_l) + self.limit_l
            mask = s.clamp(min=0., max=1.)

        else:
            # Compile new mask if not cached
            if self.compiled_mask is None:
                # Get expected sparsity
                expected_num_zeros = self.n_in - self.l0_norm().item()
                num_zeros = round(expected_num_zeros)
                # Approximate expected value of each mask variable z;
                # We use an empirically validated magic number 0.8
                soft_mask = torch.sigmoid(self.log_alpha / self.beta * 0.8)
                # Prune small values to set to 0
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
                self.compiled_mask = soft_mask
            mask = self.compiled_mask

        return mask

    def extre_repr(self) -> str:
        return str(self.n_in)

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.extre_repr())


class HardConcreteLinear(nn.Module):
    """The hard concrete equivalent of ``nn.Linear``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_mean: float = 0.5,
        init_std: float = 0.01,
    ) -> None:
        """Initialize a HardConcreteLinear module.
        Parameters
        ----------
        in_features : int
            The number of input features
        out_features : int
            The number of output features
        bias : bool, optional
            Whether to add a bias term, by default True
        init_mean : float, optional
            Initialization value for hard concrete parameter,
            by default 0.5.,
        init_std: float, optional
            Used to initialize the hard concrete parameters,
            by default 0.01.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.zeros(in_features, out_features))  # type: ignore
        self.mask = HardConcrete(in_features, init_mean, init_std)  # type: ignore

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))  # type: ignore
        else:
            self.register_parameter("bias", None)  # type: ignore

        self.indices = None
        self.compiled_weight = None
        self.reset_parameters()

    @classmethod
    def from_module(
        cls,
        module: nn.Linear,
        init_mean: float = 0.5,
        init_std: float = 0.01,
        keep_weights: bool = True,
    ) -> "HardConcreteLinear":
        """Construct from a pretrained nn.Linear module.
        IMPORTANT: the weights are conserved, but can be reinitialized
        with `keep_weights = False`.
        Parameters
        ----------
        module: nn.Linear
            A ``nn.Linear`` module.
        init_mean : float, optional
            Initialization value for hard concrete parameter,
            by default 0.5.,
        init_std: float, optional
            Used to initialize the hard concrete parameters,
            by default 0.01.
        Returns
        -------
        HardConreteLinear
            The input module with a hardconcrete mask introduced.
        """
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None
        new_module = cls(in_features, out_features, bias, init_mean, init_std)

        if keep_weights:
            new_module.weight.data = module.weight.data.transpose(0, 1).clone()
            if bias:
                new_module.bias.data = module.bias.data.clone()

        return new_module

    def reset_parameters(self):
        """Reset network parameters."""
        self.mask.reset_parameters()
        nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def num_prunable_parameters(self) -> int:
        """Get number of prunable parameters"""
        return self.in_features * self.out_features

    def num_parameters(self, train=True) -> torch.Tensor:
        """Get number of parameters."""
        params = torch.tensor(0, dtype=torch.float).to(self.weight)
        if train:
            n_active = self.mask.l0_norm()
            params += n_active * self.out_features
        elif self.compiled_weight is not None:
            if isinstance(self.compiled_weight, int) and self.compiled_weight == -1:
                return params
            params += len(self.compiled_weight.view(-1))
        return params

    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:  # type: ignore
        """Perform the forward pass.
        Parameters
        ----------
        data : torch.Tensor
            N-dimensional tensor, with last dimension `in_features`
        Returns
        -------
        torch.Tensor
            N-dimensional tensor, with last dimension `out_features`
        """
        if self.training:
            # First reset the compiled weights
            self.compiled_weight = None
            self.indices = None

            # Sample, and compile dynamically
            mask = self.mask()
            indices = mask.data.nonzero().view(-1)
            if len(indices) == 0 or len(indices) > self.in_features * 0.8:
                compiled_weight = self.weight * mask.view(-1, 1)
                U = data.matmul(compiled_weight)
            else:
                compiled_weight = self.weight * mask.view(-1, 1)
                compiled_weight = compiled_weight.index_select(0, indices)
                U = data.index_select(-1, indices).matmul(compiled_weight)
        else:
            if self.compiled_weight is None:
                mask = self.mask()
                indices = mask.nonzero().view(-1)
                self.indices = indices

                # Compute new subweight
                if len(indices) > 0:  # type: ignore
                    weight = self.weight * mask.view(-1, 1)
                    self.compiled_weight = weight.index_select(0, indices)  # type: ignore
                else:
                    self.compiled_weight = -1  # type: ignore

            # Use the precompued sub weight
            if len(self.indices) == 0:
                output_size = data.size()[:-1] + (self.out_features,)
                U = data.new(size=output_size).zero_()  # type: ignore
            else:
                U = data.index_select(-1, self.indices).matmul(self.compiled_weight)  # type: ignore

        return U if self.bias is None else U + self.bias

    def extra_repr(self) -> str:
        s = "in_features={in_features}, out_features={out_features}"
        s += ", bias={}".format(str(self.bias is not None))
        return s.format(**self.__dict__)

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.extra_repr())