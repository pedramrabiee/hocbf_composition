import torch
from hocbf_composition.utils import vectorize_tensors

class AffineInControlDynamics:
    def __init__(self, state_dim, action_dim):
        self._state_dim = state_dim
        self._action_dim = action_dim

    def f(self, x):
        x = vectorize_tensors(x)
        assert x.shape[1] == self._state_dim
        out = self._f(x)
        assert out.shape == x.shape
        return out

    def g(self, x):
        x = vectorize_tensors(x)
        assert x.shape[1] == self._state_dim
        out = self._g(x)
        assert out.shape[0] == x.shape[0]
        assert out.shape[-1] == self._action_dim
        assert out.shape[-2] == self._state_dim
        return out

    def _f(self, x):
        """
        x: batch size x trajs dim
        output: batch size x trajs dim
        """
        raise NotImplementedError

    def _g(self, x):
        """
        x: batch size x state_dim
        output: batch size x trajs dim x action dim
        """
        raise NotImplementedError

    def rhs(self, x, action):
        action = vectorize_tensors(action)
        x = vectorize_tensors(x)
        return self.f(x) + torch.bmm(self.g(x), action.unsqueeze(-1)).squeeze(-1)

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def action_dim(self):
        return self._action_dim


