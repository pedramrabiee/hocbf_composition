import torch
from hocbf_composition.utils import vectorize_tensors


class AffineInControlDynamics:
    def __init__(self, state_dim, action_dim, params=None, **kwargs):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._params = params

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
        output: batch size x state_dim
        """
        raise NotImplementedError

    def _g(self, x):
        """
        x: batch size x state_dim
        output: batch size x state_dim x action_dim
        """
        raise NotImplementedError

    def rhs(self, x, action):
        """
        Right-hand-side of dynamics
        """
        action = vectorize_tensors(action)
        x = vectorize_tensors(x)
        return self.f(x) + torch.bmm(self.g(x), action.unsqueeze(-1)).squeeze(-1)

    def rhs_zoh(self, t, x, action_func, timestep):
        """
        This function simulate the zero-order-hold dynamics,
         by calling the action function at each timestep, and hold it during the timestep
        """
        if t == 0.0:
            self._last_action_time = 0.0
            self._last_action = None  # Initialize as None

        # Adjust the time for alignment with the ZOH timestep
        t_adjusted = timestep * (t // timestep)

        # Check if we need to recalculate the action
        if t_adjusted != self._last_action_time or self._last_action is None:
            self._last_action_time = t_adjusted
            self._last_action = action_func(x)

        # Apply the ZOH dynamics using the last calculated action
        return self.rhs(x, self._last_action)

    def set_f(self, f):
        if not callable(f):
            raise TypeError("_f must be a callable function")
        self._f = f

    def set_g(self, g):
        if not callable(g):
            raise TypeError("_g must be a callable function")
        self._g = g

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def action_dim(self):
        return self._action_dim


class LowPassFilterDynamics(AffineInControlDynamics):
    def __init__(self, state_dim, action_dim, params):
        assert state_dim == action_dim, 'state_dim and action_dim should be the same'
        super().__init__(state_dim, action_dim, params)
        assert params is not None, 'params should include low pass filter gains'
        assert len(params['gains']) == state_dim, 'gains should be a list of gains of length state_dim'

        self._gains = torch.tensor(params['gains'])
        self._gains_mat = torch.diag(self._gains)
        self._gains.unsqueeze_(0)

    def _f(self, x):
        return -self._gains * x

    def _g(self, x):
        return self._gains_mat.repeat(x.shape[0], 1, 1)
