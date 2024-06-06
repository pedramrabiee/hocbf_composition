from qpth.qp import QPFunction
from hocbf_composition.safe_controls.base_safe_control import BaseSafeControl, BaseMinIntervSafeControl
from hocbf_composition.utils.utils import *
class QPSafeControl(BaseSafeControl):
    def __init__(self, action_dim, alpha=None, params=None):
        super().__init__(action_dim, alpha, params)

    def assign_state_barrier(self, barrier):
        self._barrier = barrier
        return self

    def assign_dynamics(self, dynamics):
        self._dynamics = dynamics
        return self

    def safe_optimal_control(self, x):
        x = tensify(x).to(torch.float64)
        Q = self._Q(x)
        c = self._c(x)

        assert Q.shape == (
            x.shape[0], self._action_dim, self._action_dim), 'Q should be of shape (batch_size, action_dim, action_dim)'
        assert c.shape == (x.shape[0], self._action_dim), 'c should be of shape (batch_size, action_dim)'

        hocbf, Lf_hocbf, Lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)

        A = torch.empty(Q.shape[0], 0, Q.shape[1])
        b = torch.empty(Q.shape[0], 0)

        u = QPFunction()(Q, c, -Lg_hocbf, (Lf_hocbf + self._alpha(hocbf)).squeeze(-1), A, b)
        return u

class MinIntervQPSafeControl(BaseMinIntervSafeControl, QPSafeControl):
    def assign_desired_control(self, desired_control):
        self._Q = lambda x: 2 * torch.eye(self._action_dim, dtype=torch.float64).repeat(x.shape[0], 1, 1)
        self._c = lambda x: -2 * desired_control(x)