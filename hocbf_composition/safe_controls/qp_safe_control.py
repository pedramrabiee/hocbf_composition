from qpth.qp import QPFunction
from hocbf_composition.safe_controls.base_safe_control import BaseSafeControl, BaseMinIntervSafeControl
from hocbf_composition.utils.utils import *
from hocbf_composition.safe_controls.closed_form_safe_control import InputConstCFSafeControl


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


class InputConstQPSafeControl(QPSafeControl):
    def assign_control_bounds(self, low: list, high: list):
        assert len(low) == len(high), 'low and high should have the same length'
        assert len(low) == self._action_dim, 'low and high length should match action dimension'

        ac_low_G = lambda x: -torch.eye(self._action_dim, dtype=torch.float64).repeat(x.shape[0], 1, 1)
        ac_low_h = lambda x: -torch.tensor(low).to(torch.float64).repeat(x.shape[0], 1)

        ac_high_G = lambda x: -ac_low_G(x)
        ac_high_h = lambda x: torch.tensor(high).to(torch.float64).repeat(x.shape[0], 1)

        self._ac_G = lambda x: torch.cat([ac_low_G(x), ac_high_G(x)], dim=1)
        self._ac_h = lambda x: torch.cat([ac_low_h(x), ac_high_h(x)], dim=1)

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

        ac_G = self._ac_G(x)
        ac_h = self._ac_h(x)

        u = QPFunction()(Q, c,
                         torch.cat([-Lg_hocbf, ac_G], dim=1),
                         torch.cat([(Lf_hocbf + self._alpha(hocbf)).squeeze(-1), ac_h], dim=1),
                         A, b)
        return u


class MinIntervInputConstQPSafeControl(MinIntervQPSafeControl,
                                       InputConstQPSafeControl):
    pass
