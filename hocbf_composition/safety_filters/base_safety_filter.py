class BaseSafetyFilter:
    def __init__(self, action_dim, alpha=None, params=None):
        self._action_dim = action_dim
        self._alpha = alpha if alpha is not None else lambda x: x
        self._params = params
        self._dynamics = None
        self._barrier = None
        self._Q = None
        self._c = None

    def assign_state_barrier(self, barrier):
        raise NotImplementedError

    def assign_dynamics(self, dynamics):
        raise NotImplementedError

    def assign_cost(self, Q, c):
        self._Q = Q
        self._c = c
        return self

    def safe_optimal_control(self, x):
        raise NotImplementedError

    def get_safe_optimal_trajs(self, x0, timestep=0.001, sim_time=4.0, method='dopri5'):
        # TODO: move get_safe_optimal_trajs_here
        raise NotImplementedError

