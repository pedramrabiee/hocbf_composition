from hocbf_composition.utils.utils import *

class BaseSafeControl:
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
        return get_trajs_from_action_func(x0=x0, dynamics=self._dynamics,
                                          action_func=self.safe_optimal_control,
                                          timestep=timestep, sim_time=sim_time,
                                          method=method)

    def get_safe_optimal_trajs_zoh(self, x0, timestep=0.001, sim_time=4.0, method='dopri5'):
        return get_trajs_from_action_func_zoh(x0=x0, dynamics=self._dynamics,
                                              action_func=self.safe_optimal_control,
                                              timestep=timestep, sim_time=sim_time,
                                              method=method)

class BaseMinIntervSafeControl(BaseSafeControl):
    def assign_desired_control(self, desired_control):
        self._desired_control = desired_control
        return self

    def assign_cost(self, Q, c):
        raise ('Use assign_desired_control to assign desired control.'
               ' The min intervention cost is automatically assigned.')
