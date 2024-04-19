# from agents.base_agent import BaseAgent
# from utils.cbf_utils import lie_deriv
# import torch
# from utils.torch_utils import softmin
# from torchdiffeq import odeint
from hocbf_composition.utils import tensify
import torch
from torch.nn.functional import softplus, relu
from attrdict import AttrDict

class ClosedFormSafetyFilter:
    def __init__(self, barrier, action_dim, alpha=None, params=None):
        self._barrier = barrier
        self._action_dim = action_dim
        self._alpha = alpha if alpha is not None else lambda x: x
        self._params = params if params is not None else AttrDict(slack_gain=1e24,
                                                                  use_softplus=False,
                                                                  softplus_gain=2.0)
        self._Q = None
        self._c = None

    def assign_cost(self, Q, c):
        self._Q = Q
        self._c = c
        return self

    def safe_optimal_control(self, x):
        x = tensify(x).to(torch.float64)
        Q = self._Q(x)
        c = self._c(x)
        Q_inv = torch.inverse(Q)

        assert Q.shape == (x.shape[0], self._action_dim, self._action_dim), 'Q should be of shape (batch_size, action_dim, action_dim)'
        assert c.shape == (x.shape[0], self._action_dim), 'c should be of shape (batch_size, action_dim)'

        hocbf, Lf_hocbf, Lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)

        omega = Lf_hocbf - torch.einsum('bi,bij,bj->b', Lg_hocbf, Q_inv, c).unsqueeze(-1) + self._alpha(hocbf)
        den = torch.einsum('bi,bij,bj->b', Lg_hocbf, Q_inv, Lg_hocbf).unsqueeze(-1) + (1 / self._params.slack_gain) * hocbf ** 2
        num = softplus(-omega, beta=self._params.softplus_gain) if self._params.use_softplus else relu(-omega)
        lam = num / den

        u = - torch.einsum('bij,bi->bj', Q_inv, c - Lg_hocbf * lam)
        return u


class MinInterventionSafetyFilter(ClosedFormSafetyFilter):
    def assign_cost(self, Q, c):
        raise ('Use assign_desired_control to assign desired control.'
               ' The min intervention cost is automatically assigned.')

    def assign_desired_control(self, desired_control):
        self._Q = lambda x:  torch.eye(self._action_dim, dtype=torch.float64).repeat(x.shape[0], 1, 1)
        self._c = lambda x: -2 * desired_control(x)

# Higher Order CBF Input-Constrained Shield
# class HOCBFICAgent(BaseAgent):
#     def initialize(self, params, init_dict=None):
#         self.params = params
#
#         # Desired control function
#         self.state_dim = init_dict.state_dim
#         self.ac_dim = init_dict.ac_dim
#
#         # initialize desired control and make it a function of the augmented trajs
#         self.ac_des = lambda x: init_dict.ac_des(x[:self.state_dim])
#
#         if params.input_constraint_is_on:
#             self.ac_constrained = True
#             self.barrier = InputConstrainedSoftminHOCBF(state_constraints_info=init_dict.state_constraints_info,
#                                                         dyn_f=init_dict.dyn_f, dyn_g=init_dict.dyn_g,
#                                                         state_dim=init_dict.state_dim, softmin_gain=params.softmin_gain,
#                                                         ac_dim=init_dict.ac_dim,
#                                                         ac_constraints_info=init_dict.ac_constraints_info,
#                                                         ac_dyn_f=init_dict.ac_dyn_f, ac_dyn_g=init_dict.ac_dyn_g,
#                                                         ac_dyn_rel_deg=init_dict.ac_dyn_rel_deg,
#                                                         ac_dyn_output_func=init_dict.ac_dyn_output_func)
#
#             self.controller_state = torch.zeros(init_dict.ac_dim)
#             self.ac_dyn_f = init_dict.ac_dyn_f
#             self.ac_dyn_g = init_dict.ac_dyn_g
#
#             self.ac_dyn_rel_deg = init_dict.ac_dyn_rel_deg
#             self.gammas = init_dict.gammas
#             assert len(self.gammas) == self.ac_dyn_rel_deg
#             # Gamma corresponding to the ac_dyn_rel_deg is 1.0
#             self.gammas.append(1.0)
#
#             # initialize action output function and make it a function of the augmented trajs
#             self.ac_dyn_output_func = lambda x: init_dict.ac_dyn_output_func(x[self.state_dim:])
#
#             # make surrogate desired control
#             self.ac_des_sur = self.make_surrogate_desired_control()
#
#         else:
#             self.ac_constrained = False
#             self.barrier = InputConstrainedSoftminHOCBF(state_constraints_info=init_dict.state_constraints_info,
#                                                         dyn_f=init_dict.dyn_f, dyn_g=init_dict.dyn_g,
#                                                         state_dim=init_dict.state_dim, softmin_gain=params.softmin_gain,
#                                                         ac_dim=init_dict.ac_dim)
#
#         self.aug_dyn_f = self.barrier.aug_dyn_f
#         self.aug_dyn_g = self.barrier.aug_dyn_g
#
#         self.aug_dynamics = lambda x, u: self.aug_dyn_f(x) + self.aug_dyn_g(x) * u(x)
#
#         self.dyn_f = init_dict.dyn_f
#         self.dyn_g = init_dict.dyn_g
#
#         # make safe control
#         self.safe_control = self.make_safe_control()
#
#
#
#
#
#     def step_forward_control(self, x):
#         # TODO: step
#         pass
#
#
#     def make_surrogate_desired_control(self):
#         ac_des_lie_deriv_series = self._get_high_order_lie_deriv_series(func=self.ac_des,
#                                                                         field=self.aug_dyn_f,
#                                                                         order=self.ac_dyn_rel_deg)
#
#         ac_dyn_output_func_deriv_series = self._get_high_order_lie_deriv_series(func=self.ac_dyn_output_func,
#                                                                                 field=self.aug_dyn_f,
#                                                                                 order=self.ac_dyn_rel_deg)
#
#         temp1 = lambda x: torch.vstack([gamma*(ad(x) - hc(x))
#                      for gamma, ad, hc in zip(self.gammas, ac_des_lie_deriv_series, ac_dyn_output_func_deriv_series)]).sum()
#
#         temp2 = lambda x: lie_deriv(x, ac_dyn_output_func_deriv_series[self.ac_dyn_rel_deg], self.aug_dyn_g)
#
#         # TODO: check
#         return lambda x: torch.matmul(torch.inverse(temp2(x)), temp1(x))
#
#     def make_safe_control(self):
#         ac_des = self.ac_des_sur if self.ac_constrained else self.ac_des
#         h, Lfh, Lgh = lambda x: self.barrier.get_barrier_and_lie_derivs(x)
#         omega = lambda x: Lfh(x) + torch.matmul(Lgh(x), ac_des(x)) + self.params.alpha * h(x)
#         beta = lambda x: torch.where(omega(x) >= 0.0, torch.zeros(1),
#                                      -omega(x)/(torch.matmul(Lgh(x), Lgh(x).t()) + h(x) ** 2 / self.params.slack_gain))
#
#         return lambda x: ac_des(x) + torch.matmul(Lgh(x).t, beta(x))
#
#     def overwrite_controller_state(self, controller_state):
#         self.controller_state = controller_state
#
#     @torch.enable_grad()
#     def act(self, obs, explore=True, init_phase=False):
#         # TODO: convert observation to tensor
#         obs = torch.tensor(obs)
#         if self.ac_constrained:
#             trajs = torch.stack([obs, self.controller_state], dim=-1)
#             next_state = odeint(
#                     lambda t, y: self.aug_dynamics(y, self.safe_control(y)), trajs, torch.tensor([0, self.params.timestep]))
#             self.controller_state = next_state[self.state_dim:]
#             ac = self.ac_dyn_output_func(self.controller_state)
#         else:
#             ac = self.safe_control(obs)
#
#         # TOOD: check info's role here. also in the base agent class
#         return ac.squeeze(axis=0) if (ac.ndim > 1 and ac.shape[0] == 1) else ac, None
#
#     def _get_high_order_lie_deriv_series(self, func, field, order):
#         ans = [func]
#         for i in range(order):
#             ans.append(lambda x: (lie_deriv(x, ans[i], field)).squeeze())
#         return ans
#
# class InputConstrainedSoftminHOCBF:
#     # state_constraint_info and input_constraints_into are expected to be a dictionary
#     def __init__(self, state_constraints_info, dyn_f, dyn_g, state_dim, softmin_gain,
#                  ac_dim, ac_constraints_info=None,
#                  ac_dyn_f=None, ac_dyn_g=None, ac_dyn_rel_deg=None,
#                  ac_dyn_output_func=None):
#         # Expected a list of dictionaries for state_constraints_info and action_constraints_info
#         # state_constriants_info has 3 keys: func, rel_deg, alpha_coefs
#         # ac_constriants_info has 2 keys: func, alpha_coefs
#
#         self.ac_constrained = False
#         if ac_constraints_info is not None:
#             self.ac_constrained = True
#
#
#         self.state_constraints_info = state_constraints_info
#         self.ac_constraints_info = ac_constraints_info
#         self.dyn_f = dyn_f
#         self.dyn_g = dyn_g
#         self.ac_dyn_f = ac_dyn_f
#         self.ac_dyn_g = ac_dyn_g
#         self.ac_dyn_output_func = ac_dyn_output_func
#         self.state_dim = state_dim
#         self.ac_dim = ac_dim
#         self.softmin_gain = softmin_gain
#
#         self.ac_dyn_output_func = 0 if ac_constraints_info is None else ac_dyn_rel_deg
#
#         # make input constrained softmin hocbf and
#         self.cbfs = []
#         self.hocbfs = []
#
#         # make trajs constraints
#         for constraint in self.state_constraints_info:
#             assert len(constraint['alpha_coefs']) == constraint['rel_deg'] - 1
#             self.cbfs.append(self._get_hocbf_series(h=lambda x: constraint['func'](x[:self.state_dim]),
#                                                     rel_deg=constraint['rel_deg'] + self.ac_dyn_output_func,
#                                                     alpha_coefs=constraint['alpha_coefs']))
#             self.hocbfs.append(self.cbfs[-1][-1])
#
#         # make action constraints
#         if self.ac_constrained:
#             for constraint in self.ac_constraints_info:
#                 assert len(ac_constraints_info['alpha_coefs']) == self.ac_dyn_output_func - 1
#                 self.cbfs.append(self._get_hocbf_series(h=lambda x: constraint['func'](self.ac_dyn_output_func(x[self.state_dim:])),
#                                                         rel_deg=self.ac_dyn_output_func,
#                                                         alpha_coefs=constraint['alpha_coefs']))
#                 self.hocbfs.append(self.cbfs[-1][-1])
#
#
#     def barrier(self, x):
#         return softmin(torch.stack([hocbf(x) for hocbf in self.hocbfs]), self.softmin_gain)
#
#     def get_lie_derivs(self, x):
#         return lie_deriv(x, self.barrier, self.aug_dyn_f), lie_deriv(x, self.barrier, self.aug_dyn_g)
#
#     def get_barrier_and_lie_derivs(self, x):
#         return self.barrier(x), *self.get_lie_derivs(x)
#
#     def aug_dyn_f(self, x):
#         if self.ac_constrained:
#             x_hat, x_c = torch.split(x, self.state_dim)
#             # TODO: check matrix multiplication. Check for batch
#             return torch.vstack([self.dyn_f(x_hat) + self.dyn_g(x_hat) * self.ac_dyn_output_func(x_c), self.ac_dyn_f(x_c)])
#         return self.dyn_f(x)
#
#     def aug_dyn_g(self, x):
#         if self.ac_constrained:
#             x_hat, x_c = torch.split(x, self.state_dim)
#             # TODO: check matrix multiplication. Check for batch
#             return torch.vstack([torch.zeros(self.state_dim, self.ac_dim), self.ac_dyn_g(x_c)])
#         return self.dyn_g(x)
#
#     def _get_hocbf_series(self, h, rel_deg, alpha_coefs):
#         ans = [h]
#         for i in range(rel_deg - 1):
#             ans.append(lambda x: (lie_deriv(x, ans[i], self.aug_dyn_f) + alpha_coefs[i] * ans[i](x)).squeeze())
#         return ans
#
#     def check_constraints(self, x):
#         x.requires_grad_(requires_grad=True)
#         ans = []
#         satisfied = True
#         for i, series in enumerate(self.cbfs):
#             ans.append([1] * len(series))
#             for j, barrier in enumerate(series):
#                 if barrier(x) >= 0:
#                     ans[i][j] = 0
#                 else:
#                     satisfied = False
#
#         return satisfied, ans
