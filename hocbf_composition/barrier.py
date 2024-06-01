from typing import List
from hocbf_composition.utils import *


class Barrier:
    def __init__(self, cfg=None):
        """
        Initialize Barrier class.

        Parameters:
        - cfg (optional): Configuration parameters.
        """
        self._dynamics = None
        self.cfg = cfg
        self._barrier_func = None
        self._barriers = None
        self._hocbf = None
        self._rel_deg = None
        self._hocbf_func = None
        self._alphas = None

    def assign(self, barrier_func, rel_deg=1, alphas=None):
        """
        Assign barrier function to the Barrier object.

        Parameters:
        - barrier_func: Barrier function to be assigned.
        - rel_deg: Relative degree of the barrier function.
        - alphas: List of class-K functions for higher-order barriers.

        Returns:
        - self: Updated Barrier object.
        """
        assert callable(barrier_func), "barrier_func must be a callable function"
        alphas = self._handle_alphas(alphas=alphas, rel_deg=rel_deg)

        # Assign barrier function definition: self._barrier_func is the main constraint that the
        # higher-order cbf is defined based on
        self._barrier_func = barrier_func
        self._rel_deg = rel_deg
        self._alphas = alphas
        return self

    def assign_dynamics(self, dynamics):
        """
         Assign dynamics to the Barrier object and generate higher-order barrier functions.

         Returns:
         - self: Updated Barrier object.
         """
        assert self._barrier_func is not None, \
            "Barrier functions must be assigned first. Use the assign method"

        self._dynamics = dynamics
        # make higher-order barrier function
        self._barriers = self._make_hocbf_series(barrier=self.barrier, rel_deg=self._rel_deg, alphas=self._alphas)
        self._hocbf_func = self._barriers[-1]
        return self

    def raise_rel_deg(self, x, raise_rel_deg_by=1, alphas=None):
        """
        This method takes the current hocbf and make a new hocbf with the relative degree raised
        by `raise_rel_deg_by`. The new hocbf has the relative degree of old rel_deg + raise_rel_deg_by
        """

        alphas = self._handle_alphas(alphas=alphas, rel_deg=raise_rel_deg_by)
        self._alphas.append(alphas)
        self._rel_deg += raise_rel_deg_by

        self._barriers.append(*self._make_hocbf_series(barrier=self._hocbf_func,
                                                       rel_deg=raise_rel_deg_by,
                                                       alphas=alphas))
        self._hocbf_func = self._barriers[-1]

    def barrier(self, x):
        """
        Compute the barrier function barrier(x) for a given trajs x.
        """
        return apply_and_batchize(self._barrier_func, x)

    def hocbf(self, x):
        """
        Compute the highest-order barrier function hocbf(x) for a given trajs x.
        """
        return apply_and_batchize(self._hocbf_func, x)

    def get_hocbf_and_lie_derivs(self, x):
        x = vectorize_tensors(x)
        grad_req = x.requires_grad
        x.requires_grad_()
        hocbf = self.hocbf(x)
        func_val = hocbf.sum(0)
        hocbf_deriv = [grad(fval, x, retain_graph=True)[0] for fval in func_val]
        x.requires_grad_(requires_grad=grad_req)
        Lf_hocbf = lie_deriv_from_values(x, hocbf_deriv, self._dynamics.f(x))
        Lg_hocbf = lie_deriv_from_values(x, hocbf_deriv, self._dynamics.g(x))
        return hocbf.detach(), Lf_hocbf, Lg_hocbf

    def get_hocbf_and_lie_derivs_v2(self, x):
        # For a more optimized version use get_hocbf_and_lie_derivs
        return self.hocbf(x), self.Lf_hocbf(x), self.Lg_hocbf(x)

    def Lf_hocbf(self, x):
        """
        Compute the Lie derivative of the highest-order barrier function with respect to the system dynamics f.
        """
        return lie_deriv(x, self.hocbf, self._dynamics.f)

    def Lg_hocbf(self, x):
        """
        Compute the Lie derivative of the highest-order barrier function with respect to the system dynamics g.
        """
        return lie_deriv(x, self.hocbf, self._dynamics.g)

    def compute_barriers_at(self, x):
        """
        Compute barrier values at a given trajs x.
        """
        return [apply_and_batchize(func=barrier, x=x) for barrier in self.barriers_flatten]

    def get_min_barrier_at(self, x):
        """
        Get the minimum barrier value at a given trajs x.
        """
        return torch.min(torch.hstack(self.compute_barriers_at(x)), dim=-1).values.unsqueeze(-1)

    def min_barrier(self, x):
        """
        Calculate the minimum value among all the barrier values computed at point x.
        """
        return torch.min(self.barrier(x), dim=-1).values.unsqueeze(-1)

    # Getters
    @property
    def rel_deg(self):
        """
        Get the relative degree of the barrier.
        """
        return self._rel_deg

    @property
    def barriers(self):
        """
         Get the list of barrier functions of all relative degrees upto self.rel_deg
        """
        return self._barriers

    @property
    def barriers_flatten(self):
        """
             Get the flatten list of barrier functions of all relative degrees. This method has application mainly
             in the composite barrier function class
        """
        return self.barriers

    @property
    def dynamics(self):
        """
        Get the dynamics associated with the system.
        """
        return self._dynamics

    @property
    def num_barriers(self):
        return len(self.barriers_flatten)

    # Helper methods

    def _make_hocbf_series(self, barrier, rel_deg, alphas):
        """
              Generate a series of higher-order barrier functions.

              Parameters:
              - barrier: Initial barrier function.
              - rel_deg: Relative degree of the barrier.
              - alphas: List of class-K functions.

          """
        ans = [barrier]
        for i in range(rel_deg - 1):
            hocbf_i = lambda x, hocbf=ans[i], f=self._dynamics.f, alpha=alphas[i]: \
                lie_deriv(x, hocbf, f) + apply_and_batchize(func=alpha, x=hocbf(x))
            ans.append(hocbf_i)
        return ans

    def _handle_alphas(self, alphas, rel_deg):
        if rel_deg > 1:
            if alphas is None:
                alphas = [(lambda x: x) for _ in range(rel_deg - 1)]
            assert isinstance(alphas, list) and len(alphas) == rel_deg - 1 and callable(alphas[0]), \
                "alphas must be a list with length equal to (rel_deg - 1) of callable functions "
        return alphas


class CompositionBarrier(Barrier):
    """
    CompositionBarrier class, inherits from Barrier.
    This class represents a barrier formed by composing multiple barriers with a specific rule.
    """

    def assign(self, barrier_func, rel_deg=1, alphas=None):
        """
        Override the assign method to raise an error.
        Assignment should be made through the assign_barriers_and_rule method.
        """
        raise 'For the CompositionBarrier class, assignment should be made through the assign_barriers_and_rule method'

    def assign_dynamics(self, dynamics):
        """
        Assign dynamics
        """
        self._dynamics = dynamics
        return self

    # @torch.jit.script
    def assign_barriers_and_rule(self, barriers: List[Barrier], rule: str, infer_dynamics: bool = False):
        """
        Assign multiple barriers and a composition rule to the CompositionBarrier object.

        Parameters:
            - barriers: List of Barrier objects to be composed.
            - rule: Composition rule to be applied. Choose from 'intersection', 'union', 'i' or 'u'

        Returns:
            - self: Updated CompositionBarrier object.
        """
        # Infer dynamics from the barriers
        if infer_dynamics:
            if self._dynamics is not None:
                raise Warning('The assinged dynamics is overriden by the dynamics of the'
                              ' first barrier on the barriers list')
            self._dynamics = barriers[0].dynamics
        elif self._dynamics is None:
            raise ('Dynamics should be assigned. Use infer_dynamics=True to take the dynamics of'
                   ' the first barrier as the dynamics of the composition barrier')

        self._rel_deg = 1

        # Define barrier functions and higher-order barrier function as compositions of individual barrier functions
        self._barrier_func = None
        self._barrier_funcs = lambda x: torch.hstack([barrier.barrier(x) for barrier in barriers])
        self._hocbf_func = lambda x: self.compose(rule)(torch.hstack([barrier.hocbf(x) for barrier in barriers]))

        # Concatenate barrier.barriers for all barrier in barriers. This makes a list of lists.
        self._barriers = [barrier.barriers for barrier in barriers]

        # Append the higher-order composite barrier function to the list of barriers
        self._barriers.append([self._hocbf_func])

        return self

    def raise_rel_deg(self, x, raise_rel_deg_by=1, alphas=None):
        """
        This method takes the current hocbf and make a new hocbf with the relative degree raised
        by `raise_rel_deg_by`. The new hocbf has the relative degree of old rel_deg + raise_rel_deg_by
        """

        alphas = self._handle_alphas(alphas=alphas, rel_deg=raise_rel_deg_by)
        self._alphas.append(alphas)
        self._rel_deg += raise_rel_deg_by
        new_barriers = self._make_hocbf_series(barrier=self._hocbf_func,
                                               rel_deg=raise_rel_deg_by,
                                               alphas=alphas)

        self._hocbf_func = new_barriers[-1]
        self._barriers.append(new_barriers)

    def barrier(self, x):
        """
        Compute main barrier value at x. Main barrier value is the barrier which defines all the
         higher order cbfs involved in the composite barrier function expression.
         This method returns a horizontally stacked torch tensor of the value of barriers at x.
        """
        return apply_and_batchize(self._barrier_funcs, x)

    def compose(self, c_key: str):
        """
       Select the appropriate composition rule based on the provided key.

       Parameters:
           - c_key: Composition rule key.
        """
        if c_key == 'union' or c_key == 'u':
            return self.union_rule
        if c_key == 'intersection' or c_key == 'i':
            return self.intersection_rule

        return "composition rule is not valid"

    def union_rule(self, x):
        """
        Union composition rule placeholder.
        """
        raise NotImplementedError

    def intersection_rule(self, x):
        """
        Intersection composition rule placeholder.
        """
        raise NotImplementedError

    @property
    def barriers_flatten(self):
        return [b for barrier in self._barriers for b in barrier]


class SoftCompositionBarrier(CompositionBarrier):
    """
    SoftCompositionBarrier class, inherits from CompositionBarrier.
    This class represents a soft composition of multiple barriers with specific soft composition rules.
    """

    def union_rule(self, x):
        return apply_and_match_dim(lambda y: softmax(y, rho=self.cfg.softmax_rho, dim=-1), x)

    def intersection_rule(self, x):
        return apply_and_match_dim(lambda y: softmin(y, rho=self.cfg.softmin_rho, dim=-1), x)


class NonSmoothCompositionBarrier(CompositionBarrier):
    """
    NonSmoothCompositionBarrier class, inherits from CompositionBarrier.
    This class represents a non-smooth composition of multiple barriers with specific non-smooth composition rules.
    """

    def union_rule(self, x):
        return apply_and_match_dim(lambda y: torch.max(y, dim=-1).values, x)

    def intersection_rule(self, x):
        return apply_and_match_dim(lambda y: torch.min(y, dim=-1).values, x)
