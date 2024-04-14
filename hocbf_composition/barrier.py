from typing import List
from hocbf_composition.utils import *
# import torch.jit


class Barrier:
    def __init__(self, cfg=None):
        """
        Initialize Barrier class.

        Parameters:
        - cfg (optional): Configuration parameters.
        """
        self.cfg = cfg
        self._barrier_func = None
        self._barriers = None
        self._hocbf = None
        self._rel_deg = None

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
        if rel_deg > 1:
            if alphas is None:
                alphas = [lambda x: x for _ in range(rel_deg)]
            assert isinstance(alphas, list) and len(alphas) == rel_deg and callable(alphas[0]),\
                "alphas must be a list with length equal to rel_deg of callable functions "

        # Assign barrier function definition
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
        self._barriers = self._get_hocbf_series(h=self.h, rel_deg=self._rel_deg, alphas=self._alphas)
        self._hocbf_func = self._barriers[-1]
        return self

    def h(self, x):
        """
        Compute the barrier function h(x) for a given state x.
        """
        return apply_and_batchize(self._barrier_func, x)

    def hocbf(self, x):
        """
        Compute the highest-order barrier function hocbf(x) for a given state x.
        """
        return apply_and_batchize(self._hocbf_func, x)

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
         Get the list of barrier functions.
        """
        return self._barriers

    @property
    def dynamics(self):
        """
        Get the dynamics associated with the system.
        """
        return self._dynamics

    # Helper methods
    def _get_hocbf_series(self, h, rel_deg, alphas):
        """
              Generate a series of higher-order barrier functions.

              Parameters:
              - h: Initial barrier function.
              - rel_deg: Relative degree of the barrier.
              - alphas: List of class-K functions.

          """
        ans = [h]
        for i in range(rel_deg - 1):
            hocbf_i = lambda x, hocbf=ans[i], f=self._dynamics.f, alpha=alphas[i]: \
                lie_deriv(x, hocbf, f) + apply_and_batchize(func=alpha, x=hocbf(x))
            ans.append(hocbf_i)
        return ans

    def compute_barriers_at(self, x):
        """
        Compute barrier values at a given state x.
        """
        return [apply_and_batchize(func=barrier, x=x) for barrier in self._barriers]

    def get_min_barrier_at(self, x):
        """
        Get the minimum barrier value at a given state x.
        """
        return torch.min(torch.hstack(self.compute_barriers_at(x)), dim=-1).values.unsqueeze(-1)

    def append_to_barriers(self, barriers):
        """
        Append additional barriers to the existing list.

        This method is particularly useful when creating a new barrier based on an existing one.
        It allows you to incorporate the barriers of the original barrier into the new one.
        """
        # Push barriers from the left
        self._barriers = [*barriers, *self._barriers]


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
        Override the assign_dynamics method to raise an error.
        Dynamics are inferred from the barriers after calling the assign_barriers_and_rule method.
        """
        raise 'For the CompositionBarrier class, dynamics are inferred from the barriers after calling the assign_barriers_and_rule method'

    def assign_barriers_and_rule(self, barriers: List[Barrier], rule: str):
        """
        Assign multiple barriers and a composition rule to the CompositionBarrier object.

        Parameters:
            - barriers: List of Barrier objects to be composed.
            - rule: Composition rule to be applied. Choose from 'intersection', 'union', 'i' or 'u'

        Returns:
            - self: Updated CompositionBarrier object.
        """
        # Infer dynamics from the barriers
        self._dynamics = barriers[0].dynamics

        # Define barrier functions and higher-order barrier function as compositions of individual barrier functions
        self._barrier_func = None
        self._barrier_funcs = lambda x: torch.hstack([barrier.h(x) for barrier in barriers])
        # self._barrier_func = lambda x: self.compose(rule)(torch.hstack([barrier.h(x) for barrier in barriers]))
        self._hocbf_func = lambda x: self.compose(rule)(torch.hstack([barrier.hocbf(x) for barrier in barriers]))

        # Concatenate individual barriers into a single list
        self._barriers = []
        for barrier in barriers:
            self._barriers += barrier.barriers

        # Append the higher-order composite barrier function to the list of barriers
        self._barriers.append(self._hocbf_func)

        return self

    def h(self, x):
        """
        Compute the barrier function h(x) for a given state x.
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


def make_barrier_from_barrier(barrier, rel_deg=1):
    """
        Create a new barrier based on an existing barrier.

        This method constructs a new Barrier object using the highest-order barrier of the
        input barrier and assigns the same dynamics. The existing barriers are also appended
        to the new barrier.

        Parameters:
        - barrier: Existing Barrier object.
        - rel_deg: Relative degree of the new barrier.

        Returns:
        - new_barrier: New Barrier object with the specified relative degree and composed of the input barrier's components.
    """
    new_barrier = Barrier().assign(barrier_func=barrier.hocbf, rel_deg=rel_deg).assign_dynamics(barrier.dynamics)
    new_barrier.append_to_barriers(barrier.barriers)
    return new_barrier