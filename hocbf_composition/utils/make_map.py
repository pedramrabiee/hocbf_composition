from attrdict import AttrDict
from barriers.barrier import Barrier
from barriers.composite_barrier import SoftCompositionBarrier
from hocbf_composition.utils.utils import *


class Map:
    def __init__(self, barriers_info, dynamics, cfg):
        self.barriers_info = AttrDict(barriers_info)
        self.dynamics = dynamics
        self.cfg = AttrDict(cfg)
        self.make_barrier_from_map()

    def make_barrier_from_map(self):
        self.pos_barriers = self.make_position_barrier_from_map()
        self.vel_barriers = self.make_velocity_barrier()
        self.barrier = SoftCompositionBarrier(
            cfg=self.cfg).assign_barriers_and_rule(barriers=[*self.pos_barriers, *self.vel_barriers],
                                                   rule='i',
                                                   infer_dynamics=True)

    def get_barriers(self):
        return self.pos_barriers, self.vel_barriers

    def make_position_barrier_from_map(self):
        geoms = self.barriers_info.geoms
        barriers = []

        for geom_type, geom_info in geoms:

            if geom_type == 'cylinder':
                barrier_func = make_circle_barrier_functional
                alphas = make_linear_alpha_function_form_list_of_coef(self.cfg.obstacle_alpha)
            elif geom_type == 'box':
                barrier_func = make_affine_rectangular_barrier_functional
                alphas = make_linear_alpha_function_form_list_of_coef(self.cfg.obstacle_alpha)
            elif geom_type == 'norm_box':
                barrier_func = make_norm_rectangular_barrier_functional
                alphas = make_linear_alpha_function_form_list_of_coef(self.cfg.obstacle_alpha)
            elif geom_type == 'boundary':
                barrier_func = make_affine_rectangular_boundary_functional
                alphas = make_linear_alpha_function_form_list_of_coef(self.cfg.boundary_alpha)
            elif geom_type == 'norm_boundary':
                barrier_func = make_norm_rectangular_boundary_functional
                alphas = make_linear_alpha_function_form_list_of_coef(self.cfg.boundary_alpha)
            else:
                raise NotImplementedError
            barriers.append(
                Barrier().assign(barrier_func=barrier_func(**geom_info),
                                 rel_deg=self.cfg.pos_barrier_rel_deg,
                                 alphas=alphas).assign_dynamics(self.dynamics))
        return barriers

    def make_velocity_barrier(self):
        alphas = make_linear_alpha_function_form_list_of_coef(self.cfg.velocity_alpha)
        idx, bounds = self.barriers_info.velocity
        vel_barriers = make_box_barrier_functionals(bounds=bounds, idx=idx)
        barriers = [Barrier().assign(
            barrier_func=vel_barrier,
            rel_deg=self.cfg.vel_barrier_rel_deg,
            alphas=alphas).assign_dynamics(self.dynamics) for vel_barrier in vel_barriers]

        return barriers
