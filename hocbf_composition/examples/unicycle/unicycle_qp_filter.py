from attrdict import AttrDict as AD
import matplotlib as mpl
from math import pi
from hocbf_composition.examples.unicycle.unicycle_dynamics import UnicycleDynamics
from hocbf_composition.utils.make_map import Map
from hocbf_composition.examples.unicycle.map_config import map_config
from hocbf_composition.examples.unicycle.unicycle_desired_control import desired_control
from hocbf_composition.barrier import MultiBarriers
from time import time
import datetime
import matplotlib.pyplot as plt
import torch

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'Times'

# Control gains

# Barrier configs
cfg = AD(softmax_rho=20,
         softmin_rho=20,
         pos_barrier_rel_deg=2,
         vel_barrier_rel_deg=1,
         obstacle_alpha=[2.5],
         boundary_alpha=[1.0],
         velocity_alpha=[],
         )

# Instantiate dynamics
dynamics = UnicycleDynamics(state_dim=4, action_dim=2)

# Make barrier from map_
map_ = Map(barriers_info=map_config, dynamics=dynamics, cfg=cfg)

pos_barrier, vel_barrier = map_.get_barriers()

barrier = MultiBarriers()
barrier.add_barriers([*pos_barrier, *vel_barrier], infer_dynamics=True)


x0 = torch.tensor([-1.0, -8.5, 0.0, pi / 2], dtype=torch.float64).repeat(4, 1)

# print(barrier.compute_barriers_at(x0))
# print(barrier.barrier(x0))
print(barrier.hocbf(x0))
print(barrier.Lf_hocbf(x0))
print(barrier.Lg_hocbf(x0))
print(barrier.get_hocbf_and_lie_derivs(x0))
