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
from hocbf_composition.safe_controls.qp_safe_control import MinIntervInputConstQPSafeControl
from hocbf_composition.utils.utils import *

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


safety_filter = MinIntervInputConstQPSafeControl(
    action_dim=dynamics.action_dim,
    alpha=lambda x: 0.5 * x,
    params=AD(slack_gain=1e24,
              use_softplus=False,
              softplus_gain=2.0)
).assign_dynamics(dynamics=dynamics).assign_state_barrier(barrier=barrier
                                                          ).assign_control_bounds(low=[-4.0, -1.0], high=[4.0, 1.0])

# Goal positions
goal_pos = torch.tensor([
    [3.0, 4.5],
    [-7.0, 0.0],
    [7.0, 1.5],
    [-1.0, 7.0]
])

# Initial Conditions
x0 = torch.tensor([-1.0, -8.5, 0.0, pi / 2]).repeat(goal_pos.shape[0], 1)
timestep = 0.1
sim_time = 20.0

# assign desired control based on the goal positions
safety_filter.assign_desired_control(
    desired_control=lambda x: vectorize_tensors(partial(desired_control, goal_pos=goal_pos)(x))
)

# Simulate trajectories
start_time = time()
trajs = safety_filter.get_safe_optimal_trajs(x0=x0, sim_time=sim_time, timestep=timestep, method='euler')
print(time() - start_time)

# Rearrange trajs
trajs = [torch.vstack(t.split(dynamics.state_dim)) for t in torch.hstack([tt for tt in trajs])]

# Get actions values along the trajs
actions = []
for i, traj in enumerate(trajs):
    safety_filter.assign_desired_control(
        desired_control=lambda x: vectorize_tensors(
            partial(desired_control, goal_pos=goal_pos[i].repeat(x.shape[0], 1))(x))
    )
    actions.append(safety_filter.safe_optimal_control(traj))

# Get hocbf valus along trajs
h_vals = [barrier.hocbf(traj).min(1).values for traj in trajs]

############
#  Plots   #
############

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

x = np.linspace(-10.5, 10.5, 500)
y = np.linspace(-10.5, 10.5, 500)
X, Y = np.meshgrid(x, y, )
points = np.column_stack((X.flatten(), Y.flatten()))
points = np.column_stack((points, np.zeros(points.shape)))
points = torch.tensor(points, dtype=torch.float32)
# print(barriers[0].barrier(points))
Z = map_.barrier.min_barrier(points)
Z = Z.reshape(X.shape)
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=[0], colors='k')
plt.xlabel(r'$q_x$')
plt.ylabel(r'$q_y$')

plt.plot(trajs[0][0, 0], trajs[0][0, 1], 'o', markersize=8, label='Initial State')
for i in range(4):
    plt.plot(goal_pos[i][0], goal_pos[i][1], '*', markersize=10, color='red', label='Goal' if i == 0 else None)

    plt.plot(trajs[i][:, 0], trajs[i][:, 1], label='Trajectories' if i == 0 else None, color='blue')
    plt.legend()

# plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
# Save the contour plot
plt.savefig(f'figs/Trajectories_Input_Constrained_QP_Safe_Control_{current_time}.png')
plt.show()

# Calculate time array based on the number of data points and timestep
num_points = len(trajs[0][:, 0])  # Assuming trajs has the same length for all elements
time = np.arange(0, num_points * timestep, timestep)

# Create subplot for trajs and action variables
fig, axs = plt.subplots(6, 1, figsize=(8, 10))

# Plot trajs variables
axs[0].plot(time, trajs[0][:, 0], label=r'$q_x$', color='red')
axs[0].plot(time, trajs[0][:, 1], label=r'$q_y$', color='blue')
axs[0].set_ylabel(r'$q_x, q_y$', fontsize=14)
axs[0].legend(fontsize=14)
axs[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

axs[1].plot(time, trajs[0][:, 2], label='v', color='black')
axs[1].set_ylabel(r'$v$', fontsize=14)

axs[2].plot(time, trajs[0][:, 3], label='theta', color='black')
axs[2].set_ylabel(r'$\theta$', fontsize=14)

# Plot actions
axs[3].plot(time, actions[0][:, 0], label='u_1', color='black')
axs[3].set_ylabel(r'$u_1$', fontsize=14)

axs[4].plot(time, actions[0][:, 1], label='u_2', color='black')
axs[4].set_ylabel(r'$u_2$', fontsize=14)

# Plot barrier values
axs[5].plot(time, h_vals[0], label='barrier', color='black')
axs[5].set_ylabel(r'$\min h$', fontsize=14)

axs[5].set_xlabel(r'$t~(\rm {s})$', fontsize=14)

for i in range(5):
    axs[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

# Multiply x-axis labels by timestep value (0.001)
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)

# Adjust layout and save the combined plot
plt.tight_layout()
plt.savefig(f'figs/States_Input_Constrained_QP_Safe_Control_{current_time}.png')

# Show the plots
plt.show()
