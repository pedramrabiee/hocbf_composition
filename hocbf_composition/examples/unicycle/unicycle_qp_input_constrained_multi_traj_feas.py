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
import pickle


mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# Control gains
control_gains = dict(k1=0.2, k2=1.0, k3=2.0)

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
    alpha=lambda x: 1.0 * x,
    params=AD(slack_gain=200,
              slacked=True,
              use_softplus=False,
              softplus_gain=2.0)
).assign_dynamics(dynamics=dynamics).assign_state_barrier(barrier=barrier
                                                          ).assign_control_bounds(low=[-4.0, -1.0], high=[4.0, 1.0])

# Goal positions
x = np.linspace(-10, 10, 60)
y = np.linspace(-10, 10, 60)
X, Y = np.meshgrid(x, y, )
points = np.column_stack((X.flatten(), Y.flatten()))
points = np.column_stack((points, np.zeros(points.shape)))
points = torch.tensor(points, dtype=torch.float32)
Z = map_.barrier.min_barrier(points)
goals = points[(Z >= 0).squeeze()]
print("num trajs: ", len(goals))
goal_pos = goals[:, :2]

# Initial Conditions
timestep = 0.01
sim_time = 20.0

safety_filter.assign_desired_control(
    desired_control=lambda x: vectorize_tensors(partial(desired_control, goal_pos=goal_pos, **control_gains)(x)))


x0 = torch.tensor([-1.0, -8.5, 0.0, pi / 2]).repeat(goal_pos.shape[0], 1)


# Simulate trajectories
load_trajs = True
if load_trajs:
    with open("qp_ic_multi_trajs_02.pkl", 'rb') as f:
        trajs = pickle.load(f)
else:
    start_time = time()
    trajs = safety_filter.get_safe_optimal_trajs(x0=x0, sim_time=sim_time, timestep=timestep, method='euler')
    print(time() - start_time)

    # Rearrange trajs
    trajs = [torch.vstack(t.split(dynamics.state_dim)) for t in torch.hstack([tt for tt in trajs])]

    with open("qp_ic_multi_trajs.pkl", 'wb') as f:
        pickle.dump(trajs, f)

load_cutoff = False
if load_cutoff:
    with open("cutoffs.pkl", 'rb') as f:
        cutoffs = pickle.load(f)
else:
    cutoff_threshs = [-1e-4, -1e-2, -1e-1]
    feas_counter = [0] * len(cutoff_threshs)
    cutoffs = np.ones((len(cutoff_threshs), len(trajs)), int) * (len(trajs) - 1)

    for i, traj in enumerate(trajs):
        print("Traj No: ", i)
        des_ctrl = lambda x: vectorize_tensors(
            partial(desired_control, goal_pos=goal_pos[i].repeat(x.shape[0], 1))(x))
        safety_filter.assign_desired_control(
            desired_control=des_ctrl
        )
        _, info = safety_filter.safe_optimal_control(traj, ret_info=True)
        constraint = info['constraint_at_u']

        for j, cutoff_thresh in enumerate(cutoff_threshs):
            cutoff = (torch.min(constraint, dim=-1).values < cutoff_thresh).nonzero(as_tuple=True)[0]

            if cutoff.numel() > 0:
                feas_counter[j] += 1
                cutoffs[j, i] = cutoff[0].item()
            else:
                break

    with open("cutoffs.pkl", 'wb') as f:
        pickle.dump(cutoffs, f)

    print("feasibility:", feas_counter)

############
#  Plots   #
############
cutoff_idx = 2

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

fig, ax = plt.subplots(figsize=(6, 6))

contour_plot = ax.contour(X, Y, Z, levels=[0], colors='red')
# Adding a custom legend handle for the contour
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='red', lw=1.5)]

ax.set_xlabel(r'$q_{\rm x}$', fontsize=16)
ax.set_ylabel(r'$q_{\rm y}$', fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_aspect('equal', adjustable='box')  # Maintain equal aspect ratio
ax.tick_params(axis='x', labelsize=16)  # Font size for x-axis ticks
ax.tick_params(axis='y', labelsize=16)  # Font size for y-axis ticks
ax.set_xticks([-10, -5, 0, 5, 10])
ax.set_yticks([-10, -5, 0, 5, 10])


for i in range(len(trajs)):
    cof = cutoffs[cutoff_idx][i]
    ax.plot(trajs[i][:cof, 0], trajs[i][:cof, 1], color='rebeccapurple', alpha=0.3)
# Creating a custom line for 'Trajectories' with alpha 0.8 for the legend
custom_lines.append(Line2D([0], [0], color='rebeccapurple', alpha=0.6, label='Trajectories'))


ax.plot(x0[0, 0], x0[0, 1], 'x', color='blue', markersize=8, label=r'$x_0$')


# Creating the legend
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, custom_lines[0])
labels.insert(0, r'$\mathcal{S}_{\rm s}$')
handles.insert(3, custom_lines[1])  # Add the custom line for Trajectories with higher alpha
labels.insert(3, 'Trajectories (HOCBF-QP)')  # Add the corresponding label


# ax.legend(handles, labels)
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=12)

custom_order = [r'$\mathcal{S}_{\rm s}$', 'Trajectories (HOCBF-QP)', r'$x_0$']
handle_dict = dict(zip(labels, handles))
ordered_handles = [handle_dict[label] for label in custom_order]
ordered_labels = custom_order


plt.tight_layout()

# Save the contour plot
plt.savefig(f'figs/Trajectories_Input_Constrained_QP_Safe_Control_{current_time}.png', dpi=600)
plt.show()
