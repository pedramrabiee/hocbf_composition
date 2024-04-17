from attrdict import AttrDict as AD
import torch
import numpy as np
from torchdiffeq import odeint
from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
from hocbf_composition.examples.unicycle_dynamics import UnicycleDynamics
from hocbf_composition.make_map import Map
from time import time

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'Times'


# settings
k1, k2, k3 = 0.2, 1.0, 2.0
gamma = 1e24
alpha = 0.5

cfg = AD(softmax_rho=20,
         softmin_rho=20,
         pos_barrier_rel_deg=2,
         vel_barrier_rel_deg=1,
         obstacle_alpha=[7.0, 2.5],
         boundary_alpha=[7.0, 1.0],
         velocity_alpha=[10.0],
         )

# Make map configuration
barriers_info = dict(
    geoms=(
        ('box', AD(center=[2.0, 1.5], size=[2.0, 2.0], rotation=0.0)),
        ('box', AD(center=[-2.5, 2.5], size=[1.25, 1.25], rotation=0.0)),
        ('box', AD(center=[-5.0, -5.0], size=[1.875, 1.875], rotation=0.0)),
        ('box', AD(center=[5.0, -6.0], size=[3.0, 3.0], rotation=0.0)),
        ('box', AD(center=[-7.0, 5.0], size=[2.0, 2.0], rotation=0.0)),
        ('box', AD(center=[6.0, 7.0], size=[2.0, 2.0], rotation=0.0)),
        ('boundary', AD(center=[0.0, 0.0], size=[10.0, 10.0], rotation=0.0)),
    ),
    velocity=(
        (2, [-1.0, 9.0]),
    )
)

# Instantiate dynamics
dynamics = UnicycleDynamics(state_dim=4, action_dim=2)

# Make barrier from map
map = Map(barriers_info=barriers_info, dynamics=dynamics, cfg=cfg)


def u_des(x, goal_loc):
    dist_to_goal = np.linalg.norm(x[:2] - goal_loc[:2])
    x, y, v, theta = x[0], x[1], x[2], x[3]
    psi = np.arctan2(y - goal_loc[1], x - goal_loc[0]) - theta + np.pi
    ud1 = (-(k1 + k3) * v + (1 + k1 * k3) * dist_to_goal * np.cos(psi) +
           k1 * (k2 * dist_to_goal + v) * np.power(np.sin(psi), 2))
    ud2 = (k2 + v / dist_to_goal) * np.sin(psi)
    return np.array([ud1, ud2])


def u_safe(x, u_des):
    x = torch.from_numpy(x)
    u_des = torch.from_numpy(u_des).unsqueeze(0)
    Lgh = map.barrier.Lg_hocbf(x)
    hocbf = map.barrier.hocbf(x)
    omega = map.barrier.Lf_hocbf(x) + Lgh @ u_des.t() + alpha * hocbf
    den = 0.5 * Lgh @ Lgh.t() + (1 / gamma) * hocbf ** 2
    lam = (torch.relu(-omega) / den).squeeze().item()
    u = u_des + 0.5 * Lgh * lam
    return u.squeeze()


goal_pos = [
    np.array([3.0, 4.5]),
    np.array([-7.0, 0.0]),
    np.array([7.0, 1.5]),
    np.array([-1.0, 7.0])]

thata_0 = np.pi / 2
x0 = np.array([-1.0, -8.5, 0.0, thata_0])
state = [[x0] for _ in range(4)]
actions = [[] for _ in range(4)]
timestep = 0.001
for i in range(4):
    print(i)
    start_time = time()
    for _ in range(10):
        u_d = u_des(state[i][-1], goal_pos[i])
        action = u_safe(state[i][-1], u_d)
        next_state = odeint(func=lambda t, y: partial(dynamics.rhs,
                                                      action=action)(y),
                            y0=torch.from_numpy(state[i][-1]),
                            t=torch.tensor([0.0, timestep]), method='rk4')[-1].detach().numpy()
        state[i].append(next_state)
        actions[i].append(action.numpy())

        if np.linalg.norm(next_state[:2] - goal_pos[i]) < 0.1:
            break
    print(time() - start_time)



state = [np.row_stack(s) for s in state]
actions = [np.row_stack(ac) for ac in actions]
h_vals = [map.barrier.hocbf(torch.from_numpy(s)) for s in state]

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

x = np.linspace(-10.5, 10.5, 500)
y = np.linspace(-10.5, 10.5, 500)
X, Y = np.meshgrid(x, y, )
points = np.column_stack((X.flatten(), Y.flatten()))
points = np.column_stack((points, np.zeros(points.shape)))
points = torch.tensor(points, dtype=torch.float32)
# print(barriers[0].barrier(points))
Z = map.barrier.min_barrier(points)
Z = Z.reshape(X.shape)
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=[0], colors='k')
plt.xlabel(r'$q_x$')
plt.ylabel(r'$q_y$')

plt.plot(state[0][0, 0], state[0][0, 1], 'o', markersize=8, label='Initial State')
for i in range(4):
    plt.plot(goal_pos[i][0], goal_pos[i][1], '*', markersize=10, color='red', label='Goal' if i == 0 else None)

    plt.plot(state[i][:, 0], state[i][:, 1], label='Trajectories' if i == 0 else None, color='blue')
    plt.legend()

# plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
# Save the contour plot
plt.savefig(f'contour_plot_{current_time}.png')
plt.show()



# Calculate time array based on the number of data points and timestep
num_points = len(state[0][:-1, 0])  # Assuming state has the same length for all elements
time = np.arange(0, num_points * timestep, timestep)

# Create subplot for state and action variables
fig, axs = plt.subplots(6, 1, figsize=(8, 10))

# Plot state variables
axs[0].plot(time, state[0][:-1, 0], label=r'$q_x$', color='red')
axs[0].plot(time, state[0][:-1, 1], label=r'$q_y$', color='blue')
axs[0].set_ylabel(r'$q_x, q_y$', fontsize=14)
axs[0].legend(fontsize=14)
axs[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)


axs[1].plot(time, state[0][:-1, 2], label='v', color='black')
axs[1].set_ylabel(r'$v$', fontsize=14)


axs[2].plot(time, state[0][:-1, 3], label='theta', color='black')
axs[2].set_ylabel(r'$\theta$', fontsize=14)

# Plot actions
axs[3].plot(time, actions[0][:, 0], label='u_1', color='black')
axs[3].set_ylabel(r'$u_1$', fontsize=14)

axs[4].plot(time, actions[0][:, 1], label='u_2', color='black')
axs[4].set_ylabel(r'$u_2$', fontsize=14)

# Plot barrier values
axs[5].plot(time, h_vals[0][:-1, :], label='barrier', color='black')
axs[5].set_ylabel(r'$h$', fontsize=14)

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
plt.savefig(f'combined_plot_{current_time}.png')

# Show the plots
plt.show()
