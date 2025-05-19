# An experimental model of SHNN with four neurons
# Strict gradient descent is applied, quaternionic structure is not preserved.

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Network Parameters ---
n = 4       #number of neurons
mu = 1.0        # netowork constant
gamma = 1.0     # netowrk constant
eta = 0.03      # learning rate
phi = np.tanh   # activation function
def phi_prime(x):
    return 1 - np.tanh(x) ** 2

# --- preliminaries for training ---
q0 = np.random.randn(4 * n) * 0.5       # initial state
b = np.random.randn(4 * n) * 0.1        # bias
d = np.repeat(np.random.uniform(-0.9, 0.9, size=n), 4)      # desire states
W = np.random.randn(4 * n, 4 * n) * 0.1     # initial weights

# ensure the initial weight matrix satisfies normalization condition
init_max_row = np.abs(W).sum(axis=1).max()
if init_max_row > 1.0:
    W = W / (init_max_row + 1e-12)

# ---training hyper-parameters ---
max_iter = 20000        # limitation on epoch numbers
mse_log = []        # record for mean square error
accuracy_log = []       # record for accuracy
output_trajectory = []      # record for evolution trajectories

# --- main loop ---
for iter in range(max_iter):

    # define evolution equation
    def dqdt(t, q):
        return -q + mu * W @ phi(q) + mu * b

    # solve the evolution numerically by Runge-Kutta method
    sol = solve_ivp(dqdt, [0, 10], q0, method='RK45', rtol=1e-6, atol=1e-9)
    q = sol.y[:, -1]

    # preliminaries for weights update
    phi_q = phi(q)
    delta = phi_q - d
    J_phi = np.diag(phi_prime(q))
    A = np.eye(4 * n) - (mu / gamma) * W @ J_phi
    A_inv = np.linalg.inv(A)
    grad_common = (delta * phi_prime(q)) @ A_inv

    # weights update
    for i in range(4 * n):
        for j in range(4 * n):
            W[i, j] -= eta * grad_common[i] * phi_q[j]

    # record epoch information
    output_trajectory.append(phi_q.copy())
    mse = np.mean((phi_q - d) ** 2)
    accuracy = 1 - np.mean(np.abs(phi_q - d) > 0.01)
    mse_log.append(mse)
    accuracy_log.append(accuracy)
    print(f"Iteration {iter + 1}, MSE = {mse:.6f}, Accuracy = {accuracy:.3f}")

    # stop criterion
    if mse < 1e-6:
        break

    # use this iteration's endpoint as next iteration's initial state, improve stability of training
    q0 = q

# Plot training loss and accuracy
plt.figure(figsize=(8, 5))
plt.plot(mse_log, label='MSE Loss', linewidth=1.5)
plt.plot(accuracy_log, label='Accuracy', linewidth=1.5)
plt.xlabel('Iteration')
plt.ylabel('Loss / Accuracy')
plt.title('Training Loss and Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Weight matrix heatmap
plt.figure(figsize=(6, 5))
plt.imshow(W, cmap='Blues', aspect='auto')
plt.colorbar(label='Weight Value')
plt.title('Final Weight Matrix Heatmap')
plt.xlabel('From neuron j')
plt.ylabel('To neuron i')
plt.tight_layout()
plt.show()

# Plot training output convergence of each phi(q_i)
output_trajectory = np.array(output_trajectory)
color_sets = [
    [(0.6, 0.0, 0.0), (0.7, 0.2, 0.2), (0.8, 0.4, 0.4), (0.9, 0.7, 0.7)],
    [(0.0, 0.0, 0.5), (0.2, 0.2, 0.7), (0.4, 0.4, 0.8), (0.7, 0.7, 0.9)],
    [(0.0, 0.5, 0.0), (0.2, 0.7, 0.2), (0.4, 0.8, 0.4), (0.7, 0.9, 0.7)],
    [(0.5, 0.3, 0.0), (0.7, 0.5, 0.2), (0.85, 0.7, 0.4), (0.95, 0.85, 0.6)]
]

plt.figure(figsize=(10, 6))
for q_index in range(4):
    for i in range(4):
        idx = q_index * 4 + i
        label = f'q{q_index + 1}[{i}]'
        plt.plot(output_trajectory[:, idx], label=label, color=color_sets[q_index][i], linewidth=1.8)
        plt.axhline(d[idx], color=color_sets[q_index][i], linestyle='--', linewidth=0.8)

plt.xlabel('Training Step')
plt.ylabel('phi(q)')
plt.title('Neuron Output Convergence over Training')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

# Final evolution and phase plane
q0_test = np.random.randn(4 * n) * 2.0
sol_final = solve_ivp(lambda t, q: -q + mu * W @ phi(q) + mu * b,
                      [0, 20], q0_test, method='RK45', rtol=1e-6, atol=1e-9, dense_output=True)
t_vals = np.linspace(0, 20, 300)
q_traj = sol_final.sol(t_vals)
phi_traj = phi(q_traj)

plt.figure(figsize=(8, 5))
for i in range(4):
    plt.plot(t_vals, phi_traj[i, :], label=f'q1[{i}]', color='tab:blue', linewidth=1.8)
    plt.axhline(d[i], color='tab:blue', linestyle='--', linewidth=0.8)

plt.xlabel('Time')
plt.ylabel('phi(q1)')
plt.title('Dynamics of First Quaternion Over Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(phi_traj[0], phi_traj[1], label='q1[0] vs q1[1]', color='tab:red')
# plt.xlabel('phi(q1[0])')
# plt.ylabel('phi(q1[1])')
# plt.title('Phase Plane: q1[0] vs q1[1]')
# plt.grid(True)
#
# plt.subplot(1, 2, 2)
# plt.plot(phi_traj[4], phi_traj[5], label='q2[0] vs q2[1]', color='tab:blue')
# plt.xlabel('phi(q2[0])')
# plt.ylabel('phi(q2[1])')
# plt.title('Phase Plane: q2[0] vs q2[1]')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Phase portrait of final dynamics in q[0] vs q[1]
# plt.figure(figsize=(6, 6))
# plt.plot(q_traj[0], q_traj[1], color='darkorange', linewidth=2)
# plt.xlabel('q[0]')
# plt.ylabel('q[1]')
# plt.title('Phase Portrait: q[0] vs q[1]')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

#Phase Portrait (Vector Field) in selected 2D planes
q_range = np.linspace(-2, 2, 20)
Q0, Q1 = np.meshgrid(q_range, q_range)


#Helper to plot vector field for given index pair (i, j)
def plot_phase_plane(i, j):
    U = np.zeros_like(Q0)
    V = np.zeros_like(Q1)
    M = np.zeros_like(Q0)
    for r in range(Q0.shape[0]):
        for c in range(Q0.shape[1]):
            q_sample = np.zeros(4 * n)
            q_sample[i] = Q0[r, c]
            q_sample[j] = Q1[r, c]
            dq = -q_sample + mu * W @ phi(q_sample) + mu * b
            U[r, c] = dq[i]
            V[r, c] = dq[j]
            M[r, c] = np.sqrt(dq[i] ** 2 + dq[j] ** 2)

    norm = np.sqrt(U ** 2 + V ** 2)
    U_fixed = U / (norm + 1e-8)
    V_fixed = V / (norm + 1e-8)

    plt.figure(figsize=(5, 5))
    plt.quiver(Q0, Q1, U_fixed, V_fixed, M, angles='xy', cmap='viridis', scale_units='xy', scale=6, width=0.006)

    plt.xlabel(f'q[{i}]')
    plt.ylabel(f'q[{j}]')
    plt.title(f'Phase Portrait: q[{i}] vs q[{j}]')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# Plot for multiple pairs
plot_phase_plane(0, 1)
plot_phase_plane(2, 3)
#plot_phase_plane(4, 5)
#plot_phase_plane(6, 7)

# # 2D Trajectory plots from final dynamics
# plt.figure(figsize=(6, 6))
# plt.plot(phi_traj[0], phi_traj[1], color='tab:blue', linewidth=1.8)
# plt.xlabel('phi(q1[0])')
# plt.ylabel('phi(q1[1])')
# plt.title('Trajectory in q1[0] vs q1[1] Plane')
# plt.grid(True)
# plt.axis('equal')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(6, 6))
# plt.plot(phi_traj[2], phi_traj[3], color='tab:green', linewidth=1.8)
# plt.xlabel('phi(q1[2])')
# plt.ylabel('phi(q1[3])')
# plt.title('Trajectory in q1[2] vs q1[3] Plane')
# plt.grid(True)
# plt.axis('equal')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(6, 6))
# plt.plot(phi_traj[4], phi_traj[5], color='tab:purple', linewidth=1.8)
# plt.xlabel('phi(q2[0])')
# plt.ylabel('phi(q2[1])')
# plt.title('Trajectory in q2[0] vs q2[1] Plane')
# plt.grid(True)
# plt.axis('equal')
# plt.tight_layout()
# plt.show()
