import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Network Parameters
n = 4
mu = 1.0
gamma = 1.0
eta = 0.03

phi = np.tanh


def phi_prime(x):
    return 1 - np.tanh(x) ** 2


q0 = np.random.randn(4 * n) * 2.0
b = np.random.randn(4 * n) * 0.1
d = np.repeat(np.random.uniform(-0.9, 0.9, size=n), 4)
W = np.random.randn(4 * n, 4 * n) * 0.1

# ensure the initial weight matrix satisfies normalization condition
init_max_row = np.abs(W).sum(axis=1).max()
if init_max_row > 1.0:
    W = W / (init_max_row + 1e-12)

max_iter = 2000
mse_log = []
accuracy_log = []
output_trajectory = []

for iter in range(max_iter):
    def dqdt(t, q):
        return -q + mu * W @ phi(q) + mu * b


    sol = solve_ivp(dqdt, [0, 10], q0, method='RK45', rtol=1e-6, atol=1e-9)
    q = sol.y[:, -1]

    phi_q = phi(q)
    delta = phi_q - d
    J_phi = np.diag(phi_prime(q))
    A = np.eye(4 * n) - (mu / gamma) * W @ J_phi
    A_inv = np.linalg.inv(A)
    grad_common = (delta * phi_prime(q)) @ A_inv

    for i in range(4 * n):
        for j in range(4 * n):
            W[i, j] -= eta * grad_common[i] * phi_q[j]

    output_trajectory.append(phi_q.copy())
    mse = np.mean((phi_q - d) ** 2)
    accuracy = 1 - np.mean(np.abs(phi_q - d) > 0.01)
    mse_log.append(mse)
    accuracy_log.append(accuracy)
    print(f"Iteration {iter + 1}, MSE = {mse:.6f}, Accuracy = {accuracy:.3f}")

    if mse < 1e-6:
        break

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

# random initial condition for evolution plot
q_eval0 = np.random.uniform(-1.0, 1.0, 4 * n)

def dqdt_eval(t, q):
    return -q + mu * W @ phi(q) + mu * b

dyn_sol = solve_ivp(
    dqdt_eval,
    [0, 20],
    q_eval0,
    method='RK45',
    rtol=1e-6,
    atol=1e-9
)

phi_dyn = phi(dyn_sol.y.T)    # compute activation along trajectory

plt.figure(figsize=(8, 5))
for q_index in range(n):          # n = 4
    for i in range(4):
        idx = q_index * 4 + i
        c = color_sets[q_index][i]
        label = f'q{q_index + 1}[{i}]'
        plt.plot(dyn_sol.t, phi_dyn[:, idx], color=c, linewidth=1.2,label=label)
        plt.axhline(d[idx], color=c, linestyle='--', linewidth=0.8)

plt.xlabel('Time')
plt.ylabel('phi(q_i(t))')
plt.title('System Evolution with Final Weights (targets shown)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()