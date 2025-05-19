# An experimental model of QSHNN with four neurons
# periodic projection is applied to guarantee the quaternionic structure of weights matrix blocks

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Quaternion left‑multiplication 4×4 basis and projection helpers
L1 = np.eye(4)
Li = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
Lj = np.array([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]])
Lk = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]])
_basis = np.stack([L1, Li, Lj, Lk])  # (4,4,4)

def _project_block(M: np.ndarray) -> np.ndarray:
    """Frobenius‑least‑squares projection of a 4×4 real matrix onto the quaternion‑left‑multiply subspace."""
    coeff = (_basis * M).sum((1, 2)) / 4.0
    quat_blk = (coeff[:, None, None] * _basis).sum(0)
    return quat_blk

def project_quaternion_blocks(W: np.ndarray, n_blocks: int) -> np.ndarray:
    """Project every 4×4 block of W (shape 4n×4n) to nearest quaternion block."""
    P = W.copy()
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            r0 = 4 * bi
            c0 = 4 * bj
            P[r0:r0 + 4, c0:c0 + 4] = _project_block(P[r0:r0 + 4, c0:c0 + 4])
    # global stability scaling: ensure each row l1‑norm less than 1
    # max_row_sum = np.abs(P).sum(axis=1).max()
    # if max_row_sum > 1.0:
    #     P = P / (max_row_sum + 1e-12)
    return P

# --- Network Parameters ---
n = 4       # number of neurons
mu = 1.0        # network constant
gamma = 1.0     # network constant
eta = 0.03          # initial learning rate
phi = np.tanh   #activation function
def phi_prime(x):
    return 1 - np.tanh(x) ** 2      # activation function on vector variables

# --- adaptive learning‑rate hyper‑parameters ---
lr_min = 0.01         # lower bound
lr_max = 0.05         # upper bound
decay  = 0.99          # factor when loss rebounds
grow   = 1.50         # factor after stable descent
stable_steps = 10      # consecutive descent steps to trigger grow
lr = eta  # current learning rate

# --- preliminaries of training ---
q0 = np.random.uniform(-0.8, 0.8, 4 * n)
b = np.ones(4 * n)      # neuron bias definition

#b = np.random.randn(4 * n) * 0.1
#d = np.repeat(np.random.uniform(-0.9, 0.9, size=n), 4)
d = np.random.uniform(-0.8, 0.8, 4 * n)     # training desire setup
W = np.random.randn(4 * n, 4 * n) * 0.1     # random initial weights setup

# ensure the initial weight matrix satisfies normalization condition
init_max_row = np.abs(W).sum(axis=1).max()
if init_max_row > 1.0:
    W = W / (init_max_row + 1e-12)

# ensure the initial weight matrix satisfies normalization condition
init_max_row = np.abs(W).sum(axis=1).max()
if init_max_row > 1.0:
    W = W / (init_max_row + 1e-12)

max_iter = 30000        # limitation of training iterations
mse_log = []        # record mean square error of training
accuracy_log = []       # record accuracy of training
output_trajectory = []      # record network evolution trajectories

# --- main loop ---
for iter in range(max_iter):

    # evolution equation update
    def dqdt(t, q):
        return -q + mu * W @ phi(q) + mu * b

    # numerically solve the evolution equation by Runge-Kutta method
    sol = solve_ivp(dqdt, [0, 15], q0, method='RK45', rtol=1e-6, atol=1e-9)
    q = sol.y[:, -1]

    # --- preliminaries for weights update ---
    phi_q = phi(q)
    delta = phi_q - d
    J_phi = np.diag(phi_prime(q))
    A = np.eye(4 * n) - (mu / gamma) * W @ J_phi
    A_inv = np.linalg.inv(A)
    grad_common = (delta * phi_prime(q)) @ A_inv

    # --- implement gradient descent ---
    for i in range(4 * n):
        for j in range(4 * n):
            W[i, j] -= lr * grad_common[i] * phi_q[j]

    # Project weights back onto quaternion subspace every 10 iterations
    if (iter + 1) % 10 == 0:
        W = project_quaternion_blocks(W, n)

    # --- compute and record evaluations ---
    output_trajectory.append(phi_q.copy())
    mse = np.mean((phi_q - d) ** 2)
    accuracy = 1 - np.mean(np.abs(phi_q - d) > 0.01)
    mse_log.append(mse)
    accuracy_log.append(accuracy)

    # use this iteration's endpoint as next iteration's initial state, improve stability of training
    q0 = q

    # --- adaptive learning‑rate update ---
    if iter > 0:
        if mse > mse_log[-2]:             # loss increased -> decay learning rate
            lr = max(lr * decay, lr_min)
            down_streak = 0
        else:                             # loss decreased for a number of periods -> increase learning rate
            down_streak += 1
            if down_streak >= stable_steps:
                lr = min(lr * grow, lr_max)
                down_streak = 0
    else:
        down_streak = 0

    # show epoch information on the terminator
    print(f"Iter {iter + 1:3d} | MSE {mse:.6f} | Acc {accuracy:.3f} | lr {lr:.4f}")

    # stop criterion
    if mse < 1e-6 and (iter + 1) % 10 == 0:
        break


# --- Evaluation block: deterministic dynamics with learned weights ---
def dqdt_eval(t, q):
    return -q + mu * W @ phi(q) + mu * b

# random initial condition for evolution plot
q_eval0 = np.random.uniform(-1.0, 1.0, 4 * n)

# solve evolution equation
dyn_sol = solve_ivp(
    dqdt_eval,
    [0, 8],
    q_eval0,
    method='RK45',
    rtol=1e-6,
    atol=1e-9
)

# --- Color sets: same scheme as training‑trajectory plot ---
color_sets = [
    [(0.6, 0.0, 0.0), (0.7, 0.2, 0.2), (0.8, 0.4, 0.4), (0.9, 0.7, 0.7)],
    [(0.0, 0.0, 0.5), (0.2, 0.2, 0.7), (0.4, 0.4, 0.8), (0.7, 0.7, 0.9)],
    [(0.0, 0.5, 0.0), (0.2, 0.7, 0.2), (0.4, 0.8, 0.4), (0.7, 0.9, 0.7)],
    [(0.5, 0.3, 0.0), (0.7, 0.5, 0.2), (0.85, 0.7, 0.4), (0.95, 0.85, 0.6)]
]

# Plot the evolution trajectories
phi_dyn = phi(dyn_sol.y.T)    # compute activation along trajectory

plt.figure(figsize=(8, 5))

for q_index in range(n):          # n = 4
    for i in range(4):
        idx = q_index * 4 + i
        c = color_sets[q_index][i]
        label = f'q{q_index + 1}[{i}]'
        plt.plot(dyn_sol.t, phi_dyn[:, idx], color=c, linewidth=1.2,label=label)
        plt.axhline(d[idx], color=c, linestyle='--', linewidth=0.8)

# --- plot trajectories from random initial state ---
plt.xlabel('Time')
plt.ylabel('phi(q_i(t))')
plt.title('System Evolution with Final Weights (targets shown)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot training loss and accuracy ---
mse_log = np.array(mse_log)
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

# --- Plot Weight matrix heatmap ---
plt.figure(figsize=(6, 5))
plt.imshow(np.abs(W), cmap='Blues', aspect='auto')
plt.colorbar(label='|Weight|')
plt.title('Final Weight Matrix Heatmap')
plt.xlabel('From neuron j')
plt.ylabel('To neuron i')
plt.tight_layout()
plt.show()

# --- Plot training output convergence of each phi(q_i) ---
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