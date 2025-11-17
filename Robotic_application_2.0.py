# An experimental model of QSHNN with four neurons
# periodic projection is applied to guarantee the quaternionic structure of weights matrix blocks

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Quaternion left‑multiplication 4×4 basis and projection helpers
L1 = np.eye(4)
Li = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
Lj = np.array([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]])
Lk = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]])
basis = np.stack([L1, Li, Lj, Lk])  # (4,4,4)

def _project_block(M: np.ndarray) -> np.ndarray:
    """Frobenius‑least‑squares projection of a 4×4 real matrix onto the quaternion‑left‑multiply subspace."""
    coeff = (basis * M).sum((1, 2)) / 4.0
    quat_blk = (coeff[:, None, None] * basis).sum(0)
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
lr_min = 0.001         # lower bound
lr_max = 0.20         # upper bound
decay  = 0.80          # factor when loss rebounds
grow   = 1.10         # factor after stable descent
stable_steps = 3      # consecutive descent steps to trigger grow
period =10
scale_ratio = 1.1       # scale ratio of activation function
damping_ratio = 1.0     # damping of self feedback connection
lr = eta  # current learning rate

# --- preliminaries of training ---
q0 = np.random.uniform(-1.0, 1.0, 4 * n)
b = np.ones(4 * n)/8     # neuron bias definition
#b = np.random.randn(4 * n) * 0.1
#d = np.repeat(np.random.uniform(-0.9, 0.9, size=n), 4)
d = np.random.uniform(-0.5, 0.5, 4 * n)     # training desire setup
W = np.random.randn(4 * n, 4 * n) * 0.1     # random initial weights setup

# ensure the initial weight matrix satisfies normalization condition
init_max_row = np.abs(W).sum(axis=1).max()
if init_max_row > 1.0:
    W = W / (init_max_row + 1e-12)

max_iter = 10000        # limitation of training iterations
mse_log = []        # record mean square error of training
accuracy_log = []       # record accuracy of training
output_trajectory = []      # record network evolution trajectories
down_streak = 0

# --- main loop ---
for iter in range(max_iter):

    # evolution equation update
    def dqdt(t, q):
        return -damping_ratio * q + scale_ratio * mu * W @ phi(q) + mu * b

    # numerically solve the evolution equation by Runge-Kutta method
    sol = solve_ivp(dqdt, [0, 15], q0, method='RK45', rtol=1e-6, atol=1e-9)
    q = sol.y[:, -1]

    # --- preliminaries for weights update ---
    #phi_q = phi(q)
    phi_q = q
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
    if (iter + 1) % period == 0:
        W = project_quaternion_blocks(W, n)

    # --- compute and record evaluations ---
    output_trajectory.append(phi_q.copy())
    mse = np.sqrt(np.sum((phi_q - d) ** 2))
    loss = np.max(phi_q - d)
    accuracy = 1 - np.mean(np.abs(phi_q - d) > 1e-6)
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

    # Throttle console output: print only the first iteration and then every 100th
    if iter == 0 or (iter + 1) % 100 == 0:
        print(f"Iter {iter + 1:5d} | MSE {mse:.7f} | Acc {accuracy:.3f} | lr {lr:.3f}")

    # stop criterion
    if loss < 5e-7 and (iter + 1) % period and accuracy >=1.0:
        break


# --- Evaluation block: deterministic dynamics with learned weights ---
def dqdt_eval(t, q):
    return -damping_ratio * q + scale_ratio * mu * W @ phi(q) + mu * b

# random initial condition for evolution plot
q_eval0 = d + np.random.uniform(-0.3, 0.3, 4 * n)
#q_eval0 = np.random.uniform(-0.5, 0.5, 4 * n)

# solve evolution equation
# Add dense sampling points for smoother trajectory, but keep integration interval unchanged
t_eval = np.linspace(0, 30, 500)  # 500 points between 0 and 8
dyn_sol = solve_ivp(
    dqdt_eval,
    [0, 30],
    q_eval0,
    method='RK45',
    rtol=1e-6,
    atol=1e-9,
    t_eval=t_eval
)

# --- Color sets: same scheme as training‑trajectory plot ---
color_sets = [
    [(0.6, 0.0, 0.0), (0.7, 0.2, 0.2), (0.8, 0.4, 0.4), (0.9, 0.7, 0.7)],
    [(0.0, 0.0, 0.5), (0.2, 0.2, 0.7), (0.4, 0.4, 0.8), (0.7, 0.7, 0.9)],
    [(0.0, 0.5, 0.0), (0.2, 0.7, 0.2), (0.4, 0.8, 0.4), (0.7, 0.9, 0.7)],
    [(0.5, 0.3, 0.0), (0.7, 0.5, 0.2), (0.85, 0.7, 0.4), (0.95, 0.85, 0.6)]
]

# # Plot the evolution trajectories
# phi_dyn = phi(dyn_sol.y.T)    # compute activation along trajectory
phi_dyn = dyn_sol.y.T           # compute activation along trajectory

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
plt.figure(figsize=(6, 5))
plt.plot(mse_log, label='MSE Loss', linewidth=1.5)
plt.plot(accuracy_log, label='Accuracy', linewidth=1.5)
#plt.ylim([-0.1, 1.3])
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

plt.figure(figsize=(8, 5))
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


# --- Visualize 4-DOF Robotic Arm Joint Trajectories in 3D ---
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def visualize_robot_motion(quat_traj, link_length=0.3, target_quat=None):
    """
    Visualize the 3D trajectory of a 4‑DOF serial robotic arm given a sequence of joint quaternions.
    Also marks the desired (target) pose if provided.
    quat_traj: ndarray of shape (timesteps, 16) (4 quaternions per time step)
    link_length: length of each arm segment
    """
    n_joints = 4
    n_steps = quat_traj.shape[0]
    # Each joint quaternion: (w, x, y, z) in vector order [q0, q1, q2, q3]
    # We'll use scipy's Rotation to convert to rotation matrices
    # Assume base at (0,0,0), each joint rotates its link, serial chain

    # Store end-effector positions for trajectory plotting
    # If a target quaternion is supplied, pre-compute its joint / EE position
    target_joint_positions = None
    target_ee = None
    if target_quat is not None:
        target_quat = np.asarray(target_quat).ravel()
        if target_quat.size == 4 * n_joints:
            jp_t = [np.zeros(3)]
            T_t = np.eye(3)
            p_t = np.zeros(3)
            for j in range(n_joints):
                q_t = target_quat[j*4:(j+1)*4]
                q_t = q_t / np.linalg.norm(q_t) if np.linalg.norm(q_t) else q_t
                R_t = R.from_quat([q_t[1], q_t[2], q_t[3], q_t[0]]).as_matrix()
                T_t = T_t @ R_t
                p_t = p_t + T_t @ np.array([0, 0, link_length])
                jp_t.append(p_t.copy())
            target_joint_positions = np.array(jp_t)
            target_ee = target_joint_positions[-1]
    ee_positions = []
    # Also store all joint positions for each time step for animation
    all_joint_positions = []

    for t in range(n_steps):
        joint_positions = [np.zeros(3)]  # base at origin
        current_transform = np.eye(3)
        current_pos = np.zeros(3)
        for j in range(n_joints):
            q = quat_traj[t, j*4:(j+1)*4]
            # Normalize quaternion in (w, x, y, z) format
            q_norm = q / np.linalg.norm(q) if np.linalg.norm(q) > 0 else q
            # scipy expects (x, y, z, w)
            r = R.from_quat([q_norm[1], q_norm[2], q_norm[3], q_norm[0]])
            # Compose rotation
            current_transform = current_transform @ r.as_matrix()
            # Next joint position: move along z axis of current frame
            next_pos = current_pos + current_transform @ np.array([0, 0, link_length])
            joint_positions.append(next_pos)
            current_pos = next_pos
        ee_positions.append(joint_positions[-1])
        all_joint_positions.append(joint_positions)

    ee_positions = np.array(ee_positions)

    # Plot 3D trajectory and arm pose at first and last time steps
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    # Plot trajectory of end-effector
    ax.plot(ee_positions[:,0], ee_positions[:,1], ee_positions[:,2], 'k-', label='End-Effector Trajectory', linewidth=2)
    # Draw the base as a cylinder or sphere at the origin (for visual clarity)
    origin = np.zeros(3)
    ax.scatter([origin[0]], [origin[1]], [origin[2]], color='gray', s=100, marker='o', label='Base')
    # Plot the arm at the first and last time steps
    for idx, t in enumerate([0, n_steps-1]):
        joint_positions = np.array(all_joint_positions[t])
        ax.plot(joint_positions[:,0], joint_positions[:,1], joint_positions[:,2],
                marker='o', linewidth=2, label=f'Arm Pose {"Start" if idx==0 else "End"}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('4-DOF Robotic Arm Motion (from QSHNN Quaternions)')
    if target_ee is not None:
        # Mark desired end-effector with a red star
        ax.scatter([target_ee[0]], [target_ee[1]], [target_ee[2]],
                   color='red', s=80, marker='*', label='Desired EE')
        # Optionally draw target arm pose with dashed gray
        ax.plot(target_joint_positions[:,0],
                target_joint_positions[:,1],
                target_joint_positions[:,2],
                color='gray', linestyle='--', linewidth=1.2,
                alpha=0.6, label='Desired Pose')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# Video writer: save full 3‑D motion as MP4 (falls back to GIF)
# -----------------------------------------------------------
def save_robot_motion_video(quat_traj, link_length=1.0,
                            filename='robot_motion.mp4', fps=30,
                            target_quat=None):
    """
    Render the 4‑DOF arm motion and save to video.
    If FFmpeg is unavailable, falls back to GIF via PillowWriter.

    Parameters
    ----------
    quat_traj : ndarray, shape (steps, 16)
        Quaternion sequence from dynamical simulation.
    link_length : float
        Length of each arm segment.
    filename : str
        Output filename (.mp4 or .gif will be produced).
    fps : int
        Frames per second.
    """
    from scipy.spatial.transform import Rotation as R

    n_joints = 4
    # ----- desired pose & EE (if provided) -----
    target_joint_pos = None
    target_ee = None
    if target_quat is not None:
        target_quat = np.asarray(target_quat).ravel()
        if target_quat.size == 4 * n_joints:
            jp_t = [np.zeros(3)]
            T_t = np.eye(3)
            p_t = np.zeros(3)
            for j in range(n_joints):
                q_t = target_quat[j*4:(j+1)*4]
                q_t = q_t / np.linalg.norm(q_t) if np.linalg.norm(q_t) else q_t
                R_t = R.from_quat([q_t[1], q_t[2], q_t[3], q_t[0]]).as_matrix()
                T_t = T_t @ R_t
                p_t = p_t + T_t @ np.array([0, 0, link_length])
                jp_t.append(p_t.copy())
            target_joint_pos = np.array(jp_t)
            target_ee = target_joint_pos[-1]
    # ---- pre‑compute joint positions for every frame ----
    all_joint_pos = []
    for quat in quat_traj:
        jp = [np.zeros(3)]
        T = np.eye(3)
        p = np.zeros(3)
        for j in range(n_joints):
            q = quat[j*4:(j+1)*4]
            q = q / np.linalg.norm(q) if np.linalg.norm(q) else q
            # scipy expects (x,y,z,w)
            Rj = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            T = T @ Rj
            p = p + T @ np.array([0, 0, link_length])
            jp.append(p.copy())
        all_joint_pos.append(np.array(jp))
    all_joint_pos = np.array(all_joint_pos)

    # ---- create figure ----
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    lim = n_joints * link_length * 0.8
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, lim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('4‑DOF Arm Motion')

    line, = ax.plot([], [], [], 'o-', lw=2)
    # Pre‑draw desired pose if available
    if target_joint_pos is not None:
        ax.plot(target_joint_pos[:, 0],
                target_joint_pos[:, 1],
                target_joint_pos[:, 2],
                linestyle='--', color='gray', linewidth=1.2, alpha=0.6,
                label='Desired Pose')
        ax.scatter([target_ee[0]], [target_ee[1]], [target_ee[2]],
                   color='red', s=60, marker='*', label='Desired EE')

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line,

    def update(frame):
        jp = all_joint_pos[frame]
        line.set_data(jp[:, 0], jp[:, 1])
        line.set_3d_properties(jp[:, 2])
        return line,

    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=len(all_joint_pos), interval=1000/fps, blit=False)

    try:
        writer = animation.FFMpegWriter(fps=fps)
        out_path = os.path.abspath(filename)
        anim.save(out_path, writer=writer)
        print(f"Motion video saved to {out_path}")
        return out_path
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        # Fallback to GIF if FFmpeg is not available
        gif_name = filename.rsplit('.', 1)[0] + '.gif'
        print("FFmpeg unavailable or failed (", exc, ") – saving GIF instead…")
        writer = animation.PillowWriter(fps=fps)
        gif_path = os.path.abspath(gif_name)
        anim.save(gif_path, writer=writer)
        print(f"Motion animation saved to {gif_path}")
        return gif_path
    finally:
        plt.close(fig)
# Call the visualization with the simulated quaternion trajectories
visualize_robot_motion(dyn_sol.y.T, target_quat=d)

# Save the same trajectory as a video / GIF for review
video_path = save_robot_motion_video(dyn_sol.y.T, target_quat=d)
print("\n======== Saved motion video ========")
print(video_path)
print("====================================\n")


# === Interactive Plotly Visualisation (keeps previous plots intact) ===
# This block only adds an interactive 3‑D figure; it does not alter prior results.
try:
    import plotly.graph_objs as go
    from scipy.spatial.transform import Rotation as R

    def _compute_joint_positions(q_traj, link_len=0.3):
        """Return (steps, joints+1, 3) joint positions and EE path for 4‑DOF arm."""
        n_j = 4
        all_j, ee = [], []
        for quat in q_traj:
            jp = [np.zeros(3)]
            T, p = np.eye(3), np.zeros(3)
            for j in range(n_j):
                q = quat[j*4:(j+1)*4]
                q = q / np.linalg.norm(q) if np.linalg.norm(q) else q
                Rj = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                T = T @ Rj
                p = p + T @ np.array([0, 0, link_len])
                jp.append(p.copy())
            all_j.append(jp)
            ee.append(jp[-1])
        return np.array(all_j), np.array(ee)

    _all_joint_pos, _ee_path = _compute_joint_positions(dyn_sol.y.T)

    # --- desired pose & EE marker ---
    _target_joint_pos, _ = _compute_joint_positions(np.array([d]))
    _target_joint_pos = _target_joint_pos[0]
    _target_ee = _target_joint_pos[-1]

    fig = go.Figure()

    # End‑effector trajectory
    fig.add_trace(go.Scatter3d(
        x=_ee_path[:, 0], y=_ee_path[:, 1], z=_ee_path[:, 2],
        mode='lines', line=dict(color='black', width=4),
        name='End‑Effector Trajectory'
    ))

    # Start and end poses
    for _idx, _step in enumerate([0, -1]):
        _pts = _all_joint_pos[_step]
        fig.add_trace(go.Scatter3d(
            x=_pts[:, 0], y=_pts[:, 1], z=_pts[:, 2],
            mode='lines+markers',
            marker=dict(size=4),
            line=dict(width=6),
            name='Start Pose' if _idx == 0 else 'End Pose'
        ))

    # Base
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers', marker=dict(size=7, color='gray'),
        name='Base'
    ))

    # Desired end‑effector position
    fig.add_trace(go.Scatter3d(
        x=[_target_ee[0]], y=[_target_ee[1]], z=[_target_ee[2]],
        mode='markers',
        marker=dict(size=7, color='red', symbol='diamond-open'),
        name='Desired EE'
    ))

    # Desired arm pose (dashed gray)
    fig.add_trace(go.Scatter3d(
        x=_target_joint_pos[:, 0],
        y=_target_joint_pos[:, 1],
        z=_target_joint_pos[:, 2],
        mode='lines',
        line=dict(width=3, color='gray', dash='dash'),
        name='Desired Pose'
    ))

    fig.update_layout(
        title='Interactive 4‑DOF Robotic Arm Motion (Drag to Rotate)',
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        legend=dict(x=0.02, y=0.98)
    )
    fig.show()

except ImportError:
    print("Plotly is not installed; skipping interactive visualisation.")