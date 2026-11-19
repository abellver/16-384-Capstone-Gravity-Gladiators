# Inverse Kinematics by Gradient Descent

import importlib
import numpy as np

_kin = importlib.import_module("6_2_inverse_kinematics")
forward_kinematics = _kin.forward_kinematics
jacobian = _kin.jacobian
rotation_error = _kin.rotation_error

# xArm7 joint limits (rad), from the UFactory xArm7 spec sheet:
JOINT_LIMITS = np.array([
    [-2 * np.pi, 2 * np.pi],   # J1
    [-2.059,     2.0944],      # J2
    [-2 * np.pi, 2 * np.pi],   # J3
    [-0.19198,   3.927],       # J4
    [-2 * np.pi, 2 * np.pi],   # J5
    [-1.69297,   np.pi],       # J6
    [-2 * np.pi, 2 * np.pi],   # J7
])


# mm to m conversion:
POS_SCALE = 1000.0


# Runs FK at q and returns position and rotation error:
def pose_error(T_target, q):
    """6-vector [position error (m); rotation error (rad)] at joints q."""
    T = forward_kinematics(q)
    e_pos = (T_target[:3, 3] - T[:3, 3]) / POS_SCALE
    e_rot = rotation_error(T_target[:3, :3], T[:3, :3])
    return np.concatenate([e_pos, e_rot])



# IK solver:
def inverse_kinematics(T_target, q_init, max_iters=10000, step=0.5,
                       pos_tol=0.1, rot_tol=1e-3):

    q = np.array(q_init, dtype=float).reshape(7)
    lo, hi = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    q = np.clip(q, lo, hi)

    e = pose_error(T_target, q)
    cost = 0.5 * e @ e

    for it in range(max_iters):
        pos_err_mm = np.linalg.norm(e[:3]) * POS_SCALE
        rot_err = np.linalg.norm(e[3:])

        # checks if the position and rotation error are within tolerance
        if pos_err_mm < pos_tol and rot_err < rot_tol:
            return q, True, it

        J = jacobian(q).copy()
        J[:3, :] /= POS_SCALE           # mm/rad -> m/rad
        grad = -J.T @ e                 # gradient of the cost

        # Takes a step; if it makes things worse, it halves it until it's better
        a = step
        while True:
            q_new = np.clip(q - a * grad, lo, hi)
            e_new = pose_error(T_target, q_new)
            cost_new = 0.5 * e_new @ e_new
            
            # checks if the step reduced the error
            if cost_new < cost or a < 1e-6:
                break
            a *= 0.5

        q, e, cost = q_new, e_new, cost_new
        # Let the step grow back after a successful move.
        step = min(step * 1.2, 2.0) if a == step else a

    pos_err_mm = np.linalg.norm(e[:3]) * POS_SCALE
    rot_err = np.linalg.norm(e[3:])
    return q, (pos_err_mm < pos_tol and rot_err < rot_tol), max_iters



# Checking to make sure the IK is correct:
# Run FK on the IK result and compare POSES, not joints
def check_solution(T_target, q, pos_tol=1.0, rot_tol=0.01):
    T = forward_kinematics(q)
    pos_err = np.linalg.norm(T[:3, 3] - T_target[:3, 3])
    rot_err = np.linalg.norm(rotation_error(T_target[:3, :3], T[:3, :3]))
    return pos_err < pos_tol and rot_err < rot_tol, pos_err, rot_err



# Sanity checks before sending q to the real arm.
# - every joint within limits
# - no joint has to swing more than max_joint_jump from where the arm is now
# - every joint frame and the gripper stay above min_z (mm, base frame).
def is_safe(q, q_current, max_joint_jump=np.pi, min_z=-30.0):
    q = np.asarray(q, dtype=float)
    problems = []

    if np.any(q < JOINT_LIMITS[:, 0]) or np.any(q > JOINT_LIMITS[:, 1]):
        problems.append("joint limit violated")

    jump = np.abs(q - np.asarray(q_current, dtype=float))
    if np.max(jump) > max_joint_jump:
        problems.append(f"joint {np.argmax(jump) + 1} would move {np.max(jump):.2f} rad")

    for i, T in enumerate(_kin.joint_transforms(q)):
        if i >= 3 and T[2, 3] < min_z:   # frames 1-3 sit at the base anyway
            problems.append(f"frame {i + 1} below z={min_z} mm (z={T[2, 3]:.1f})")

    return len(problems) == 0, problems



def make_pose(position_m, R):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(position_m, dtype=float) * 1000.0   # -> mm
    return T