import time
import numpy as np

ARM_PARAMETERS = [(0.0, 0.0, 267.0, 0.0),
                  (0.0, (-np.pi / 2), 0.0, 0.0),
                  (0.0, (np.pi / 2), 293.0, 0.0),
                  (52.5, (np.pi / 2), 0.0, 0.0),
                  (77.5, (np.pi / 2), 342.5, 0.0),
                  (0.0, (np.pi / 2), 0.0, 0.0),
                  (76.0, (-np.pi / 2), 269.0, 0.0)]


# 6.1 FK
def cdh_transform(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    return np.array([[ct, -st, 0.0, a],
                     [(st * ca), (ct * ca), -sa, (-d * sa)],
                     [(st * sa), (ct * sa), ca, (d * ca)],
                     [0.0, 0.0, 0.0, 1.0]])


# Transforms base -> frame i 
def joint_transforms(q):
    q = np.asarray(q, dtype=float).reshape(7)

    transforms = []
    T = np.eye(4)

    for i in range(7):
        a, alpha, d, theta_offset = ARM_PARAMETERS[i]
        T = T @ cdh_transform(a, alpha, d, q[i] + theta_offset)
        transforms.append(T)

    return transforms


def forward_kinematics(q):
    return joint_transforms(q)[-1]


# Computes the 6x7 geometric jacobian
def jacobian(q):
    transforms = joint_transforms(q)
    position_ee = transforms[-1][:3, 3]

    J = np.zeros((6, 7))

    for i in range(7):
        z_i = transforms[i][:3, 2]
        p_i = transforms[i][:3, 3]
        J[:3, i] = np.cross(z_i, position_ee - p_i)   # linear velocity part
        J[3:, i] = z_i                                 # angular velocity part

    return J


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def rotation_error(R_des, R_cur):
    R = R_des @ R_cur.T
    cos_th = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(cos_th)

    if th < 1e-9:
        return np.zeros(3)

    if np.pi - th < 1e-6:   
        axis = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        k = np.argmax(axis)
        for j in ((k + 1) % 3, (k + 2) % 3):
            s = np.sign(R[k, j] + R[j, k])
            axis[j] *= s if s != 0 else 1
        return th * axis / np.linalg.norm(axis)

    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th * w / (2 * np.sin(th))


#6.2 IK
def move_ee(robot, dp=(0.0, 0.0, 0.0), dyaw=0.0,
            lin_speed=20.0,       # mm/s
            ang_speed=0.25,       # rad/s
            rate=100.0,           # Hz- servo_joints rate
            pos_tol=0.05,         # mm
            ang_tol=0.0005,       # rad
            max_joint_step=0.01,  # rad per tick
            timeout=30.0):
    dt = 1.0 / rate
    q = np.array(robot.joint_values, dtype=float)

    T_start = forward_kinematics(q)
    position_goal = T_start[:3, 3] + np.asarray(dp, dtype=float)
    R_goal = rot_z(dyaw) @ T_start[:3, :3]    

    t_start = time.perf_counter()
    next_tick = t_start

    while True:
        T = forward_kinematics(q)
        error_position = position_goal - T[:3, 3]
        error_rotation = rotation_error(R_goal, T[:3, :3])

        if np.linalg.norm(error_position) < pos_tol and np.linalg.norm(error_rotation) < ang_tol:
            break
        if time.perf_counter() - t_start > timeout:
            print("move_ee: timeout, stopping")
            break

        v = error_position / dt
        if np.linalg.norm(v) > lin_speed:
            v *= lin_speed / np.linalg.norm(v)

        w = error_rotation / dt
        if np.linalg.norm(w) > ang_speed:
            w *= ang_speed / np.linalg.norm(w)

        x_dot = np.concatenate([v, w])

        q_dot = np.linalg.pinv(jacobian(q), rcond=1e-4) @ x_dot

        # Integrate one small step
        dq = q_dot * dt
        biggest = np.max(np.abs(dq))
        if biggest > max_joint_step:
            dq *= max_joint_step / biggest
        q = q + dq

        robot.servo_joints(q)

        next_tick += dt
        sleep = next_tick - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_tick = time.perf_counter()

    time.sleep(0.3)   
    return q


def report(label, q_before, q_after):
    T0, T1 = forward_kinematics(q_before), forward_kinematics(q_after)
    dp = T1[:3, 3] - T0[:3, 3]
    R_rel = T1[:3, :3] @ T0[:3, :3].T
    yaw = np.arctan2(R_rel[1, 0], R_rel[0, 0])
    total_rotation = np.linalg.norm(rotation_error(T1[:3, :3], T0[:3, :3]))

    print(f"\n[{label}]")
    print(f"  start position (mm): {np.round(T0[:3, 3], 2)}")
    print(f"  end position   (mm): {np.round(T1[:3, 3], 2)}")
    print(f"  delta position (mm): {np.round(dp, 2)}")
    print(f"  rotation about z: {yaw:.4f} rad   total rotation: {total_rotation:.4f} rad")