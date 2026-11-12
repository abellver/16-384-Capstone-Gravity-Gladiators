import numpy as np

# Builds modified Craig Denavit-Hartenberg transform:
def cdh_transform(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    return np.array([[ct, -st, 0.0, a],
                     [(st * ca), (ct * ca), -sa, (-d * sa)],
                     [(st * sa), (ct * sa), ca, (d * ca)],
                     [0.0, 0.0, 0.0, 1.0]])


# Computes ee pose in the base frame (in mm)
def forward_kinematics(q):
    q = np.asarray(q, dtype=float).reshape(7)

    arm_parameters = [(0.0, 0.0, 267.0, 0.0),
                      (0.0, (-np.pi / 2), 0.0,  0.0),
                      (0.0, (np.pi / 2), 293.0, 0.0),
                      (52.5, (np.pi / 2), 0.0, 0.0),
                      (77.5, (np.pi / 2), 342.5, 0.0),
                      (0.0, (np.pi / 2), 0.0, 0.0),
                      (76.0, (-np.pi / 2), 269.0, 0.0)]

    T = np.eye(4)

    for i in range(7):
        a, alpha, d, theta_offset = arm_parameters[i]
        T = T @ cdh_transform(a, alpha, d, q[i] + theta_offset)

    return T


# Computes the jacobian (in mm/rad)
def jacobian(q):
    q = np.asarray(q, dtype=float).reshape(7)

    arm_parameters = [(0.0, 0.0, 267.0, 0.0),
                      (0.0, (-np.pi / 2), 0.0,  0.0),
                      (0.0, (np.pi / 2), 293.0, 0.0),
                      (52.5, (np.pi / 2), 0.0, 0.0),
                      (77.5, (np.pi / 2), 342.5, 0.0),
                      (0.0, (np.pi / 2), 0.0, 0.0),
                      (76.0, (-np.pi / 2), 269.0, 0.0)]

    transforms = []
    
    T = np.eye(4)
    
    for i in range(7):
        a, alpha, d, theta_offset = arm_parameters[i]
        T = T @ cdh_transform(a, alpha, d, q[i] + theta_offset)
        transforms.append(T)

    position_ee = transforms[-1][:3, 3]

    J = np.zeros((3, 7))

    for i in range(7):
        z_i = transforms[i][:3, 2]
        p_i = transforms[i][:3, 3]
        J[:, i] = np.cross(z_i, position_ee - p_i)

    return J