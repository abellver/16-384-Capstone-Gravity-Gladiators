# IK test on the robot.

import numpy as np
from xarm7_lib import Robot
from IKGradientDescent import (inverse_kinematics, check_solution, is_safe,
                                make_pose, forward_kinematics)

np.set_printoptions(precision=4, suppress=True)

# Table 4 (positions in metres, converted to mm by make_pose)
POSES = [
    ("Pose 1", (0.5, 0.0, 0.0),   [[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
    ("Pose 2", (0.6, 0.0, 0.3),   [[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
    ("Pose 3", (0.4, 0.2, 0.25),  [[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
]

SPEED = 0.2   # rad/s



with Robot() as robot:
    for name, pos, R in POSES:
        T_target = make_pose(pos, R)
        q_now = np.array(robot.joint_values, dtype=float)

        q_sol, converged, iters = inverse_kinematics(T_target, q_now)
        ok, pos_err, rot_err = check_solution(T_target, q_sol)
        safe, problems = is_safe(q_sol, q_now)

        print(f"\n=== {name} ===")
        print(f"IK converged: {converged} after {iters} iterations")
        print(f"FK check: position error {pos_err:.3f} mm, rotation error {rot_err:.5f} rad")
        print("Joint solution (rad):", q_sol)
        print("FK of solution (mm):\n", forward_kinematics(q_sol))

        if not ok:
            print("FK does not match the target pose. NOT moving.")
            continue
        if not safe:
            print("Safety check failed:", problems, "NOT moving.")
            continue

        input(f"Press Enter to move to {name} (hand on the e-stop)")
        robot.set_joint_targets(q_sol, speed=SPEED, wait=True)
        input("Arm stopped. Take the photo, then press Enter for the next pose")

    robot.stop()
    print("\nDone.")