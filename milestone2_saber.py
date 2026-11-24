import os
import importlib
import numpy as np
from xarm7_lib import Robot

from IKGradientDescent import inverse_kinematics, check_solution, is_safe
from saber_transform import (calibrate_ee_to_tip, save_transform, tip_pose,
                             ee_pose_for_tip, invert_transform)

_kin = importlib.import_module("6_2_inverse_kinematics")
forward_kinematics = _kin.forward_kinematics
move_ee = _kin.move_ee

np.set_printoptions(precision=4, suppress=True)

SAVED_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

DISTANCE_MM = 85.0

HOLD_EE_MM = np.array([450.0, 0.0, 400.0])

SPEED = 0.3   

REAL_ARM = bool(os.environ.get("ROBOT_IP", "").strip())


def rotation_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0, 1.0, 0])
        perp /= np.linalg.norm(perp)
        return 2 * np.outer(perp, perp) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def ee_target_for_blade(q_now, T_ee_tip, blade_dir):
    R_tip_now = tip_pose(q_now, T_ee_tip)[:3, :3]
    R_tip_des = rotation_between(R_tip_now[:, 2], blade_dir) @ R_tip_now

    T = np.eye(4)
    T[:3, :3] = R_tip_des @ T_ee_tip[:3, :3].T     
    T[:3, 3] = HOLD_EE_MM
    return T


def move_to_ee_pose(robot, T_target, label):
    q_now = np.array(robot.joint_values, dtype=float)

    seeds = [q_now]
    for k, d in [(6, 1.0), (6, -1.0), (4, 0.5), (0, 0.5), (0, -0.5)]:
        s = q_now.copy()
        s[k] += d
        seeds.append(s)

    best = None
    for seed in seeds:
        q_sol, converged, iters = inverse_kinematics(T_target, seed)
        ok, pos_err, rot_err = check_solution(T_target, q_sol)
        if not ok:
            continue
        safe, problems = is_safe(q_sol, q_now)
        violation = robot.check_safety(q_sol)
        if not safe or violation is not None:
            continue
        jump = np.max(np.abs(q_sol - q_now))
        if best is None or jump < best[1]:
            best = (q_sol, jump, pos_err, rot_err, iters)

    print(f"\n=== {label} ===")
    if best is None:
        print("No safe IK solution matched the target pose. NOT moving.")
        return False

    q_sol, jump, pos_err, rot_err, iters = best
    print(f"IK: {iters} iterations, FK check {pos_err:.3f} mm / {rot_err:.5f} rad, "
          f"largest joint move {jump:.2f} rad")
    print("Joint solution (rad):", q_sol)

    input(f"Press Enter to move to {label} (hand on the e-stop)")
    robot.set_joint_targets(q_sol, speed=SPEED, wait=True)
    return True


class FakeDetector:
    def __init__(self, robot):
        self.robot = robot
        self.tip_in_ee = np.array([800.0, 0.0, 0.0])
        self.handle_in_ee = np.array([-100.0, 0.0, 0.0])

    def query(self, color=None):
        T = forward_kinematics(self.robot.joint_values)
        tip = T[:3, :3] @ self.tip_in_ee + T[:3, 3] + np.random.normal(0, 3, 3)
        handle = T[:3, :3] @ self.handle_in_ee + T[:3, 3] + np.random.normal(0, 3, 3)
        pts = [tip / 1000.0, handle / 1000.0]
        np.random.shuffle(pts)
        return np.array(pts)


def main():
    if REAL_ARM:
        from libcapstone import LightsaberDetector
        detector = LightsaberDetector()
    above_saber = np.array(SAVED_JOINTS, dtype=float)
    if not REAL_ARM and not np.any(above_saber):
        above_saber = np.array([0.0, 0.15, 0.0, 1.0, 0.0, 0.85, 0.0])

    with Robot() as robot:
        if REAL_ARM:
            robot.register_visualizer(detector)
        else:
            detector = FakeDetector(robot)
            print("Simulator: using a fake detector in place of the camera.")

        # grasp 1
        input("\nPress Enter to start the grasp (hand on the e-stop)")
        print("Moving to the saved pose above the lightsaber")
        robot.set_joint_targets(above_saber, speed=SPEED, wait=True)
        print("Opening the gripper")
        robot.grip(1.0)
        print(f"Moving down {DISTANCE_MM} mm")
        move_ee(robot, dp=[0.0, 0.0, -DISTANCE_MM])
        print("Closing the gripper")
        robot.grip(0.0)
        input("Check that the jaws are holding the saber, then press Enter to lift it")
        print(f"Moving up {DISTANCE_MM} mm")
        move_ee(robot, dp=[0.0, 0.0, DISTANCE_MM])

        # gripper-to-tip transform
        input("\nPress Enter to measure the gripper-to-tip transform")
        T_ee_tip, _ = calibrate_ee_to_tip(robot, detector)
        save_transform(T_ee_tip)
        print("Saved to T_ee_tip.json")

        # grasp 2: horizontal, along left-right
        q_now = np.array(robot.joint_values, dtype=float)
        z_now = tip_pose(q_now, T_ee_tip)[:3, 2]
        y_dir = np.array([0.0, 1.0, 0.0]) if z_now[1] >= 0 else np.array([0.0, -1.0, 0.0])
        T_h = ee_target_for_blade(q_now, T_ee_tip, y_dir)
        if move_to_ee_pose(robot, T_h, "horizontal (left-right)"):
            report_tip(robot, T_ee_tip, y_dir)
            input("Holding horizontal. Press Enter for vertical")

        # grasp 3: vertical, tip up 
        q_now = np.array(robot.joint_values, dtype=float)
        up = np.array([0.0, 0.0, 1.0])
        T_v = ee_target_for_blade(q_now, T_ee_tip, up)
        if move_to_ee_pose(robot, T_v, "vertical (tip up)"):
            report_tip(robot, T_ee_tip, up)
            input("Holding vertical. Press Enter to finish")

        robot.stop()
        print("\nDone.")


def report_tip(robot, T_ee_tip, wanted_dir):
    T_tip = tip_pose(np.array(robot.joint_values, dtype=float), T_ee_tip)
    angle = np.degrees(np.arccos(np.clip(T_tip[:3, 2] @ wanted_dir, -1, 1)))
    print(f"Tip position (mm): {np.round(T_tip[:3, 3], 1)}")
    print(f"Blade direction:   {np.round(T_tip[:3, 2], 3)}  ({angle:.2f} deg from target)")


if __name__ == "__main__":
    main()