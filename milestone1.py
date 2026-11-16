import numpy as np
from xarm7_lib import Robot
import importlib
_kin = importlib.import_module("6_2_inverse_kinematics")
forward_kinematics, move_ee, report = _kin.forward_kinematics, _kin.move_ee, _kin.report
np.set_printoptions(precision=4, suppress=True)

with Robot() as robot:
    try:
        robot.free_drive()
        input("Free-drive the arm to a comfortable pose (room to move +x, +y, +z), then press Enter\n")
        robot.end_free_drive()
    except NotImplementedError:
        print("Simulator: free drive not available, starting from the current pose.")

    q_start = np.array(robot.joint_values)
    print("Start joints (rad):", q_start)
    print("Start gripper pose (mm):\n", forward_kinematics(q_start))

    moves = [
        ("+10 cm along x", dict(dp=[100.0, 0.0, 0.0])),
        ("+10 cm along y", dict(dp=[0.0, 100.0, 0.0])),
        ("+10 cm along z", dict(dp=[0.0, 0.0, 100.0])),
        ("rotate pi/3 about z", dict(dyaw=np.pi / 3)),
    ]

    for label, kwargs in moves:
        input(f"\nPress Enter to {label}")
        q_before = np.array(robot.joint_values)
        move_ee(robot, **kwargs)
        q_after = np.array(robot.joint_values)   
        report(label, q_before, q_after)

    robot.stop()
    print("\nDone.")