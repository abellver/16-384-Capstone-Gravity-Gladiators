import numpy as np
import importlib
from xarm7_lib import Robot

_kin = importlib.import_module("6_2_inverse_kinematics")
move_ee = _kin.move_ee

# ---- Fill these in before running ----
# Joint angles (rad) of the pose above the lightsaber. Paste the
# "as radians [...]" list printed by: python -m xarm7_lib.free_drive
SAVED_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# How far the gripper has to move down to reach the handle (mm)
DISTANCE_MM = 85.0
# ---------------------------------------

with Robot() as robot:
    # 1. Move to the saved pose above the lightsaber
    print("1. Moving to the saved pose")
    robot.set_joint_targets(SAVED_JOINTS, speed=0.3, wait=True)

    # 2. Open the gripper
    print("2. Opening the gripper")
    robot.grip(1.0)

    # 3. Move down by the measured distance
    print(f"3. Moving down {DISTANCE_MM} mm")
    move_ee(robot, dp=[0.0, 0.0, -DISTANCE_MM])

    # 4. Close the gripper
    print("4. Closing the gripper")
    robot.grip(0.0)
    input("Check that the jaws are holding the saber, then press Enter to lift it")

    # 5. Move up by the same distance
    print(f"5. Moving up {DISTANCE_MM} mm")
    move_ee(robot, dp=[0.0, 0.0, DISTANCE_MM])

    robot.stop()
    print("Done")