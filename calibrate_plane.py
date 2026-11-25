import time
import numpy as np
from xarm7_lib import Robot
from libcapstone import LightsaberDetector

SABER_COLOR = "green"
MIN_PTS = 12
NUM_PTS = 25
MAX_REJECTS = 3
MAX_JUMP = 0.15
MIN_SPREAD = 0.2

def wait_for_saber(detector):
    print("show lightsaber in front of camera \n")
    while (detector.query(SABER_COLOR) is None):
        time.sleep(0.05)
    print("lightsaber detected \n")

def record_sweep(detector):
    # clear previous plane 
    detector.draw_plane(None)
    wait_for_saber(detector)
    input("Press Enter, then sweep the tip across your plane")

    points = []
    last = None
    prev = None
    rejects = 0
    start = time.monotonic()

    while len(points) < NUM_PTS and time.monotonic() - start < 8.0:
        ends = detector.query(SABER_COLOR, max_age = 0.2)
        
        if ends is None or (last is not None and np.array_equal(ends, last)):
            time.sleep(0.02)
            continue
        last = ends
        
        tip = ends[0]
        if prev is not None and (np.linalg.norm(ends[1] - prev) < np.linalg.norm(tip - prev)):
            tip = ends[1]

        if prev is not None and (np.linalg.norm(tip - prev) > MAX_JUMP):
            rejects += 1
            if rejects < MAX_REJECTS:
                continue
        
        rejects = 0
        points.append(tip)
        prev = tip
    
    print(f"\n Recorded {len(points)} points")
    return np.array(points)

def fit_plane(points, camera_pos):
    centroid = np.mean(points, axis=0)
    U, S, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    n = Vt[-1]

    if n @ (camera_pos - centroid) < 0:
        n = -n 

    offset = n @ centroid
    rms = S[2] / np.sqrt(len(points))
    return n, offset, centroid, S, rms

detector = LightsaberDetector()
with Robot() as robot: 
    robot.register_visualizer(detector)

    camera_pos = detector.handeye.camera_position

    while True: 
        points = record_sweep(detector)
        if len(points) < MIN_PTS:
            print("Too few points\n")
            continue

        n, offset, centroid, S, rms = fit_plane(points, camera_pos)
        if S[1] / S[0] < MIN_SPREAD:
            print("points are on one line, try again\n")
            continue

        print("\nPlane  n · x = d  in the robot base frame (meters):")
        print(f"  n = {np.round(n, 4)}")
        print(f"  d = n · c = {offset:.4f}")
        print(f"  [{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}] · x = {offset:.3f}")
        print(f"  {n[0]:+.3f} x {n[1]:+.3f} y {n[2]:+.3f} z = {offset:.3f}")

        radius = np.max(np.linalg.norm(points - centroid, axis=1))
        plane_size = np.maximum(0.3, 2.2 * radius)
        detector.draw_plane((n[0], n[1], n[2], -offset),
                            center = centroid, size = plane_size, color = 0x3080FF)