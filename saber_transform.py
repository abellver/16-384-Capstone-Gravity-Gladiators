import json
import time
import importlib
import numpy as np

_kin = importlib.import_module("6_2_inverse_kinematics")
forward_kinematics = _kin.forward_kinematics
move_ee = _kin.move_ee


SABER_COLOR = "blue"

def _extract_two_points(raw):
    if raw is None:
        return None

    if isinstance(raw, (list, tuple)):
        if len(raw) == 0:
            return None
        try:
            arr = np.asarray(raw, dtype=float)
            if arr.shape == (2, 3):           # [point, point]
                return arr[0], arr[1]
        except (TypeError, ValueError):
            pass
        return _extract_two_points(raw[0])    # a list of detections: take the first

    if isinstance(raw, dict):
        for a, b in (("start", "end"), ("p0", "p1"), ("handle", "tip"), ("a", "b")):
            if a in raw and b in raw:
                return np.asarray(raw[a], float), np.asarray(raw[b], float)
        for key in ("ends", "points", "endpoints"):
            if key in raw:
                return _extract_two_points(raw[key])

    for a, b in (("start", "end"), ("p0", "p1"), ("handle", "tip"), ("a", "b")):
        if hasattr(raw, a) and hasattr(raw, b):
            return np.asarray(getattr(raw, a), float), np.asarray(getattr(raw, b), float)
    for key in ("ends", "points", "endpoints"):
        if hasattr(raw, key):
            return _extract_two_points(getattr(raw, key))

    arr = np.asarray(raw, dtype=float)
    if arr.shape == (2, 3):
        return arr[0], arr[1]
    if arr.shape == (6,):
        return arr[:3], arr[3:]

    raise ValueError(f"Unrecognised detector output, adapt _extract_two_points: {raw!r}")


# returns the two saber end points in mm (base frame) or None if nothing was seen
def read_saber_points(detector, color=SABER_COLOR):
    raw = detector.query(color)
    pts = _extract_two_points(raw)
    if pts is None:
        return None
    p, q = (np.asarray(x, dtype=float).reshape(3) for x in pts)

    if max(np.linalg.norm(p), np.linalg.norm(q)) < 5.0:
        p, q = p * 1000.0, q * 1000.0
    return p, q


def read_saber_points_median(detector, n=15, wait=0.05, color=SABER_COLOR, max_tries=100):
    readings = []
    tries = 0
    while len(readings) < n and tries < max_tries:
        tries += 1
        r = read_saber_points(detector, color)
        if r is not None:
            readings.append(r)
        time.sleep(wait)
    if len(readings) < max(3, n // 3):
        return None

    ref_a, ref_b = readings[0]
    A, B = [], []
    for a, b in readings:
        if np.linalg.norm(a - ref_a) + np.linalg.norm(b - ref_b) > \
           np.linalg.norm(a - ref_b) + np.linalg.norm(b - ref_a):
            a, b = b, a
        A.append(a)
        B.append(b)
    return np.median(A, axis=0), np.median(B, axis=0)


def saber_frame(p1, p2, T_base_ee):
    p_ee = T_base_ee[:3, 3]
    if np.linalg.norm(p1 - p_ee) > np.linalg.norm(p2 - p_ee):
        tip, handle = p1, p2
    else:
        tip, handle = p2, p1

    z = tip - handle
    z /= np.linalg.norm(z)

    y_ee = T_base_ee[:3, 1]
    y = y_ee - (y_ee @ z) * z
    if np.linalg.norm(y) < 1e-3:
        raise ValueError("End effector y axis is parallel to the blade; y is undefined")
    y /= np.linalg.norm(y)

    x = np.cross(y, z)

    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, y, z
    T[:3, 3] = tip
    return T, handle


def invert_transform(T):
    R, p = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def average_transforms(Ts):
    Ts = np.asarray(Ts)
    p = Ts[:, :3, 3].mean(axis=0)
    M = Ts[:, :3, :3].mean(axis=0)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    T = np.eye(4)
    T[:3, :3] = U @ D @ Vt
    T[:3, 3] = p
    return T


def measure_once(robot, detector, n_readings=15):
    q = np.array(robot.joint_values, dtype=float)
    T_base_ee = forward_kinematics(q)
    pts = read_saber_points_median(detector, n=n_readings)
    if pts is None:
        return None
    T_base_tip, _ = saber_frame(pts[0], pts[1], T_base_ee)
    return invert_transform(T_base_ee) @ T_base_tip


CALIBRATION_MOVES = [
    ((0, 0, 0), 0.0),
    ((0, 0, 0), 0.35),
    ((0, 0, 0), -0.70),
    ((0, 0, 60), 0.35),
    ((40, -40, 0), 0.0),
    ((-40, 80, 0), 0.0),
]


def calibrate_ee_to_tip(robot, detector, moves=CALIBRATION_MOVES, n_readings=15,
                        settle=0.5, return_to_start=True, verbose=True):
    q_start = np.array(robot.joint_values, dtype=float)
    Ts = []

    for i, (dp, dyaw) in enumerate(moves):
        if np.any(dp) or dyaw != 0.0:
            move_ee(robot, dp=dp, dyaw=dyaw)
        time.sleep(settle)
        T = measure_once(robot, detector, n_readings)
        if T is None:
            if verbose:
                print(f"  pose {i + 1}: saber not detected, skipped")
            continue
        Ts.append(T)
        if verbose:
            print(f"  pose {i + 1}: tip offset (mm) {np.round(T[:3, 3], 1)}")

    if return_to_start:
        robot.set_joint_targets(q_start, speed=0.3, wait=True)

    if len(Ts) < 3:
        raise RuntimeError(f"Only {len(Ts)} good measurements; need at least 3")

    T_avg = average_transforms(Ts)
    pos_spread = max(np.linalg.norm(T[:3, 3] - T_avg[:3, 3]) for T in Ts)
    rot_spread = max(np.linalg.norm(_kin.rotation_error(T[:3, :3], T_avg[:3, :3])) for T in Ts)

    if verbose:
        print(f"\nT_ee->tip from {len(Ts)} poses (mm):\n{np.round(T_avg, 4)}")
        print(f"Saber length from end effector: {np.linalg.norm(T_avg[:3, 3]):.1f} mm")
        print(f"Worst single-pose deviation: {pos_spread:.1f} mm, {np.degrees(rot_spread):.2f} deg")
    return T_avg, (pos_spread, rot_spread)


def tip_pose(q, T_ee_tip):
    return forward_kinematics(q) @ T_ee_tip


def ee_pose_for_tip(T_base_tip, T_ee_tip):
    return T_base_tip @ invert_transform(T_ee_tip)


def save_transform(T, path="T_ee_tip.json"):
    with open(path, "w") as f:
        json.dump({"T_ee_tip_mm": np.asarray(T).tolist(), "saved": time.ctime()}, f, indent=2)


def load_transform(path="T_ee_tip.json"):
    with open(path) as f:
        return np.array(json.load(f)["T_ee_tip_mm"])


if __name__ == "__main__":
    import sys
    if "--probe" in sys.argv:
        from libcapstone import LightsaberDetector
        det = LightsaberDetector()
        help(det.query)
        for _ in range(10):
            raw = det.query(SABER_COLOR)
            print("raw:", repr(raw))
            try:
                print("  parsed (mm):", read_saber_points(det))
            except Exception as e:
                print("  parse failed:", e)
            time.sleep(0.5)