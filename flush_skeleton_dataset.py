import os
import sys
import glob
import json
import math
import numpy as np

# ===================== Config =====================
INPUT_DIR = "./skeleton_npz_new"
OUTPUT_DIR = "./flushed_data_new"
FIXED_T = 64                 # target sequence length after resampling
SMOOTH_WIN = 3               # temporal smoothing window (odd >= 1); 1 disables smoothing
USE_ZSCORE = True            # apply global z-score after geometry & resampling
MAX_HANDS = 2                # expected hands dimension in input
# MediaPipe hand indices
WRIST = 0
INDEX_MCP = 5
PINKY_MCP = 17

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== Utils =====================

def log(msg: str):
    print(f"[INFO] {msg}")

def warn(msg: str):
    print(f"[WARNING] {msg}")

def err(msg: str):
    print(f"[ERROR] {msg}")

def to_masked(arr: np.ndarray):
    """Return (data, mask) where mask=True for valid points (not -1 in all coords)."""
    # valid if any coord != -1 (stricter: require x&y valid; keep simple here)
    valid = ~(np.all(arr == -1, axis=-1))  # shape: (T, H, J)
    data = arr.astype(np.float32).copy()
    data[~valid, :] = 0.0  # fill invalid with 0; mask retains validity
    return data, valid

def sort_hands_by_wrist_x(frame_xy: np.ndarray, frame_mask: np.ndarray):
    """
    Per-frame reorder hands by WRIST.x (ascending). Inputs are:
      frame_xy: (H, J, 3)   frame_mask: (H, J)
    Returns re-ordered copies.
    """
    H, J, _ = frame_xy.shape
    if H <= 1:
        return frame_xy, frame_mask
    # Get wrist x if valid; else use +inf to push invalid to the end (keep order)
    wrist_x = []
    for h in range(H):
        if frame_mask[h, WRIST]:
            wrist_x.append(frame_xy[h, WRIST, 0])
        else:
            wrist_x.append(np.inf)
    order = np.argsort(wrist_x)
    return frame_xy[order], frame_mask[order]

def center_scale_rotate_xy(seq_xyz: np.ndarray, mask: np.ndarray):
    """
    Geometry normalization on XY (Z untouched):
      - center by WRIST
      - scale by palm width (INDEX_MCP <-> PINKY_MCP)
      - rotate so vector (WRIST->INDEX_MCP) aligns with +x axis (2D)
    Inputs:
      seq_xyz: (T, H, J, 3), mask: (T, H, J)
    Returns normalized seq_xyz (same shape). Missing frames/hands/joints are skipped by mask.
    """
    out = seq_xyz.copy()
    T, H, J, _ = out.shape

    for t in range(T):
        # sort hands by wrist x each frame for consistency
        out[t], mask[t] = sort_hands_by_wrist_x(out[t], mask[t])

        for h in range(H):
            # Need WRIST valid to center
            if not mask[t, h, WRIST]:
                continue

            # Center (all joints in this hand)
            wrist_xy = out[t, h, WRIST, :2].copy()
            out[t, h, :, :2] -= wrist_xy  # (J, 2)

            # Scale by palm width if possible
            scale = None
            if mask[t, h, INDEX_MCP] and mask[t, h, PINKY_MCP]:
                p1 = out[t, h, INDEX_MCP, :2]
                p2 = out[t, h, PINKY_MCP, :2]
                scale = np.linalg.norm(p1 - p2)
            if scale is not None and scale > 1e-6:
                out[t, h, :, :2] /= scale

            # Rotate so WRIST->INDEX_MCP goes to +x
            if mask[t, h, INDEX_MCP]:
                vec = out[t, h, INDEX_MCP, :2]  # already centered (INDEX relative to WRIST)
                angle = math.atan2(vec[1], vec[0])  # radians
                # rotate by -angle
                c, s = math.cos(-angle), math.sin(-angle)
                R = np.array([[c, -s],
                              [s,  c]], dtype=np.float32)
                # apply rotation only to valid joints
                valid_j = mask[t, h]  # (J,)
                xy = out[t, h, valid_j, :2]  # (Jv, 2)
                out[t, h, valid_j, :2] = xy @ R.T

            # Z is kept as-is (you can also center/scale Z if desired)

    return out, mask

def resample_nearest(seq: np.ndarray, mask: np.ndarray, T_target: int):
    """
    Nearest-neighbor resampling along time to fixed length.
    Inputs: seq (T,H,J,3), mask (T,H,J) -> Outputs: (T_target, H, J, 3)/(T_target,H,J)
    """
    T = seq.shape[0]
    if T == T_target:
        return seq.copy(), mask.copy()
    idx = np.linspace(0, T - 1, T_target).astype(int)
    return seq[idx], mask[idx]

def masked_moving_average(seq: np.ndarray, mask: np.ndarray, win: int):
    """
    Apply temporal moving average with mask support along time axis (axis=0).
    seq: (T,H,J,3) float32; mask: (T,H,J) bool
    Returns smoothed seq (same shape). Mask unchanged.
    """
    if win <= 1 or win % 2 == 0:
        return seq
    T, H, J, C = seq.shape
    half = win // 2
    out = seq.copy()

    # Build weights = 1 for valid, 0 for invalid, broadcast over C
    w = mask.astype(np.float32)  # (T,H,J)
    wC = w[..., None]             # (T,H,J,1)

    # Pad for easy windowing
    pad_mode = "edge"
    seq_pad = np.pad(seq, ((half, half), (0, 0), (0, 0), (0, 0)), mode=pad_mode)
    wC_pad  = np.pad(wC,  ((half, half), (0, 0), (0, 0), (0, 0)), mode=pad_mode)

    # Convolution by sliding window (simple loop to avoid extra deps)
    for t in range(T):
        sl = slice(t, t + win)
        num = np.sum(seq_pad[sl] * wC_pad[sl], axis=0)        # (H,J,C)
        den = np.sum(wC_pad[sl], axis=0) + 1e-8               # (H,J,1)
        out[t] = num / den

    # Keep invalid positions at 0 (optional, not strictly needed)
    out[~mask] = 0.0
    return out

class RunningMeanStd:
    """Welford's online algorithm for mean/std over valid entries."""
    def __init__(self):
        self.n = 0
        self.mean = np.zeros(3, dtype=np.float64)
        self.M2 = np.zeros(3, dtype=np.float64)

    def update(self, x: np.ndarray, m: np.ndarray):
        # x: (...,3) float; m: (...) bool -> flatten valid rows
        valid = m.reshape(-1)
        if not np.any(valid):
            return
        xf = x.reshape(-1, 3)[valid]
        for v in xf:
            self.n += 1
            delta = v - self.mean
            self.mean += delta / self.n
            delta2 = v - self.mean
            self.M2 += delta * delta2

    def finalize(self):
        if self.n < 2:
            std = np.ones(3, dtype=np.float64)
        else:
            var = self.M2 / (self.n - 1)
            std = np.sqrt(np.maximum(var, 1e-12))
        return self.mean.astype(np.float32), std.astype(np.float32)


# ===================== Main =====================

def process_all():
    print(f"[INFO] Python version: {sys.version}")
    log(f"Input dir: {INPUT_DIR}")
    log(f"Output dir: {OUTPUT_DIR}")

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npz")))
    if not files:
        warn("No .npz files found. Exit.")
        return

    # First pass: load -> mask -> geometry -> resample -> smooth
    processed = []  # list of dicts with temp data before z-score
    rms = RunningMeanStd()

    for i, path in enumerate(files, 1):
        base = os.path.splitext(os.path.basename(path))[0]
        log(f"({i}/{len(files)}) Loading {path}")
        try:
            dat = np.load(path, allow_pickle=False)
            X = dat["hand_keypoints"].astype(np.float32)  # (T,H,21,3)
        except Exception as e:
            err(f"Failed to load {path}: {e}")
            continue

        # Make sure dimensions are as expected
        if X.ndim != 4 or X.shape[1] != MAX_HANDS or X.shape[2] != 21 or X.shape[3] != 3:
            warn(f"Unexpected shape {X.shape} in {path}; skipping.")
            continue

        # Build mask & zero invalids
        X, M = to_masked(X)  # X: zeros for invalid, M: bool

        # Geometry normalization on XY
        X, M = center_scale_rotate_xy(X, M)

        # Resample to fixed length
        Xr, Mr = resample_nearest(X, M, FIXED_T)

        # Temporal smoothing (masked)
        Xs = masked_moving_average(Xr, Mr, SMOOTH_WIN)

        # Update global stats (for z-score) on valid entries only
        rms.update(Xs, Mr)

        processed.append({
            "name": base,
            "X": Xs,   # (T,H,21,3)
            "M": Mr,   # (T,H,21)
            "src": path,
        })

    # Finalize mean/std
    mean, std = rms.finalize()
    log(f"Global mean (per coord): {mean.tolist()}")
    log(f"Global std  (per coord): {std.tolist()}")
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "global_stats.npz"),
        mean=mean, std=std, fixed_T=FIXED_T, smooth_win=SMOOTH_WIN
    )

    # Second pass: apply z-score and save
    for obj in processed:
        X = obj["X"]
        M = obj["M"]
        # z-score only on valid positions; keep invalids at 0 with mask=False
        if USE_ZSCORE:
            X[M] = (X[M] - mean) / std

        out_path = os.path.join(OUTPUT_DIR, f"{obj['name']}.npz")
        np.savez_compressed(
            out_path,
            X=X.astype(np.float32),        # normalized coords
            mask=M.astype(np.bool_),       # validity mask
            meta=json.dumps({
                "source": obj["src"],
                "fixed_T": FIXED_T,
                "smooth_win": SMOOTH_WIN,
                "zscore": USE_ZSCORE,
                "geom_norm": {
                    "center": "WRIST",
                    "scale": "palm width (INDEX_MCP-PINKY_MCP)",
                    "rotate": "align WRIST->INDEX_MCP to +x (2D)"
                }
            })
        )
        log(f"Saved: {out_path}")

    log("All done.")


if __name__ == "__main__":
    process_all()
