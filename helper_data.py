# repair_and_diagnose_npz.py
# Diagnose + Repair skeleton NPZ dataset.
# - Diagnose: visibility ratio, motion-energy alignment, anomalies
# - Repair: center on energy peak, clamp, per-sequence z-score, mask handling, save fixed NPZs
# Usage:
#   python repair_and_diagnose_npz.py --root ./skeleton_npz_new --out ./diagnostics --fix --save ./flushed_data --T 64
# Options:
#   --min-vis 0.6 (visibility threshold), --keep-lowvis (still save low-vis samples),
#   --clamp 4.0 (winsorize bound), --workers 8, --force-H2 (pad to H=2 with zeros)

import os
import re
import json
import csv
import math
import glob
import argparse
import numpy as np
from typing import Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# -------------------- Utils --------------------

def infer_label_from_path(path: str) -> str:
    """Infer class label from filename stem like 'afraid_13.npz' -> 'afraid'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^(.*)_(\d+)$", stem)
    return m.group(1) if m else stem

def valid_mask_from_X(X: np.ndarray) -> np.ndarray:
    """Fallback mask when 'mask' is absent: mark joints valid if finite and not all-zero."""
    finite = np.isfinite(X).all(axis=-1)             # (T,H,21)
    nonzero = ~(np.isclose(X, 0.0).all(axis=-1))     # (T,H,21)
    return finite & nonzero

def motion_energy(X: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Per-frame motion energy = mean L2 displacement across visible joints (between consecutive frames)."""
    T = X.shape[0]
    if T < 2:
        return np.zeros((T,), dtype=np.float32)
    d = X[1:] - X[:-1]  # (T-1,H,21,3)
    if mask is not None:
        m = mask[1:] & mask[:-1]  # visible on both frames
        d = d * m[..., None]
        denom = np.maximum(m.sum(axis=(1, 2)), 1)  # (T-1,)
    else:
        denom = np.full((T-1,), X.shape[1]*X.shape[2], dtype=np.int32)
    disp = np.linalg.norm(d, axis=-1).sum(axis=(1, 2)) / denom  # (T-1,)
    energy = np.concatenate([np.zeros((1,), dtype=disp.dtype), disp], axis=0)
    return energy

def peak_center_distance(energy: np.ndarray) -> float:
    """Distance of energy peak to the clip center, normalized to [0,1]."""
    if energy.size == 0:
        return 1.0
    T = energy.size
    peak_idx = int(np.argmax(energy))
    center = (T - 1) / 2.0
    return abs(peak_idx - center) / center if center > 0 else 1.0

def hand_visibility(mask: np.ndarray) -> Dict[str, float]:
    """Compute visibility ratios: any, left, right, both, one-only."""
    T, H, J = mask.shape
    any_vis = (mask.any(axis=(1, 2))).mean()
    stats = {"any_vis_ratio": float(any_vis)}
    if H == 1:
        left = mask[:, 0, :].any(axis=1)
        stats.update({
            "left_vis_ratio":  float(left.mean()),
            "right_vis_ratio": 0.0,
            "both_vis_ratio":  0.0,
            "one_only_ratio":  float(left.mean())
        })
    else:
        left  = mask[:, 0, :].any(axis=1)
        right = mask[:, 1, :].any(axis=1)
        both  = left & right
        one_only = (left ^ right)
        stats.update({
            "left_vis_ratio":  float(left.mean()),
            "right_vis_ratio": float(right.mean()),
            "both_vis_ratio":  float(both.mean()),
            "one_only_ratio":  float(one_only.mean())
        })
    return stats

def detect_anomalies(X: np.ndarray) -> Dict[str, Any]:
    """Simple sanity checks on coordinates."""
    stats = {}
    stats["has_nan"] = bool(np.isnan(X).any())
    stats["has_inf"] = bool(np.isinf(X).any())
    max_abs = float(np.max(np.abs(X))) if X.size else 0.0
    stats["max_abs"] = max_abs
    stats["too_large"] = bool(max_abs > 5.0)  # sentinel threshold before clamp
    return stats

# -------------------- Repair ops --------------------

def center_window_indices(T: int, center_idx: int, T_out: int) -> Tuple[int, int]:
    """Return [start, end) indices of length T_out centered around center_idx with edge padding."""
    half = T_out // 2
    start = center_idx - half
    end = start + T_out
    # shift if exceeds bounds
    if start < 0:
        end -= start
        start = 0
    if end > T:
        start -= (end - T)
        end = T
        if start < 0:
            start = 0
    # If the slice is still shorter than T_out (when T < T_out), we'll pad later.
    return start, end

def slice_with_edge_pad(arr: np.ndarray, start: int, end: int, T_out: int) -> np.ndarray:
    """
    Slice arr[start:end,...] and pad with edge frames to reach T_out.
    arr is (T, ...) or (T, H, 21, 3) / (T,H,21)
    """
    sliced = arr[start:end]
    T_cur = sliced.shape[0]
    if T_cur == T_out:
        return sliced
    if T_cur <= 0:
        # nothing valid -> just repeat zeros of expected trailing dims
        trailing = arr.shape[1:]
        return np.zeros((T_out,) + trailing, dtype=arr.dtype)
    # pad by repeating first/last frame
    pad_front = max(0, (T_out - T_cur) // 2)
    pad_back = T_out - T_cur - pad_front
    front = np.repeat(sliced[:1], pad_front, axis=0) if pad_front > 0 else np.empty((0,)+sliced.shape[1:], dtype=arr.dtype)
    back  = np.repeat(sliced[-1:], pad_back, axis=0) if pad_back > 0 else np.empty((0,)+sliced.shape[1:], dtype=arr.dtype)
    return np.concatenate([front, sliced, back], axis=0)

def clamp_array(X: np.ndarray, clamp_v: float) -> np.ndarray:
    """Winsorize/clamp by absolute value."""
    if clamp_v is None or clamp_v <= 0:
        return X
    return np.clip(X, -clamp_v, clamp_v)

def zscore_per_sequence(X: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Per-sequence z-score using only valid entries (mask==True).
    Compute mean/std over all valid entries and normalize the whole X.
    """
    valid = mask[..., None]  # (T,H,21,1)
    # Avoid empty valid set
    denom = max(int(valid.sum()), 1)
    mean = (X * valid).sum() / denom
    var  = (((X - mean) * valid) ** 2).sum() / denom
    std = float(np.sqrt(var) + eps)
    Xn = (X - float(mean)) / std
    # Keep invalid entries as zero for stability
    Xn[~mask] = 0.0
    return Xn.astype(np.float32)

def maybe_pad_to_H2(X: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optionally pad to H=2 if H==1 by appending a zero hand channel with mask=False.
    """
    T, H, J, C = X.shape
    if H >= 2:
        return X, mask
    X2 = np.concatenate([X, np.zeros((T,1,J,C), dtype=X.dtype)], axis=1)
    m2 = np.concatenate([mask, np.zeros((T,1,J), dtype=mask.dtype)], axis=1)
    return X2, m2

# -------------------- IO --------------------

def safe_load_npz(path: str):
    with np.load(path, allow_pickle=True) as d:
        keys = set(d.files)
        if "X" not in keys:
            raise KeyError(f"missing X; keys={keys}")
        X = d["X"]
        if X.ndim != 4 or X.shape[2] != 21 or X.shape[3] != 3:
            raise ValueError(f"X shape invalid: {X.shape}")
        mask = d["mask"] if "mask" in keys else None
        meta = d["meta"].item() if "meta" in keys and d["meta"].ndim == 0 else None
    return X, mask, meta

def save_npz(out_path: str, X: np.ndarray, mask: np.ndarray, meta: Dict[str, Any]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, X=X.astype(np.float32), mask=mask.astype(bool), meta=json.dumps(meta, ensure_ascii=False))

# -------------------- Per-sample pipeline --------------------

def diagnose_only(X: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """Compute basic diagnostics for a sample (without modification)."""
    if mask is None:
        mask = valid_mask_from_X(X)
    vis = hand_visibility(mask)
    e = motion_energy(X, mask)
    an = detect_anomalies(X)
    return {
        "valid_frame_ratio": float((mask.any(axis=(1,2))).mean()),
        "peak_center_dist": float(peak_center_distance(e)),
        "energy_mean": float(np.mean(e)),
        "energy_std":  float(np.std(e)),
        "too_large": an["too_large"],
        "max_abs": an["max_abs"],
        **vis
    }

def repair_sample(
    X: np.ndarray,
    mask: np.ndarray,
    T_out: int = 64,
    clamp_v: float = 4.0,
    force_H2: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply: energy-peak alignment -> clamp -> z-score -> (optional H pad). Return X_fix, mask_fix, meta_extra."""
    if mask is None:
        mask = valid_mask_from_X(X)

    # 1) center on energy peak
    energy = motion_energy(X, mask)
    c_idx = int(np.argmax(energy))
    s, e = center_window_indices(T=X.shape[0], center_idx=c_idx, T_out=T_out)
    Xc   = slice_with_edge_pad(X,   s, e, T_out)
    mc   = slice_with_edge_pad(mask, s, e, T_out)

    # 2) clamp
    Xc = clamp_array(Xc, clamp_v=clamp_v)

    # 3) z-score per sequence (valid entries only)
    Xn = zscore_per_sequence(Xc, mc)

    # 4) optional: pad to H=2
    if force_H2:
        Xn, mc = maybe_pad_to_H2(Xn, mc)

    meta_extra = {
        "fixed_T": int(T_out),
        "center_on_peak_idx": int(c_idx),
        "slice": [int(s), int(e)],
        "clamp_abs": float(clamp_v),
        "zscore": True,
        "force_H2": bool(force_H2)
    }
    return Xn, mc, meta_extra

# -------------------- Worker --------------------

def process_file(
    path: str,
    args
) -> Dict[str, Any]:
    """
    Diagnose and (optionally) repair one file.
    Returns a row dict (also used for CSV).
    """
    row = {
        "path": path,
        "label": infer_label_from_path(path),
        "ok": False,
        "error": "",
        "saved": False
    }
    try:
        X, mask, meta = safe_load_npz(path)
        base_diag = diagnose_only(X, mask)

        row.update({
            "T": X.shape[0], "H": X.shape[1],
            **base_diag
        })
        row["ok"] = True

        if not args.fix:
            return row

        # visibility gate
        if (not args.keep_lowvis) and base_diag["valid_frame_ratio"] < args.min_vis:
            row["error"] = f"low_visibility({base_diag['valid_frame_ratio']:.3f})"
            return row

        # repair
        X_fix, m_fix, meta_extra = repair_sample(
            X, mask,
            T_out=args.T,
            clamp_v=args.clamp,
            force_H2=args.force_H2
        )

        # final sanity (optional)
        final_diag = diagnose_only(X_fix, m_fix)
        # decide save path
        rel = os.path.basename(path)
        out_path = os.path.join(args.save, rel)
        # update meta and save
        meta_out = meta_extra if meta is None else {**(meta if isinstance(meta, dict) else {}), **meta_extra}
        save_npz(out_path, X_fix, m_fix, meta_out)

        row["saved"] = True
        row["out_path"] = out_path
        row["fixed_T"] = int(args.T)
        row["final_valid_ratio"] = final_diag["valid_frame_ratio"]
        row["final_peak_center_dist"] = final_diag["peak_center_dist"]
        row["final_max_abs"] = final_diag["max_abs"]
        return row

    except Exception as ex:
        row["error"] = repr(ex)
        return row

# -------------------- Aggregation & Reporting --------------------

def save_csv(rows, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    path = os.path.join(out_dir, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path

def aggregate(rows, out_dir):
    total = len(rows)
    ok = sum(r.get("ok", False) for r in rows)
    bad = total - ok
    saved = sum(r.get("saved", False) for r in rows)
    lowvis_skipped = sum(1 for r in rows if r.get("error","").startswith("low_visibility"))
    errors = [r for r in rows if (r.get("error") and not r.get("error","").startswith("low_visibility"))]

    # quick rates
    vis70 = sum(1 for r in rows if r.get("ok") and r.get("valid_frame_ratio",0.0) >= 0.70)
    vis50 = sum(1 for r in rows if r.get("ok") and r.get("valid_frame_ratio",0.0) >= 0.50)
    centered = sum(1 for r in rows if r.get("ok") and r.get("peak_center_dist",1.0) <= 0.25)
    too_large = sum(1 for r in rows if r.get("ok") and r.get("max_abs",0.0) > 5.0)

    # save CSVs
    per_csv = save_csv(rows, out_dir, "diagnostics_and_repairs.csv")

    print("==== NPZ Diagnose + Repair Report ====")
    print(f"Total files        : {total}")
    print(f"Readable & OK      : {ok}")
    print(f"Abnormal/Unread    : {bad}")
    print("")
    print(f"Visibility >= 70%  : {vis70} ({vis70/max(1,total):.1%})")
    print(f"Visibility >= 50%  : {vis50} ({vis50/max(1,total):.1%})")
    print(f"Peak near center   : {centered} ({centered/max(1,total):.1%})")
    print(f"Too-large coords   : {too_large} ({too_large/max(1,total):.1%})")
    print("")
    print(f"Saved (repaired)   : {saved}")
    print(f"Skipped low-vis    : {lowvis_skipped}  # set --keep-lowvis to still save")
    print(f"Other errors       : {len(errors)}")
    if errors[:5]:
        print("Examples of errors:")
        for r in errors[:5]:
            print(" -", r.get("path"), "->", r.get("error"))
    print("")
    print(f"Per-sample CSV     : {per_csv}")

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./flushed_data_new", help="Directory containing *.npz")
    ap.add_argument("--out",  type=str, default="./diagnostics", help="Output dir for CSV")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2), help="Parallel threads")
    ap.add_argument("--win", type=int, default=16, help="Window length for energy smoothing (diagnostics only)")

    # Repair options
    # ap.add_argument("--fix", action="store_true", help="Enable repair and save fixed NPZs")
    ap.add_argument("--save", type=str, default="./npz_fixed",
                    help="Output dir for repaired NPZs (originals untouched)")
    ap.add_argument("--T", type=int, default=64, help="Target sequence length after realignment")
    ap.add_argument("--clamp", type=float, default=4.0, help="Winsorize |xyz| to this bound (<=0 disables)")
    ap.add_argument("--min-vis", type=float, default=0.60, help="Min visibility ratio to save (unless --keep-lowvis)")
    ap.add_argument("--keep-lowvis", action="store_true", help="Still save low-visibility samples")
    ap.add_argument("--force-H2", action="store_true", help="Pad to H=2 if H==1")
    ap.add_argument("--no-fix", action="store_true",
                    help="Disable repair; run diagnostics only")

    args = ap.parse_args()
    args.fix = not args.no_fix

    files = sorted(glob.glob(os.path.join(args.root, "*.npz")))
    if not files:
        print(f"[INFO] No .npz found in: {args.root}")
        return

    rows = []
    bar = tqdm(total=len(files), desc="Diagnose & Repair", ncols=100) if tqdm else None

    def _task(p):
        return process_file(p, args)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_task, p) for p in files]
        for fu in as_completed(futs):
            rows.append(fu.result())
            if bar: bar.update(1)
    if bar: bar.close()

    aggregate(rows, args.out)

if __name__ == "__main__":
    main()
