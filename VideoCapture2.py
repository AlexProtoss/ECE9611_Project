import os
import json
import subprocess
import shutil
from urllib.parse import urlparse, parse_qs

JSON_PATH = "MS-ASL/MSASL_test.json"  # download from
VIDEO_DIR = "videos_full"  # download to
CLIP_DIR = "BackupVideo"  # store clipped video by json data
LOG_PATH = "download_log.json"  # key log file to resume downloading

YT_DLP_BIN = None
FFMPEG_BIN = None

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(CLIP_DIR, exist_ok=True)

def resolve_binaries():
    global YT_DLP_BIN, FFMPEG_BIN

    if YT_DLP_BIN is None:
        YT_DLP_BIN = shutil.which("yt-dlp") or shutil.which("yt_dlp")
    if not YT_DLP_BIN:
        raise FileNotFoundError(
            "yt-dlp not found."
        )

    if FFMPEG_BIN is None:
        FFMPEG_BIN = shutil.which("ffmpeg")
    if not FFMPEG_BIN:
        raise FileNotFoundError(
            "ffmpeg not found."
        )

    print(f"yt-dlp: {YT_DLP_BIN}")
    print(f"ffmpeg: {FFMPEG_BIN}")


def load_log():
    if not os.path.exists(LOG_PATH):
        return {}, {}
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    text_counts = data.get("_text_counts", {})
    data.pop("_text_counts", None)
    return data, text_counts


def save_log(log_dict, text_counts):
    to_save = dict(log_dict)
    to_save["_text_counts"] = text_counts
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)


def make_row_key(row):
    url = str(row.get("url", "")).strip()
    st = str(row.get("start_time", "")).strip()
    et = str(row.get("end_time", "")).strip()
    text = str(row.get("text", row.get("clean_text", ""))).strip()
    return f"{url}|{st}|{et}|{text}"

def normalize_url(url: str) -> str:
    url = str(url).strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url.lstrip("/")
    return url


def get_youtube_id(url: str) -> str:
    url = normalize_url(url)
    parsed = urlparse(url)

    if parsed.hostname in ["www.youtube.com", "youtube.com"]:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]

    if parsed.hostname in ["youtu.be"]:
        return parsed.path.lstrip("/")

    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]

    return "".join(c for c in url if c.isalnum())


def download_video_if_needed(url: str, video_cache: dict) -> str:
    url = normalize_url(url)
    yt_id = get_youtube_id(url)

    if yt_id in video_cache:
        return video_cache[yt_id]

    video_path = os.path.join(VIDEO_DIR, f"{yt_id}.mp4")

    if not os.path.exists(video_path):
        print(f"Downloading video: {url}")
        cmd = [YT_DLP_BIN, "-f", "mp4", "-o", os.path.join(VIDEO_DIR, f"{yt_id}.%(ext)s"), url]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = (result.stderr or "") + (result.stdout or "")
            raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)
    else:
        print(f"Video already exists: {video_path}")

    video_cache[yt_id] = video_path
    return video_path


def compute_crop_from_row(row):
    x1 = float(row.get("box/0", 0.0))
    y1 = float(row.get("box/1", 0.0))
    x2 = float(row.get("box/2", 1.0))
    y2 = float(row.get("box/3", 1.0))

    width = int(row.get("width", 0))
    height = int(row.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Invalid width/height in row.")

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    if x2 <= x1:
        x2 = min(1.0, x1 + 0.999)
    if y2 <= y1:
        y2 = min(1.0, y1 + 0.999)

    crop_x = int(round(x1 * width))
    crop_y = int(round(y1 * height))
    crop_w = int(round((x2 - x1) * width))
    crop_h = int(round((y2 - y1) * height))

    if crop_x < 0:
        crop_x = 0
    if crop_y < 0:
        crop_y = 0
    if crop_x + crop_w > width:
        crop_w = width - crop_x
    if crop_y + crop_h > height:
        crop_h = height - crop_y

    crop_w = max(1, crop_w)
    crop_h = max(1, crop_h)

    return crop_w, crop_h, crop_x, crop_y


def make_safe_text_name(text: str) -> str:
    text = (text or "").strip()
    if not text:
        text = "clip"
    safe = "".join(c for c in text if c.isalnum() or c in ("_", "-"))
    return safe.lower() or "clip"


def create_cropped_clip(row, video_path, text_counts):
    start_time = float(row.get("start_time", 0.0))
    end_time = float(row.get("end_time", 0.0))
    duration = max(0.01, end_time - start_time)

    crop_w, crop_h, crop_x, crop_y = compute_crop_from_row(row)

    base_text = str(row.get("text", row.get("clean_text", "clip")))
    safe_text = make_safe_text_name(base_text)

    current_idx = int(text_counts.get(safe_text, 0)) + 1
    out_name = f"{safe_text}_{current_idx}.mp4"
    out_path = os.path.join(CLIP_DIR, out_name)

    print(f"Clipping to: {out_path}")

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss", f"{start_time:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-vf", f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-strict", "experimental",
        out_path,
    ]
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        error_msg = (result.stderr or "") + (result.stdout or "")
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )

    text_counts[safe_text] = current_idx
    return out_name, safe_text, current_idx


def main():
    # import sys
    # print(f"Python version: {sys.version}")

    resolve_binaries()

    log_dict, text_counts = load_log()
    print(f"Loaded log entries: {len(log_dict)}")
    print(f"Loaded text count entries: {len(text_counts)}")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of annotation objects.")

    video_cache = {}

    for idx, row in enumerate(data):
        row_key = make_row_key(row)
        print(f"\n[INFO] Processing row {idx + 1}/{len(data)}")

        # Check log: skip if already successful
        entry = log_dict.get(row_key)
        if entry and entry.get("status") == "success":
            print(f"Row already done: {row_key}")
            continue

        try:
            url = row.get("url", "")
            video_path = download_video_if_needed(url, video_cache)

            out_name, text_used, idx_used = create_cropped_clip(row, video_path, text_counts)

            log_dict[row_key] = {
                "status": "success",
                "filename": out_name,
                "text": text_used,
                "text_index": idx_used,
            }
            save_log(log_dict, text_counts)
            print(f"Success: {out_name}")

        except Exception as e:
            msg = str(e)
            # detecting 403 banned error, which would be tried in next download process
            is_403 = ("403" in msg)
            print(f"Failed: {idx + 1}: {msg}")

            log_dict[row_key] = {"status": "error", "error_message": msg, "is_403": is_403}
            save_log(log_dict, text_counts)

    print("\n Finished.")


if __name__ == "__main__":
    main()
