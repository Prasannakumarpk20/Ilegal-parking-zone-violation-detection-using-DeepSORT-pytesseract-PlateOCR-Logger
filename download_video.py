"""
download_video.py — Downloads a suitable parking-lot CCTV sample video.

Uses yt-dlp to pull a publicly available overhead/angled parking lot video
from YouTube so you have a clear view of parking spaces for violation detection.

Usage:
    python download_video.py
"""

import subprocess
import sys
import os

# ── Choose a good parking lot video ──────────────────────────────────────────
# This is a real CCTV-style overhead parking lot clip used widely in CV demos.
# Change the URL if you prefer a different one.

VIDEO_URL  = "https://www.youtube.com/watch?v=PJ5xXXcfuTc"  # Parking lot surveillance
BACKUP_URL = "https://www.youtube.com/watch?v=zb7IHOlVJGE"  # Airport parking overhead
OUTPUT_FILE = "sample.mp4"

def run(cmd: list[str]) -> int:
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode

def main():
    print("=" * 60)
    print("  📥  PARKING LOT VIDEO DOWNLOADER")
    print("=" * 60)

    # ── Check yt-dlp is available ─────────────────────────────────
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        print("  ✅  yt-dlp found")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  ⚙   Installing yt-dlp …")
        subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp", "-q"], check=True)
        print("  ✅  yt-dlp installed")

    # ── Download best quality ≤ 720p (keeps file size manageable) ─
    print(f"\n  📹  Downloading: {VIDEO_URL}")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "--merge-output-format", "mp4",
        "-o", OUTPUT_FILE,
        "--no-playlist",
        VIDEO_URL,
    ]
    rc = run(cmd)

    if rc != 0:
        print(f"\n  ⚠   Primary URL failed — trying backup …")
        cmd[-1] = BACKUP_URL
        rc = run(cmd)

    if rc == 0 and os.path.exists(OUTPUT_FILE):
        size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        print(f"\n  ✅  Saved → {OUTPUT_FILE}  ({size_mb:.1f} MB)")
        print("  👉  Now run:  python main.py")
    else:
        print("\n  ❌  Download failed. Please manually download a parking lot video")
        print("      and save it as 'sample.mp4' in this folder.")
        print("\n  Good free sources:")
        print("    • https://www.pexels.com/search/videos/parking+lot+aerial/")
        print("    • https://pixabay.com/videos/search/parking/")
        print("    • https://www.youtube.com/results?search_query=parking+lot+cctv+overhead")
        sys.exit(1)


if __name__ == "__main__":
    main()
