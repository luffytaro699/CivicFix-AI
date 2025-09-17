import os
from pathlib import Path
from yt_dlp import YoutubeDL

# ---------------------------
# Departments and keywords
# ---------------------------
departments = {
    "Street Lights": [
        "street light not working", "broken street light", "street light outage",
        "street lamp flickering", "street light repair", "damaged street light"],
    }

# ---------------------------
# Download and Rename
# ---------------------------
def download_videos(dept_name, keywords, max_results=10):
    dept_dir = Path("videos_dataset") / dept_name
    dept_dir.mkdir(parents=True, exist_ok=True)

    for keyword in keywords:
        query = f"ytsearch{max_results}:{keyword}"
        ydl_opts = {
            "format": "mp4",
            "ignoreerrors": True,
            "quiet": False,
            "outtmpl": str(dept_dir / "%(title).100s.%(ext)s")
        }
        print(f"🔍 Searching for: {keyword}")
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([query])

    # Rename everything sequentially after all downloads
    videos = sorted(
        [f for f in dept_dir.iterdir() if f.suffix.lower() in ['.mp4', '.webm', '.mkv', '.mov']],
        key=lambda x: x.stat().st_mtime
    )
    for idx, vid in enumerate(videos, start=1):
        new_name = f"{idx:06d}.mp4"
        new_path = dept_dir / new_name
        vid.rename(new_path)
    print(f"✅ Renamed {len(videos)} videos in {dept_name}")

# ---------------------------
# Main Runner
# ---------------------------
for dept, keywords in departments.items():
    print(f"\n📦 Downloading videos for: {dept}")
    download_videos(dept, keywords, max_results=10)  # each keyword downloads 20 results

print("\n🎉 All videos downloaded and renamed successfully!")
