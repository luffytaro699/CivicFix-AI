import os
from pathlib import Path
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

# ---------------------------
# Departments and keywords
# ---------------------------
departments = {
   "Street Lights": [
        "street light not working", "broken street light", "street light outage",
        "street lamp flickering", "street light repair", "damaged street light"],
    }
   
# ---------------------------
# Image Crawling Function
# ---------------------------
def crawl_images(dept_name, keywords, max_num=150):
    dept_dir = Path("dataset") / dept_name
    dept_dir.mkdir(parents=True, exist_ok=True)

    for keyword in keywords:
        # Google
        google_crawler = GoogleImageCrawler(storage={'root_dir': str(dept_dir)})
        google_crawler.crawl(keyword=keyword, max_num=max_num, file_idx_offset='auto')

        # Bing
        bing_crawler = BingImageCrawler(storage={'root_dir': str(dept_dir)})
        bing_crawler.crawl(keyword=keyword, max_num=max_num, file_idx_offset='auto')

# ---------------------------
# Renaming Function
# ---------------------------
def rename_images(dept_name):
    dept_dir = Path("dataset") / dept_name
    images = sorted(
        [f for f in dept_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']],
        key=lambda x: x.stat().st_mtime
    )
    for idx, img_path in enumerate(images, start=1):
        new_name = f"{idx:06d}{img_path.suffix.lower()}"
        new_path = dept_dir / new_name
        img_path.rename(new_path)
    print(f"✅ Renamed {len(images)} images in {dept_name}")

# ---------------------------
# Main Runner
# ---------------------------
for dept, keywords in departments.items():
    print(f"\n🔍 Downloading for department: {dept}")
    crawl_images(dept, keywords, max_num=150)
    rename_images(dept)

print("\n🎉 All downloads complete!")
