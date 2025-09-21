# ai_services/clean/clean_images.py
import os
import shutil
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import hashlib

# configurable
IMAGES_ROOT = "ai_services/dataset/images_dataset"
CLEAN_ROOT = "ai_services/dataset/clean/dataset/images_cleaned"
REVIEW_ROOT = "ai_services/dataset/images_review"
MIN_WIDTH = 200
MIN_HEIGHT = 200
BLUR_THRESHOLD = 100.0   # variance of Laplacian
COLOR_VAR_THRESHOLD = 10.0
DUP_HASHES = set()

os.makedirs(CLEAN_ROOT, exist_ok=True)
os.makedirs(REVIEW_ROOT, exist_ok=True)

def image_hash(im, size=8):
    im = im.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(im).astype(np.float32)
    avg = arr.mean()
    bits = arr > avg
    hashstr = ''.join('1' if v else '0' for v in bits.flatten())
    return hex(int(hashstr, 2))[2:]

def is_blurry(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD

def color_variance(np_img):
    return np.std(np_img.reshape(-1, 3), axis=0).mean()

def pass_basic_checks(path):
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return False, "cannot_open"
    w,h = im.size
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False, "too_small"
    np_img = np.array(im)
    if is_blurry(np_img):
        return False, "blurry"
    if color_variance(np_img) < COLOR_VAR_THRESHOLD:
        return False, "low_color_var"  # likely graphic / icon
    # duplicate check
    hsh = image_hash(im)
    if hsh in DUP_HASHES:
        return False, "duplicate"
    DUP_HASHES.add(hsh)
    return True, None

def process():
    for dept in sorted(os.listdir(IMAGES_ROOT)):
        src_dir = os.path.join(IMAGES_ROOT, dept)
        if not os.path.isdir(src_dir):
            continue
        dest_dir = os.path.join(CLEAN_ROOT, dept)
        review_dir = os.path.join(REVIEW_ROOT, dept)
        os.makedirs(dest_dir, exist_ok=True)
        os.makedirs(review_dir, exist_ok=True)

        for fname in tqdm(os.listdir(src_dir), desc=f"Cleaning {dept}", unit="img"):
            src_path = os.path.join(src_dir, fname)
            if not os.path.isfile(src_path):
                continue
            ok, reason = pass_basic_checks(src_path)
            if ok:
                shutil.copy2(src_path, os.path.join(dest_dir, fname))
            else:
                shutil.copy2(src_path, os.path.join(review_dir, f"{reason}__{fname}"))

    print("Image cleaning done. Cleaned images in:", CLEAN_ROOT)
    print("Questionable images moved to:", REVIEW_ROOT)

if __name__ == "__main__":
    process()
