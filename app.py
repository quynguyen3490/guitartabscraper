import os
import subprocess
import cv2
from ultralytics import YOLO
from PIL import Image
import shutil
from skimage.metrics import structural_similarity as ssim
import numpy as np
import imagehash
import check_duplicate as duplicate_checker
import check_duplicate_model as duplicate_model_checker

# ===== CONFIG =====
VIDEO_DIR = "cache\\video"
IMG_DIR = "cache\\img"
CROP_DIR = "cache\\crops"

MERGED_DIR = "merged"
os.makedirs(MERGED_DIR, exist_ok=True)

MODEL_PATH = r"runs\detect\runs\yolov8n_results5\weights\best.pt"

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

# ===== LOAD MODEL =====
model = YOLO(MODEL_PATH)

# ===== MERGE CROPS =====
def merge_crops_to_image_and_pdf():
    from collections import defaultdict
    from PIL import Image

    groups = defaultdict(list)

    os.makedirs(CROP_DIR, exist_ok=True)
    # nhóm theo video name
    for f in os.listdir(CROP_DIR):
        if not f.endswith(".jpg"):
            continue

        prefix = f.rsplit("_", 1)[0]
        groups[prefix].append(f)

    for video_name, files in groups.items():
        files = sorted(files, key=sort_by_index)

        images = [Image.open(os.path.join(CROP_DIR, f)).convert("RGB") for f in files]

        target_width = max(img.width for img in images)

        resized = []
        total_height = 0

        for img in images:
            if img.width != target_width:
                new_h = int(img.height * target_width / img.width)
                img = img.resize((target_width, new_h))

            resized.append(img)
            total_height += img.height

        merged = Image.new("RGB", (target_width, total_height), (255, 255, 255))

        y = 0
        for img in resized:
            merged.paste(img, (0, y))
            y += img.height

        output_dir = os.path.join(MERGED_DIR, video_name)
        os.makedirs(output_dir, exist_ok=True)

        # # save image
        # img_path = os.path.join(output_dir, f"{video_name}_all_tabs.jpg")
        # merged.save(img_path, quality=95)

        # save pdf
        pdf_path = os.path.join(output_dir, f"{video_name}_all_tabs.pdf")

        pages = []
        page_height = 3000

        y = 0
        while y < merged.height:
            page = merged.crop((0, y, merged.width, min(y + page_height, merged.height)))
            pages.append(page)
            y += page_height

        pages[0].save(pdf_path, save_all=True, append_images=pages[1:])

        print(f"Saved: {video_name}")

# ===== DOWNLOAD =====
def download_video(url):
    output_path = os.path.join(VIDEO_DIR, "%(title)s.%(ext)s")

    # --- THAY ĐỔI Ở ĐÂY: Thêm User-Agent và các Header cơ bản ---
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080][vcodec^=avc1]",
        "-o", output_path,
        # 1. Giả lập trình duyệt (Rất quan trọng)
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        # 2. Thêm Header chấp nhận nội dung
        "--add-header", "Accept: text/html,application/xhtml+xml;q=0.9,image/webp,*/*;q=0.8",
        url
    ]

    print(f"Bắt đầu tải video từ URL: {url}")
    try:
        # Thêm encoding='utf-8' để đảm bảo output chuẩn
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Tải video thành công!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi tải video {url}:")
        print(f"   Stdout: {e.stdout}")
        print(f"   Stderr: {e.stderr}")

def cleanup_temp_files(confirm=True):
    if not confirm:
        print("Skip cleanup")
        return

    for folder in [VIDEO_DIR, IMG_DIR, CROP_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)
            print(f"Cleaned: {folder}")

# ===== CAPTURE FRAME =====
def capture_frames(video_path, interval_sec, start_sec=0, end_sec=None):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    if end_sec is None or end_sec > duration:
        end_sec = duration

    frame_interval = int(interval_sec * fps)

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    frame_id = start_frame
    saved_count = 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_name = clean_filename(video_name)

    video_img_dir = os.path.join(IMG_DIR, video_name)
    os.makedirs(video_img_dir, exist_ok=True)

    while frame_id <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if not ret:
            break

        img_name = f"{video_name}_{saved_count}.jpg"
        img_path = os.path.join(video_img_dir, img_name)

        cv2.imwrite(img_path, frame)

        saved_count += 1
        frame_id += frame_interval

    cap.release()

def sort_by_index(filename):
    return int(filename.rsplit("_", 1)[-1].split(".")[0])

def detect_and_crop(similarity_threshold=0.96):
    duplicate_checker_model = duplicate_model_checker.ImageComparator(
        threshold=similarity_threshold)

    images = []
    for root, dirs, files in os.walk(IMG_DIR):
        for file in files:
            if file.endswith(".jpg"):
                images.append(os.path.join(root, file))
    images = sorted(images)

    last_crop = None
    saved_count = 0

    for img_path in images:
        img = cv2.imread(img_path)

        results = model(img, conf=0.5)

        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy

            if len(boxes) == 0:
                continue

            # 👉 lấy box lớn nhất (tab thường là vùng lớn nhất)
            areas = [(int((b[2]-b[0])*(b[3]-b[1])), b) for b in boxes]
            _, best_box = max(areas)

            x1, y1, x2, y2 = map(int, best_box)
            crop = img[y1:y2, x1:x2].copy()

            # 👉 bỏ crop trùng
            if last_crop is not None:
                is_duplicate, score = duplicate_checker_model.compare_advanced(last_crop, crop)
                # cv2.putText(crop, f"Model Sim: {sim_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if is_similar else (0, 0, 255), 2)
                # cv2.putText(crop, f"Sim: {score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if is_duplicate else (0, 0, 255), 2)
                # cv2.putText(crop, f"Duplicate (Sim: {score['final_score']:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if is_duplicate:
                    # continue
                    cv2.putText(crop, f"Duplicate (Sim: {score['final_score']:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    continue
                else:
                    cv2.putText(crop, f"Unique (Sim: {score['final_score']:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    last_crop = crop.copy()

            else:
                last_crop = crop.copy() 

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            crop_name = f"{base_name}.jpg"
            crop_path = os.path.join(CROP_DIR, crop_name)

            cv2.imwrite(crop_path, crop)

            # last_crop = crop
            saved_count += 1

def clean_filename(name):
    return (
        name.replace(" ", "_")
            .replace("&", "and")
            .replace("(", "")
            .replace(")", "")
    )

# ===== RUN PROCESS =====
def run_process(interval_sec, start_sec=0, end_sec=None, cleanup=True, type="links", links=None, videos=None, similarity_threshold=0.96):

    if type == "links":
        # ===== READ LINKS =====
        with open("yt-link.txt", "r") as f:
            links = [line.strip() for line in f if line.strip()]

        # ===== DOWNLOAD =====
        for url in links:
            try:
                download_video(url)
            except Exception as e:
                print(f"Download failed: {url} - {e}")

    # ===== CAPTURE =====
    for file in os.listdir(VIDEO_DIR):
        if file.endswith(".mp4"):
            video_path = os.path.join(VIDEO_DIR, file)
            capture_frames(video_path, interval_sec, start_sec, end_sec)

    # ===== DETECT + CROP =====
    detect_and_crop(similarity_threshold)
    merge_crops_to_image_and_pdf()
    cleanup_temp_files(cleanup)

# ===== MAIN =====
def main():
    interval_sec = float(input("Capture mỗi bao nhiêu giây? (vd: 1): "))

    start_sec = input("Bắt đầu từ giây (Enter = 0): ")
    start_sec = float(start_sec) if start_sec else 0

    end_sec = input("Kết thúc ở giây (Enter = hết video): ")
    end_sec = float(end_sec) if end_sec else None

    cleanup = input("Cleanup temporary files? (y/n): ").lower() == "y"

    run_process(interval_sec, start_sec, end_sec, cleanup)


if __name__ == "__main__":
    main()