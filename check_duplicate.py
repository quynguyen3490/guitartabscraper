from skimage.metrics import structural_similarity as ssim
import numpy as np
import imagehash
from PIL import Image
import cv2

def check_duplicate(img1_cv, img2_cv):
    # 1. Kiểm tra đầu vào
    if img1_cv is None or img2_cv is None:
        return False

    # 2. Đảm bảo ảnh ở dạng Grayscale (nếu là ảnh màu 3 kênh thì chuyển về 1 kênh)
    def to_gray(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    gray1 = to_gray(img1_cv)
    gray2 = to_gray(img2_cv)

    # 3. Resize ảnh 2 về cùng kích thước với ảnh 1
    height, width = gray1.shape
    gray2_resized = cv2.resize(gray2, (width, height))

    # --- LỚP 1: SSIM (So sánh cấu trúc tổng thể) ---
    # win_size=7 để xử lý tốt các chi tiết nhỏ như nốt nhạc
    score_ssim, _ = ssim(gray1, gray2_resized, full=True)
    
    # --- LỚP 2: DHash (So sánh chi tiết nốt nhạc) ---
    # Chuyển từ mảng NumPy (OpenCV) sang PIL Image để imagehash có thể đọc được
    pil_img1 = Image.fromarray(gray1)
    pil_img2 = Image.fromarray(gray2_resized)
    
    hash1 = imagehash.dhash(pil_img1)
    hash2 = imagehash.dhash(pil_img2)
    hash_diff = hash1 - hash2 # Khoảng cách Hamming

    # Log để debug nếu cần
    print(f"DEBUG: SSIM = {score_ssim:.4f} | Hash Diff = {hash_diff}")

    # --- ĐIỀU KIỆN KẾT HỢP ---
    # SSIM > 0.9: Cùng bố cục, cùng font chữ, cùng lề
    # Hash Diff <= 2: Các nốt nhạc nằm đúng vị trí (ngưỡng 14 bạn để hơi cao, dễ bị nhận nhầm)
    is_duplicate = (score_ssim >= 0.80 and hash_diff <= 20)
    
    return is_duplicate


# Khi gọi hàm:
img1_cv = cv2.imread("Canon89.jpg")
img2_cv = cv2.imread("Canon90.jpg")

result = check_duplicate(img1_cv, img2_cv)
print(result)