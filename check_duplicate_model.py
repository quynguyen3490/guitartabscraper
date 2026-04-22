import cv2
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics.pairwise import cosine_similarity
import os
from skimage.metrics import structural_similarity as ssim


class ImageComparator:
    def __init__(self, threshold=0.9, use_gray=True, debug=True):
        """
        threshold: ngưỡng similarity để quyết định True/False
        use_gray: convert ảnh sang grayscale (tốt cho tab/music sheet)
        debug: bật log chi tiết
        """
        self.threshold = threshold
        self.use_gray = use_gray
        self.debug = debug

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained ResNet18
        weights = ResNet18_Weights.DEFAULT
        base_model = resnet18(weights=weights)
        # base_model = models.resnet18(pretrained=True)

        # Remove classification layer (fc)
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def _log(self, msg):
        if self.debug:
            print(f"[DEBUG] {msg}")

    def preprocess(self, img):
        """
        Input: cv2 image (BGR)
        Output: tensor (1, C, H, W)
        """
        if self.use_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = self.transform(img)

        # nếu grayscale thì thêm channel
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        img = img.unsqueeze(0).to(self.device)
        return img

    def extract_feature(self, img):
        """
        Extract feature vector từ ảnh
        """
        img_tensor = self.preprocess(img)

        with torch.no_grad():
            feat = self.model(img_tensor)  # (1, 512, 1, 1)
            feat = feat.view(-1).cpu().numpy()

        # Normalize (rất quan trọng)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm

        return feat

    def align_images(self,img1, img2):
        """
        Align img2 về img1 bằng ORB feature matching
        """
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(2000)

        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            return img2  # fallback

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 10:
            return img2  # không đủ match

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

        # estimate transform
        M, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts)

        if M is None:
            return img2

        h, w = img1.shape[:2]
        aligned = cv2.warpAffine(img2, M, (w, h))

        return aligned

    def compare(self, img1, img2, return_score=True, visualize=False):
        """
        So sánh 2 ảnh

        return:
            is_similar (bool)
            score (float)
        """
        self._log("Extract feature image 1...")
        f1 = self.extract_feature(img1)

        self._log("Extract feature image 2...")
        f2 = self.extract_feature(img2)

        # cosine similarity
        sim = cosine_similarity([f1], [f2])[0][0]

        self._log(f"Similarity score: {sim:.4f}")

        is_similar = sim >= self.threshold

        self._log(f"Result: {'SIMILAR' if is_similar else 'DIFFERENT'}")

        if visualize:
            self.visualize_diff(img1, img2)

        if return_score:
            return is_similar, sim
        else:
            return is_similar

    def compare_advanced(self, img1, img2, grid=6, visualize=False):
        """
        Advanced compare:
        - SSIM (global + diff ratio)
        - CNN patch similarity
        - Auto decision (không cần threshold cố định)

        return:
            result (bool)
            detail (dict)
        """

        self._log("=== ADVANCED COMPARE START ===")
        img2 = self.align_images(img1, img2)

        # ===== Resize về cùng size =====
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])

        img1_r = cv2.resize(img1, (w, h))
        img2_r = cv2.resize(img2, (w, h))

        # ===== SSIM =====
        gray1 = cv2.cvtColor(img1_r, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_r, cv2.COLOR_BGR2GRAY)

        ssim_score, diff = ssim(gray1, gray2, full=True)

        # diff map: càng nhỏ càng khác
        diff_ratio = np.mean(diff < 0.95)

        self._log(f"SSIM score: {ssim_score:.4f}")
        self._log(f"Diff ratio: {diff_ratio:.4f}")

        # ===== PATCH CNN =====
        patch_h = h // grid
        patch_w = w // grid

        patch_scores = []

        for i in range(grid):
            for j in range(grid):
                y1 = i * patch_h
                y2 = (i + 1) * patch_h
                x1 = j * patch_w
                x2 = (j + 1) * patch_w

                p1 = img1_r[y1:y2, x1:x2]
                p2 = img2_r[y1:y2, x1:x2]

                f1 = self.extract_feature(p1)
                f2 = self.extract_feature(p2)

                sim = cosine_similarity([f1], [f2])[0][0]
                patch_scores.append(sim)

        patch_scores = np.array(patch_scores)

        patch_mean = patch_scores.mean()
        patch_min = patch_scores.min()

        self._log(f"Patch mean: {patch_mean:.4f}")
        self._log(f"Patch min : {patch_min:.4f}")

        # ===== AUTO DECISION (không cần threshold cứng) =====
        # Logic thực chiến:
        # - SSIM cao + diff_ratio thấp → giống
        # - patch_min thấp → có vùng khác rõ
        # - kết hợp nhiều điều kiện

        score = (
            0.5 * ssim_score +
            0.3 * patch_mean +
            0.2 * patch_min
        )

        self._log(f"Final combined score: {score:.4f}")

        # heuristic decision
        if ssim_score > 0.95 and diff_ratio < 0.01:
            result = True
        elif patch_min < 0.75:
            result = False
        elif score > 0.92:
            result = True
        else:
            result = False

        self._log(f"FINAL RESULT: {'SIMILAR' if result else 'DIFFERENT'}")

        # ===== VISUALIZE =====
        if visualize:
            self._visualize_advanced(img1_r, img2_r, diff)

        return result, {
            "ssim": ssim_score,
            "diff_ratio": diff_ratio,
            "patch_mean": patch_mean,
            "patch_min": patch_min,
            "final_score": score
        }

    def _visualize_advanced(self, img1, img2, diff):
        """
        Hiển thị vùng khác nhau bằng heatmap
        """
        self._log("Visualizing advanced diff...")

        # normalize diff
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        diff_norm = diff_norm.astype("uint8")

        heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

        # highlight vùng khác mạnh
        mask = diff < 0.95
        highlight = img2.copy()
        highlight[mask] = [0, 0, 255]  # đỏ

        cv2.imshow("Image 1", img1)
        cv2.imshow("Image 2", img2)
        cv2.imshow("Diff Heatmap", heatmap)
        cv2.imshow("Highlight Differences", highlight)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # ===== CONFIG =====
    img_path1 = "t1.jpg"
    img_path2 = "t2.jpg"
    threshold = 0.9
    debug = True
    visualize = False

    # ===== CHECK FILE =====
    if not os.path.exists(img_path1):
        print(f"[ERROR] File not found: {img_path1}")
        return

    if not os.path.exists(img_path2):
        print(f"[ERROR] File not found: {img_path2}")
        return

    # ===== LOAD IMAGE =====
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    if img1 is None:
        print(f"[ERROR] Cannot read image: {img_path1}")
        return

    if img2 is None:
        print(f"[ERROR] Cannot read image: {img_path2}")
        return

    print("[INFO] Images loaded successfully")

    # ===== INIT MODEL =====
    comparator = ImageComparator(
        threshold=threshold,
        use_gray=True,
        debug=debug
    )

    result, detail = comparator.compare_advanced(
        img1,
        img2,
        grid=6,
        visualize=True
    )

    print(result)
    print(detail)


# ===== RUN =====
if __name__ == "__main__":
    main()