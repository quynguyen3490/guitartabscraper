import os
import random
import shutil

IMG_DIR = "dataset/images"
LBL_DIR = "dataset/labels"

OUT_IMG_TRAIN = "dataset/images/train"
OUT_IMG_VAL = "dataset/images/val"
OUT_LBL_TRAIN = "dataset/labels/train"
OUT_LBL_VAL = "dataset/labels/val"

os.makedirs(OUT_IMG_TRAIN, exist_ok=True)
os.makedirs(OUT_IMG_VAL, exist_ok=True)
os.makedirs(OUT_LBL_TRAIN, exist_ok=True)
os.makedirs(OUT_LBL_VAL, exist_ok=True)

images = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]

random.shuffle(images)

split_idx = int(len(images) * 0.9)

train_imgs = images[:split_idx]
val_imgs = images[split_idx:]


def move_files(img_list, img_out, lbl_out):
    for img in img_list:
        name = os.path.splitext(img)[0]
        lbl = name + ".txt"

        shutil.move(os.path.join(IMG_DIR, img), os.path.join(img_out, img))
        shutil.move(os.path.join(LBL_DIR, lbl), os.path.join(lbl_out, lbl))


move_files(train_imgs, OUT_IMG_TRAIN, OUT_LBL_TRAIN)
move_files(val_imgs, OUT_IMG_VAL, OUT_LBL_VAL)

print("Done split train/val")