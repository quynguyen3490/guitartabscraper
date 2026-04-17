from ultralytics import YOLO
import torch

print(torch.cuda.get_device_name(0))

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/data.yaml",
        project="runs",
        name="yolov8n_results",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0
    )

if __name__ == "__main__":
    main()