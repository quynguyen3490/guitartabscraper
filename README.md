# 🎸 YOLOv8 Object Detection & Music Data Project

This project implements an object detection system built with **YOLOv8**, designed to detect objects within images or videos. It is tightly integrated with a music data component, allowing it to process and display results alongside associated guitar tablature (TAB) sheets and lesson materials.

## 🚀 Features
*   **Object Detection:** Uses YOLOv8 models (`yolov8n.pt`, `yolo26n.pt`) for fast object detection inference.
*   **Music Data Management:** Organizes and stores associated media for various musical pieces (e.g., Canon in D, Bésame Mucho), including:
    *   PDFs of sheet music/tabs.
    *   JPG previews of tabs.
    *   MP4 video lessons.

## 📂 Project Structure Overview

```
.
├── .venv/             # Python virtual environment
├── app.py             # Core application logic (handles direct execution)
├── gradio_gui.py      # The Gradio web interface script
├── main.py            # Main entry point for running the application
├── requirements.txt   # List of project dependencies
├── dataset/           # Custom dataset folder
│   ├── data.yaml      # Configuration file for YOLO (defines classes, paths)
│   ├── images/        # Input images for training/validation
│   └── labels/        # Ground-truth bounding box annotations (.txt files)
│       ├── train/     # Training set labels
│       └── val/      # Validation set labels
├── merged/            # Processed and organized media assets (Music Tabs/Lessons)
│   ├── Autumn_Leaves_-_Fingerstyle_Lesson_+_TAB/
│   ├── Canon_in_D_-_Guitar_Lesson_+_TAB/
│   └── ... (other song folders)
├── runs/              # Directory to store training and prediction results
│   ├── detect/        # Results from inference runs
│   │   ├── train/     # Training run outputs (weights, metrics)
│   │   └── predict/   # Single image prediction outputs
│   └── ... (other result folders like yolov8n_results5)
├── New folder/        # Raw media files (MP4 videos)
│   ├── Autumn Leaves - Fingerstyle Lesson + TAB.mp4
│   └── ...
├── yolo26n.pt         # Pre-trained or custom trained model weights
└── yolov8n.pt         # Standard YOLOv8 nano model weights
```

## 🛠️ Getting Started

### Prerequisites
Make sure you have Python (3.9+) installed.

### Installation
1.  **Clone the repository:**
    ```bash
    git clone [repository-url]
    cd [project-directory]
    ```
2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # or .\venv\Scripts\activate # On Windows PowerShell
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage Modes

#### 🟢 Direct Execution (via Command Line)
Run the core application logic directly:
```bash
python app.py
```

#### 🌐 Web GUI Mode
Launch the interactive web interface using Gradio:
```bash
python gradio_gui.py
# The output will provide a local URL, e.g., http://127.0.0.1:7860
```

## ⚙️ Training the Model (Optional)

If you wish to train a custom model, use the `train.py` script:
```bash
python train.py --data dataset/data.yaml --epochs 50 --imgsz 640 # Adjust arguments as necessary
```

## 📚 Data Details

### Dataset Classes
The classes are defined in `dataset/data.yaml`. Based on the file names, common classes include:
*   `Francis Lai Love Story` (or variations)
*   `A Town with an Ocean View`
*   `Can't Help Falling In Love - Elvis Presley`
*   `Carrying You...`
*   `Fantasie (Weiss)`
*   `Graceful Ghost`
*   `Jazz Waltz No.2 - D. Shostakovich`
*   `Passacaglia (Tremolo Ver.)`

## 🤝 Contributing

Feel free to fork this repository and contribute! Please follow standard Python practices and ensure your changes are well-documented.

## 📄 License

This project is licensed under the [MIT License](LICENSE). (Assuming an MIT license based on common practice, please verify in the root directory!)
