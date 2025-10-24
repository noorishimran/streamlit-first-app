# YOLOv8 Streamlit App Setup Instructions

## Prerequisites
- Python 3.8 or higher
- pip package manager

## Installation Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Images
- Place your images in the `images/` folder
- Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp

### 3. Run the App
```bash
streamlit run yolo_streamlit_app.py
```

## Features
- **Random Image Selection**: Automatically picks a random image from the 'images' folder
- **YOLOv8 Object Detection**: Uses the latest YOLOv8 model for object detection
- **Interactive Interface**: Adjustable confidence threshold
- **Visual Results**: Shows original image and detection results with bounding boxes
- **Detection Details**: Lists all detected objects with confidence scores

## Usage
1. Open the app in your browser (usually http://localhost:8501)
2. Click "Pick Random Image" to select an image from the 'images' folder
3. The app will automatically run YOLOv8 detection on the selected image
4. View the results with bounding boxes and object labels
5. Adjust the confidence threshold in the sidebar to filter detections

## Model Information
- Uses YOLOv8n (nano) by default for fast inference
- Automatically downloads the model on first run
- Detects 80 different object classes (COCO dataset)

## Troubleshooting
- If you get import errors, make sure all dependencies are installed
- If the model fails to load, check your internet connection (model downloads on first run)
- If no images are found, ensure the 'images' folder contains supported image files
