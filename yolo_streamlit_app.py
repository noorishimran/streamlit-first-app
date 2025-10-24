import streamlit as st
import os
import random
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Page configuration
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🔍",
    layout="wide"
)

def load_yolo_model():
    """Load YOLOv8 model"""
    try:
        model = YOLO('yolov8n.pt')  # Load nano version for faster inference
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

def get_random_image():
    """Get a random image from the 'images' folder"""
    images_folder = "images"
    
    if not os.path.exists(images_folder):
        st.error(f"Images folder '{images_folder}' not found!")
        return None, None
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_files = [f for f in os.listdir(images_folder) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        st.error(f"No images found in '{images_folder}' folder!")
        return None, None
    
    # Select random image
    random_image = random.choice(image_files)
    image_path = os.path.join(images_folder, random_image)
    
    return image_path, random_image

def detect_objects(model, image_path):
    """Run YOLOv8 object detection on the image"""
    try:
        # Run inference
        results = model(image_path)
        result = results[0]
        
        # Get detection information
        boxes = result.boxes
        if boxes is not None:
            detections = []
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence),
                    'class_name': class_name
                })
            
            return detections, result
        else:
            return [], result
            
    except Exception as e:
        st.error(f"Error during detection: {e}")
        return [], None

def draw_detections(image, detections):
    """Draw bounding boxes and labels on the image"""
    image_with_boxes = image.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        
        # Draw bounding box
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label with confidence
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Draw label background
        cv2.rectangle(image_with_boxes, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(image_with_boxes, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return image_with_boxes

def main():
    st.title("🔍 YOLOv8 Object Detection App")
    st.markdown("This app randomly selects an image from the 'images' folder and runs YOLOv8 object detection on it.")
    
    # Sidebar for configuration
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05
    )
    
    # Load model
    with st.spinner("Loading YOLOv8 model..."):
        model = load_yolo_model()
    
    if model is None:
        st.error("Failed to load YOLOv8 model. Please check your installation.")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Random Image")
        
        if st.button("🎲 Pick Random Image", type="primary"):
            image_path, image_name = get_random_image()
            
            if image_path:
                # Store in session state
                st.session_state.image_path = image_path
                st.session_state.image_name = image_name
                st.success(f"Selected: {image_name}")
            else:
                st.error("No images found!")
    
    with col2:
        st.subheader("🔍 Detection Results")
        
        # Process image if one is selected
        if hasattr(st.session_state, 'image_path') and st.session_state.image_path:
            # Load original image
            original_image = cv2.imread(st.session_state.image_path)
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Run detection
            with st.spinner("Running object detection..."):
                detections, result = detect_objects(model, st.session_state.image_path)
            
            # Filter detections by confidence threshold
            filtered_detections = [d for d in detections if d['confidence'] >= confidence_threshold]
            
            # Display results
            if filtered_detections:
                st.success(f"Found {len(filtered_detections)} objects!")
                
                # Draw detections on image
                image_with_detections = draw_detections(original_image_rgb, filtered_detections)
                
                # Display images
                st.image(original_image_rgb, caption="Original Image", use_column_width=True)
                st.image(image_with_detections, caption="Detection Results", use_column_width=True)
                
                # Display detection details
                st.subheader("📊 Detection Details")
                for i, detection in enumerate(filtered_detections):
                    st.write(f"**Object {i+1}:** {detection['class_name']} (Confidence: {detection['confidence']:.3f})")
                
            else:
                st.warning("No objects detected above the confidence threshold!")
                st.image(original_image_rgb, caption="Original Image", use_column_width=True)
    
    # Instructions
    st.markdown("---")
    st.markdown("### 📁 Setup Instructions")
    st.markdown("""
    1. Create a folder named 'images' in the same directory as this script
    2. Add some images (.jpg, .png, .jpeg, .bmp, .tiff, .webp) to the 'images' folder
    3. Click 'Pick Random Image' to select and analyze an image
    4. Adjust the confidence threshold in the sidebar to filter detections
    """)

if __name__ == "__main__":
    main()
