"""
Image Inspection Module - VGG16 + YOLOv5 Integration
Detects equipment defects and locates their positions using computer vision
Uses subprocess approach to match exact training configuration

Developer: Eng. Mahmoud Khalid Alkodousy
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path
import tempfile
import os
import subprocess
import shutil
from config import VGG_MODEL_PATH, YOLO_WEIGHTS_PATH, YOLOV5_DIR


# ============================================
# MODEL LOADING FUNCTIONS
# ============================================

@st.cache_resource
def load_vgg_model():
    """
    Load pre-trained VGG16 binary classification model
    
    Model classifies images as:
    - Defected (probability < 0.5)
    - Non-Defected (probability >= 0.5)
    
    Returns:
        tf.keras.Model or None: Loaded model or None if loading fails
    """
    try:
        # Verify model file exists
        if not VGG_MODEL_PATH.exists():
            st.error(f"❌ VGG model not found at: {VGG_MODEL_PATH}")
            return None
        
        # Load model without compilation (we'll compile manually)
        model = tf.keras.models.load_model(str(VGG_MODEL_PATH), compile=False)
        
        # Compile with same settings as training
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except Exception as e:
        st.error(f"❌ Failed to load VGG16 model: {e}")
        return None


def check_yolo_setup():
    """
    Verify YOLOv5 installation and model weights availability
    
    Checks for:
    1. YOLOv5 directory existence
    2. Trained weights file
    3. detect.py script (required for inference)
    
    Returns:
        dict: Status dictionary with paths and availability flags
    """
    detect_script = YOLOV5_DIR / "detect.py"
    
    yolo_exists = YOLOV5_DIR.exists()
    weights_exists = YOLO_WEIGHTS_PATH.exists()
    detect_exists = detect_script.exists()
    
    return {
        'ready': yolo_exists and weights_exists and detect_exists,
        'yolov5_path': str(YOLOV5_DIR),
        'weights_path': str(YOLO_WEIGHTS_PATH),
        'detect_script': str(detect_script),
        'yolo_exists': yolo_exists,
        'weights_exists': weights_exists,
        'detect_exists': detect_exists
    }


# ============================================
# DEFECT CLASSIFICATION (VGG16)
# ============================================

def classify_defect(vgg_model, image, threshold=0.5):
    """
    Classify image as Defected or Non-Defected using VGG16
    
    Process:
    1. Resize image to 224x224 (VGG16 input size)
    2. Normalize pixel values to [0, 1]
    3. Get prediction probability
    4. Apply threshold (0.5 by default)
    
    Args:
        vgg_model: Loaded VGG16 model
        image (PIL.Image): Input image
        threshold (float): Classification threshold (default 0.5)
        
    Returns:
        dict: Classification results with label, confidence, and probabilities
    """
    # Preprocess image to match training
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Get prediction
    pred = vgg_model.predict(img_array, verbose=0)
    prob = float(pred[0][0])
    
    # Interpret probability (prob >= 0.5 → Non-Defected)
    if prob >= threshold:
        label = 'Non-Defected'
        confidence = prob
    else:
        label = 'Defected'
        confidence = 1 - prob
    
    return {
        'label': label,
        'confidence': confidence,
        'probability': prob,
        'is_defected': label == 'Defected'
    }


# ============================================
# DEFECT LOCALIZATION (YOLOv5)
# ============================================

def locate_defect_yolo_subprocess(image, yolo_config):
    """
    Locate defects using YOLOv5 via subprocess (matches exact training config)
    
    Uses subprocess to call detect.py with exact same parameters as training:
    - Image size: 416x416
    - Confidence threshold: 0.1
    - IoU threshold: 0.45
    
    This ensures consistency between training and inference.
    
    Args:
        image (PIL.Image): Input image
        yolo_config (dict): Configuration dictionary with YOLO paths
    
    Returns:
        dict: Detection results with defect count, bounding boxes, and annotated image
    """
    
    # Create temporary directory for this detection session
    temp_dir = tempfile.mkdtemp()
    temp_image_path = os.path.join(temp_dir, "temp_image.jpg")
    
    try:
        # Save image to temporary location
        image.save(temp_image_path)
        
        # Display detection configuration
        st.write(f"🔍 **YOLO Detection Settings:**")
        st.write(f"- Image size: 416")
        st.write(f"- Confidence threshold: 0.1")
        st.write(f"- IoU threshold: 0.45")
        st.write(f"- Using detect.py subprocess")
        
        # ============================================
        # CRITICAL: Build subprocess command with exact training parameters
        # ============================================
        cmd = [
            'python',
            yolo_config['detect_script'],
            '--weights', yolo_config['weights_path'],
            '--img', '416',                    # Image size (must match training)
            '--conf', '0.1',                   # Confidence threshold
            '--iou-thres', '0.45',             # IoU threshold for NMS
            '--source', temp_image_path,       # Input image path
            '--project', temp_dir,             # Output directory
            '--name', 'results',               # Results folder name
            '--exist-ok',                      # Overwrite if exists
            '--save-txt',                      # Save detection coordinates
            '--save-conf'                      # Save confidence scores
        ]
        
        # Execute YOLO detection
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=yolo_config['yolov5_path']  # Run from YOLOv5 directory
        )
        
        # Check for execution errors
        if result.returncode != 0:
            st.error(f"❌ YOLO detection failed!")
            st.code(result.stderr)
            return {
                'defect_count': 0,
                'defects': [],
                'annotated_image': image,
                'error': result.stderr
            }
        
        # ============================================
        # Parse detection results
        # ============================================
        
        # Locate output files
        output_dir = os.path.join(temp_dir, 'results')
        output_image_path = os.path.join(output_dir, "temp_image.jpg")
        labels_dir = os.path.join(output_dir, 'labels')
        label_file = os.path.join(labels_dir, "temp_image.txt")
        
        defects = []
        
        # Read detection labels if file exists
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Get original image dimensions for coordinate conversion
            img_width, img_height = image.size
            
            # Parse each detection line
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO format: class_id x_center y_center width height [confidence]
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    confidence = float(parts[5]) if len(parts) > 5 else 0.0
                    
                    # Convert from center format to corner coordinates
                    x1 = int(x_center - width/2)
                    y1 = int(y_center - height/2)
                    x2 = int(x_center + width/2)
                    y2 = int(y_center + height/2)
                    
                    defects.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence
                    })
        
        # Display detection statistics
        st.write(f"- Total detections: {len(defects)}")
        
        if len(defects) > 0:
            for idx, defect in enumerate(defects, 1):
                st.write(f"  - Detection {idx}: conf={defect['confidence']:.3f}, box={defect['bbox']}")
        
        # Load annotated image (with bounding boxes drawn)
        annotated_image = image
        if os.path.exists(output_image_path):
            annotated_image = Image.open(output_image_path)
        
        return {
            'defect_count': len(defects),
            'defects': defects,
            'annotated_image': annotated_image
        }
    
    except Exception as e:
        st.error(f"⚠️ Error in YOLO detection: {e}")
        import traceback
        st.code(traceback.format_exc())
        
        return {
            'defect_count': 0,
            'defects': [],
            'annotated_image': image,
            'error': str(e)
        }
    
    finally:
        # Cleanup temporary files
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


# ============================================
# SINGLE IMAGE INSPECTION UI
# ============================================

def render_image_inspection_tab():
    """
    Render the single image inspection interface
    
    Workflow:
    1. Upload image
    2. Classify with VGG16 (Defected/Non-Defected)
    3. If defected, locate defects with YOLO
    4. Display results and recommendations
    """
    
    st.header("🔍 Equipment Image Inspection")
    st.markdown("Upload an equipment image to detect defects and locate their positions")
    
    # Initialize session state for file uploader key (allows reset)
    if 'uploaded_file_key' not in st.session_state:
        st.session_state.uploaded_file_key = 0
    
    # ============================================
    # Load models and check status
    # ============================================
    
    with st.spinner("Loading VGG16 model..."):
        vgg_model = load_vgg_model()
    
    yolo_config = check_yolo_setup()
    
    # Display model availability status
    col1, col2 = st.columns(2)
    with col1:
        if vgg_model is not None:
            st.success("✅ VGG16 loaded")
        else:
            st.error("❌ VGG16 failed")
    
    with col2:
        if yolo_config['ready']:
            st.success("✅ YOLO ready")
        else:
            st.warning("⚠️ YOLO unavailable")
            
            # Show specific missing components
            if not yolo_config['yolo_exists']:
                st.error(f"❌ YOLOv5 folder not found at: {yolo_config['yolov5_path']}")
            if not yolo_config['weights_exists']:
                st.error(f"❌ Weights not found at: {yolo_config['weights_path']}")
            if not yolo_config['detect_exists']:
                st.error(f"❌ detect.py not found at: {yolo_config['detect_script']}")
    
    # Stop if VGG16 is not available
    if vgg_model is None:
        st.error("❌ VGG16 model not loaded. Cannot proceed.")
        return
    
    st.markdown("---")
    
    # ============================================
    # Image upload and inspection
    # ============================================
    
    uploaded_file = st.file_uploader(
        "Choose an equipment image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the equipment",
        key=f"uploader_{st.session_state.uploaded_file_key}"
    )
    
    if uploaded_file is not None:
        # Load and display uploaded image
        image = Image.open(uploaded_file)
        
        st.subheader("📸 Original Image")
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
        
        # Inspection trigger options
        col1, col2 = st.columns([3, 1])
        
        with col1:
            auto_inspect = st.checkbox(
                "🔄 Auto-inspect on upload", 
                value=True, 
                help="Automatically analyze image when uploaded"
            )
        
        with col2:
            manual_button = st.button(
                "🔍 Inspect Now", 
                type="primary", 
                use_container_width=True
            )
        
        # Determine if inspection should run
        should_inspect = auto_inspect or manual_button
        
        if should_inspect:
            
            # ============================================
            # STEP 1: VGG16 Classification
            # ============================================
            
            with st.spinner("🤖 Analyzing image with VGG16..."):
                classification_result = classify_defect(vgg_model, image)
            
            st.markdown("---")
            st.subheader("📊 Classification Result")
            
            # Display classification metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if classification_result['is_defected']:
                    st.error(f"### 🔴 {classification_result['label']}")
                else:
                    st.success(f"### ✅ {classification_result['label']}")
            
            with col2:
                st.metric(
                    "Confidence",
                    f"{classification_result['confidence']:.1%}"
                )
            
            with col3:
                st.metric(
                    "Raw Probability",
                    f"{classification_result['probability']:.4f}",
                    help="VGG16 output (≥0.5 = Non-Defected)"
                )
            
            # ============================================
            # STEP 2: YOLO Localization (if defected)
            # ============================================
            
            if classification_result['is_defected']:
                st.markdown("---")
                st.subheader("🎯 Defect Localization")
                
                if yolo_config['ready']:
                    with st.spinner("🔍 Running YOLO detection..."):
                        yolo_result = locate_defect_yolo_subprocess(image, yolo_config)
                    
                    # Case A: Defects found
                    if yolo_result['defect_count'] > 0:
                        st.warning(f"⚠️ Found **{yolo_result['defect_count']}** defect(s)")
                        
                        # Display annotated image with bounding boxes
                        st.image(
                            yolo_result['annotated_image'],
                            caption="Defects Located (YOLO Detection)",
                            use_container_width=True
                        )
                        
                        # Show detailed defect information
                        with st.expander("📋 Detailed Defect Information"):
                            for i, defect in enumerate(yolo_result['defects'], 1):
                                st.write(f"**Defect {i}:**")
                                st.write(f"- Position (x1, y1, x2, y2): {defect['bbox']}")
                                st.write(f"- Confidence: {defect['confidence']:.2%}")
                                st.write("---")
                        
                        # Recommendations
                        st.error("""
                        ### 🚨 Recommended Actions:
                        - Schedule immediate inspection
                        - Review maintenance history
                        - Consider equipment replacement if multiple defects
                        - Document defect location for maintenance team
                        """)
                    
                    # Case B: No defects detected by YOLO
                    else:
                        st.info("ℹ️ VGG classified as defected but YOLO detected no specific regions")
                        st.warning("""
                        **Possible reasons:**
                        - Overall wear or degradation (not localized)
                        - Very small defects below detection threshold
                        - Defect type not in YOLO training data
                        - Low confidence detections filtered out
                        
                        **Recommendation:** Manual inspection recommended
                        """)
                
                # Case C: YOLO not available
                else:
                    st.warning("⚠️ Defect detected but YOLO model unavailable")
                    st.info("""
                    **VGG Classification:** Defect detected
                    
                    **Recommendations:**
                    - Schedule manual inspection
                    - Review equipment history
                    - Set up YOLO for precise defect location
                    """)
            
            # ============================================
            # Non-Defected Case
            # ============================================
            
            else:
                st.markdown("---")
                st.success("""
                ### ✅ Equipment Status: Normal
                
                No defects detected. Equipment appears to be in good condition.
                
                **Recommendations:**
                - Continue regular maintenance schedule
                - Re-inspect after standard operating period
                - Keep equipment monitoring logs updated
                """)
            
            # Upload another image button
            st.markdown("---")
            if st.button("📤 Upload Another Image", use_container_width=True):
                st.session_state.uploaded_file_key += 1
                st.rerun()


# ============================================
# BATCH INSPECTION UI
# ============================================

def render_batch_inspection_tab():
    """
    Render batch processing interface for multiple images
    
    Features:
    - Upload multiple images at once
    - Process all images automatically
    - Display summary statistics
    - Export results as CSV
    """
    
    st.header("📦 Batch Image Inspection")
    st.info("Upload multiple images for batch processing")
    
    # Multi-file uploader
    uploaded_files = st.file_uploader(
        "Choose multiple equipment images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"📁 Uploaded **{len(uploaded_files)}** images")
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            include_yolo = st.checkbox("Include YOLO defect count", value=True)
        with col2:
            show_preview = st.checkbox("Show image previews", value=False)
        
        if st.button("🔍 Inspect All Images", type="primary", use_container_width=True):
            
            # Load models
            vgg_model = load_vgg_model()
            yolo_config = check_yolo_setup()
            
            if vgg_model is None:
                st.error("❌ VGG16 model not loaded")
                return
            
            # Process each image
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {file.name}")
                
                image = Image.open(file)
                
                # Show preview if enabled
                if show_preview:
                    with st.expander(f"Preview: {file.name}"):
                        st.image(image, use_container_width=True)
                
                # Classify with VGG16
                class_result = classify_defect(vgg_model, image)
                
                result = {
                    'filename': file.name,
                    'label': class_result['label'],
                    'confidence': class_result['confidence'],
                    'defect_count': 0
                }
                
                # Count defects with YOLO if defected
                if include_yolo and class_result['is_defected'] and yolo_config['ready']:
                    yolo_result = locate_defect_yolo_subprocess(image, yolo_config)
                    result['defect_count'] = yolo_result['defect_count']
                
                results.append(result)
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ Processing complete!")
            
            # ============================================
            # Display results
            # ============================================
            
            import pandas as pd
            results_df = pd.DataFrame(results)
            
            st.markdown("---")
            st.subheader("📊 Batch Inspection Results")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            defected_count = (results_df['label'] == 'Defected').sum()
            total_defects = results_df['defect_count'].sum() if include_yolo else 0
            
            with col1:
                st.metric("Total Images", len(results_df))
            with col2:
                st.metric("Defected", defected_count, delta=f"{defected_count/len(results_df)*100:.1f}%")
            with col3:
                if include_yolo:
                    st.metric("Total Defects Found", int(total_defects))
            
            # Results table with conditional formatting
            st.dataframe(
                results_df.style.applymap(
                    lambda x: 'background-color: #ffcccc' if x == 'Defected' else '',
                    subset=['label']
                ),
                use_container_width=True
            )
            
            # CSV download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="batch_inspection_results.csv",
                mime="text/csv",
                use_container_width=True
            )


# ============================================
# MAIN ENTRY POINT
# ============================================

def main():
    """
    Main function for standalone testing
    Allows running this module independently
    """
    st.set_page_config(
        page_title="Image Inspection",
        page_icon="🔍",
        layout="wide"
    )
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose Mode", ["Single Image", "Batch Processing"])
    
    if page == "Single Image":
        render_image_inspection_tab()
    else:
        render_batch_inspection_tab()


if __name__ == "__main__":
    main()