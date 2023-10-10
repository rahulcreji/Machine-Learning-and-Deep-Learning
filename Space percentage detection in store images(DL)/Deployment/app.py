import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
from Percentage import Calculate_Space

# Load YOLOv5 model
model = YOLO("best.pt")

# Define the sidebar
buff1, col1, buff2, buff3 = st.sidebar.columns([1, 1, 1, 1])

st.sidebar.markdown("<h1 style='text-align: center; color: red;'>Project On-Shelf Availability</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h1 style='text-align: center; color: blue;'>Team A-Data Science</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h1 style='text-align: center; color: blue;'>Beinex</h1>", unsafe_allow_html=True)

# Main content
st.markdown("<h1 style='text-align: center; color: #ff6347;'>Space Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload an image for space detection:</p>", unsafe_allow_html=True)

# Load and display image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Object detection
    if st.button('Detect Space'):
        # Convert the PIL image to NumPy array
        image_array = np.array(image)
        
        # Convert the image to BGR format (ultralytics format)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Perform object detection
        results = model.predict([image_bgr], conf=0.33, iou=0.3)
        result_image = results[0].plot()[:, :, ::-1]  # Convert to RGB format
        st.image(result_image, caption="Detected Objects.", use_column_width=True)
        
        # Calculate available space percentage
        available_space_percentage = Calculate_Space(results)
        
        # Categorize the available space based on your classification criteria
        if available_space_percentage > 20:
            label = "<span style='color: red;'>Urgent Fill Required</span>"
        elif 10 < available_space_percentage <= 20:
            label = "<span style='color: orange;'>Needs Attention</span>"
        else:
            label = "<span style='color: green;'>Managed Inventory</span>"
        
        st.markdown(
            f"<p style='text-align: center; font-size: 35px;'>{label}<br>Available Space: <span style='font-size: 28px;'>{available_space_percentage:.2f}%</span></p>",
            unsafe_allow_html=True
        )
else:
    st.text("Please upload an image.")