import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to apply Conservative Smoothing Filter
def conservative_smoothing(image, kernel_size=3):
    padded_img = np.pad(image, (kernel_size//2, kernel_size//2), mode='edge')
    new_img = np.copy(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighbors = padded_img[i:i+kernel_size, j:j+kernel_size]
            min_val, max_val = np.min(neighbors), np.max(neighbors)
            
            if image[i, j] < min_val:
                new_img[i, j] = min_val
            elif image[i, j] > max_val:
                new_img[i, j] = max_val
            else:
                new_img[i, j] = image[i, j]

    return new_img.astype(np.uint8)

# Function to apply Median Smoothing Filter
def median_smoothing(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

# Function to apply Mean Smoothing Filter
def mean_smoothing(image, kernel_size=3):
    return cv2.blur(image, (kernel_size, kernel_size))

# Function to apply Edge Detection (Extra Feature)
def edge_detection(image):
    return cv2.Canny(image, 50, 150)

# Streamlit GUI
st.title("Image Processing GUI")
st.write("Upload an image and apply a smoothing filter.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image.convert("L"))  # Convert to grayscale

    # Resize image for proper scaling
    image = cv2.resize(image, (400, 400))  

    # Store the selected filter in session state
    if "filter_option" not in st.session_state:
        st.session_state.filter_option = "Conservative Smoothing"

    # Select filter (with dynamic updating)
    filter_option = st.radio("Choose a filter:", 
                             ["Conservative Smoothing", "Median Smoothing", "Mean Smoothing", "Edge Detection"],
                             index=["Conservative Smoothing", "Median Smoothing", "Mean Smoothing", "Edge Detection"].index(st.session_state.filter_option))

    # Update session state
    st.session_state.filter_option = filter_option

    # Apply selected filter dynamically
    if filter_option == "Conservative Smoothing":
        filtered_image = conservative_smoothing(image)
    elif filter_option == "Median Smoothing":
        filtered_image = median_smoothing(image)
    elif filter_option == "Mean Smoothing":
        filtered_image = mean_smoothing(image)
    else:
        filtered_image = edge_detection(image)

    # Display side-by-side images
    col1, col2 = st.columns(2)
    col1.image(image, caption="Original Image", use_column_width=True)
    col2.image(filtered_image, caption="Filtered Image", use_column_width=True)

    # Save filtered image
    filtered_image_pil = Image.fromarray(filtered_image)
    filtered_image_pil.save("filtered_image.jpg")

    # Provide download link
    with open("filtered_image.jpg", "rb") as file:
        st.download_button(
            label="Download Filtered Image",
            data=file,
            file_name="filtered_image.jpg",
            mime="image/jpeg"
        )
