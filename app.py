import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageFilter
import subprocess
import os

st.set_page_config(page_title="C++ AI Engine", layout="centered")

st.title("Bare-Metal C++ Inference Engine")
st.write("Draw a digit (0-9) below. Python will format it, and C++ will run the math.")

canvas_result = st_canvas(
    fill_color="black",
    background_color="black",
    stroke_color="white",
    stroke_width=20, 
    update_streamlit=True,
    height=280,         
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict with C++"):
    if canvas_result.image_data is not None:

        img_array = canvas_result.image_data

        img = Image.fromarray(img_array.astype('uint8'), 'RGBA')
        img = img.convert('L') 
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
# --- THE MNIST PREPROCESSING ALGORITHM ---
        
        # A. Find the exact bounding box of the ink
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
            
        # B. Shrink it so the longest edge is exactly 20 pixels
        img.thumbnail((20, 20), Image.Resampling.LANCZOS)
        
        # C. Paste it into the dead center of a 28x28 black canvas
        final_img = Image.new('L', (28, 28), color=0)
        paste_x = (28 - img.width) // 2
        paste_y = (28 - img.height) // 2
        final_img.paste(img, (paste_x, paste_y))
        
        # D. Add a tiny blur to simulate the scanner bleeding
        final_img = final_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # 4. Flatten to 1D array and normalize
        pixel_array = np.array(final_img).flatten()
        pixel_array = pixel_array.astype('float32') / 255.0
        
        txt_filepath = "custom_digit.txt"
        with open(txt_filepath, "w") as f:

            f.write(" ".join(map(str, pixel_array)))
            

        with st.spinner("C++ is crunching the matrices..."):
            try:
                # Call ./ai custom_digit.txt and capture the terminal output
                result = subprocess.run(["./ai", txt_filepath], capture_output=True, text=True)
                output = result.stdout
                
                # Parse the output to find our "PREDICTION:X" string
                prediction = "Unknown"
                for line in output.split('\n'):
                    if "PREDICTION:" in line:
                        prediction = line.split(":")[1]
                
                st.success(f"### The C++ Engine Predicts: {prediction}")
                
                # ---> ADD THIS: Print the trapped C++ terminal output to the web app <---
                with st.expander("See raw C++ Engine Output (ASCII Art)"):
                    st.code(output, language="text")
                
            except Exception as e:
                st.error(f"Failed to run C++ engine: {e}")