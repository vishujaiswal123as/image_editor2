import streamlit as st
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from rembg import remove
import os


# filter
def convert_image_colorspace(image, colorspace):
    return cv2.cvtColor(image, colorspace)

def image_filters():
    st.header('Upload Image for filtering Image')
    image_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        image = cv2.imdecode(np.frombuffer(
            image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        colorspaces = {0: 'COLOR_BGR2GRAY', 1: 'COLOR_BGR2RGB', 2: 'COLOR_BGR2HSV', 3: 'COLOR_BGR2LAB', 4: 'COLOR_BGR2XYZ', 5: 'COLOR_BGR2YCrCb', 6: 'COLOR_BGR2YUV',
                       7: 'COLOR_BGR2HLS', 8: 'COLOR_BGR2Luv', 9: 'COLOR_BGR2HLS_FULL', 10: 'COLOR_BGR2HSV_FULL', 11: 'COLOR_BGR2LAB', 12: 'COLOR_BGR2Luv', 13: 'COLOR_BGR2RGB', 14: 'COLOR_BGR2XYZ'}

        conversion_code = st.selectbox(
            "Select a color space conversion", list(colorspaces.keys()))

        if conversion_code is not None:
            colorspace = colorspaces[conversion_code]
            converted_image = convert_image_colorspace(
                image, getattr(cv2, colorspace))
            st.image(
                converted_image, caption=f"Converted image ({colorspaces[conversion_code]})")
            # if st.button('Tap For download'):
            rgb_converted_image = cv2.cvtColor(
                converted_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite('filtered.png', rgb_converted_image)
            with open("filtered.png", "rb") as file:
                btn = st.download_button(
                    label="Download image",
                    data=file,
                    file_name="filtered.png",
                    mime="image/png"
                )

####### editor
def sharpen_kernel(level):
    if level == 1:
        return np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    elif level == 2:
        return np.array([[-1, -1, -1],
                         [-1, 9, -1],
                         [-1, -1, -1]])
    elif level == 3:
        return np.array([[-1, -1, -1, -1, -1],
                         [-1, -1, -1, -1, -1],
                         [-1, -1, 25, -1, -1],
                         [-1, -1, -1, -1, -1],
                         [-1, -1, -1, -1, -1]])

def image_editer():
    st.header('Upload Image for edit')
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        image = np.array(Image.open(uploaded_file))

        # Display the original image

        # Add a slider to adjust the contrast
        contrast = st.slider('Contrast', -100, 100, 0)

        # Adjust the contrast of the image
        if contrast != 0:
            image = cv2.addWeighted(image, 1 + (contrast/100),
                                    np.zeros(image.shape, image.dtype), 0, 0)

        # Add a slider to adjust the brightness
        brightness = st.slider('Brightness', -100, 100, 0)

        # Adjust the brightness of the image
        image = cv2.addWeighted(image, 1, np.zeros(
            image.shape, image.dtype), 0, brightness)

        # Display the adjusted image

        col1, col2 = st.columns(2)
        col1.image(uploaded_file, use_column_width=True)
        col2.image(image, use_column_width=True)
        # Add a slider to adjust the sharpness
        sharpness = st.slider('Sharpness', 0, 3, 0)

        # Create a sharpening kernel
        if sharpness > 0:
            kernel = sharpen_kernel(sharpness)

            # Apply sharpening
            image = cv2.filter2D(image, -1, kernel)

        col1, col2 = st.columns(2)
        # Display the adjusted image
        col1.image(uploaded_file, use_column_width=True)
        col2.image(image, use_column_width=True)

        # Add a button to crop the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('image.png', image)
        with open("image.png", "rb") as file:
            btn = st.download_button(
                label="Download image",
                data=file,
                file_name="edited.png",
                mime="image/png"
            )

# remov background
def remove_background():
    st.header('Upload Your image for removing background')

    def remove_background_with_rembg(image_path, output_path):
        # Load the image
        input_image = Image.open(image_path)

        # Remove the background
        output_image = remove(input_image)

        # Save the output image
        output_image.save(output_path)
        return output_path
    user_image = st.file_uploader(
        'Upload Image here', type=['png', 'jpg', 'jpeg'])
    if user_image:
        st.subheader('Please wait a second')
        col1,col2=st.columns(2)
        col1.image(user_image)
        
        output_path = remove_background_with_rembg(user_image, "output.png")
        col2.image(output_path)
       
        if os.path.exists(output_path):
            st.write("Background removed successfully!")
            with open(output_path, "rb") as file:
                st.download_button(label="Download image", data=file,
                                   file_name="removed_background.png", mime="image/png")
        else:
            st.write("Error: The output image does not exist.")


sidebar=st.sidebar.selectbox('Menu',['Image Editor','Filters','Background Remover'])
if sidebar=='Image Editor':
   image_editer()
elif sidebar=='Filters':
  image_filters()
elif sidebar=='Background Remover':
   remove_background()
