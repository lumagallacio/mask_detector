# Core Pkgs
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from scipy import ndimage, misc

detector = MTCNN()
model = load_model("detector.h5")


def detect_mask(our_image):
    new_img = np.array(our_image.convert("RGB"))
    img = cv2.cvtColor(new_img, 1)

    faces = detector.detect_faces(img)

    for face in faces:
        x1, y1, w, h = face["box"]
        # x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + w, y1 + h

        roi = img[y1:y2, x1:x2]

        if np.sum([roi]) != 0:

            roi = cv2.resize(roi, (160, 160))
            roi = (roi[..., ::-1].astype(np.float32)) / 255.0

            # PREDIÇÃO
            pred = model.predict(np.array([roi]))

            pred = pred[0]  ## pegando o vetor interno da classificação
            print("pred")
            print(pred)
            label = "NO MASK"

            if pred[0] >= pred[1]:
                label = "NO MASK"
                color = (0, 0, 255)
            else:
                label = "MASK"
                color = (0, 255, 0)

            # label_position = (x1-100, y1+250)
            label_position = (x1, y1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
            return img, faces

    # else:
    # 		cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
    return []


def set_enhance(our_image):
    enhance_type = st.sidebar.radio(
        "Enhance Type", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"]
    )
    new_image = []
    print(enhance_type)
    if enhance_type == "Gray-Scale":
        new_img = np.array(our_image.convert("RGB"))
        img = cv2.cvtColor(new_img, 1)
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif enhance_type == "Contrast":
        c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
        enhancer = ImageEnhance.Contrast(our_image)
        img_output = enhancer.enhance(c_rate)
        st.image(img_output)

    elif enhance_type == "Brightness":
        c_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
        enhancer = ImageEnhance.Brightness(our_image)
        img_output = enhancer.enhance(c_rate)
        st.image(img_output)

    elif enhance_type == "Blurring":
        new_img = np.array(our_image.convert("RGB"))
        blur_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
        img = cv2.cvtColor(new_img, 1)
        blur_img = cv2.GaussianBlur(img, (11, 11), blur_rate)
        st.image(blur_img)
    else:
        return None
    return new_img


def set_detection(our_image):

    if st.sidebar.button("Apply Mask Detection"):
        result_img, result_faces = detect_mask(our_image)
        st.image(result_img)
        st.success("Found {} faces".format(len(result_faces)))


def main():
    st.title("Mask Detection")
    st.text("Build with Streamlit and OpenCV")

    activities = ["Detection", "About"]
    choice = st.sidebar.selectbox("Select Activty", activities)

    if choice == "Detection":

        image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.subheader("Original Image")
            st.image(our_image, width=400)

            new_image = set_enhance(our_image)
            if new_image is not None:
                st.image(new_image, width=400)

            set_detection(new_image)

    elif choice == "About":
        st.subheader("About Maks Detection App")
        st.markdown(
            "Built with Streamlit by [Luma](https://www.linkedin.com/in/luma-gallacio-4a90b4aa/)"
        )
        st.text("Luma Gallacio")


if __name__ == "__main__":
    main()

