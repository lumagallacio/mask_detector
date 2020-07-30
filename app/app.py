# Core Pkgs
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
from  tensorflow.keras.models import load_model
from mtcnn.mtcnn import MTCNN
import gc

def detect_mask(img):
    # new_img = np.array(our_image.convert("RGB"))
    # img = cv2.cvtColor(new_img, 1)
    detector = MTCNN()
    faces = detector.detect_faces(img)
    del detector

    if faces:
        labels=[]
        for face in faces:
            x1, y1, w, h = face["box"]
            x2, y2 = x1 + w, y1 + h

            roi = img[y1:y2, x1:x2]

            if np.sum([roi]) != 0:

                roi = cv2.resize(roi, (160, 160))
                roi = (roi[..., ::-1].astype(np.float32)) / 255.0

                # PREDIÇÃO
                model = load_model("detector.h5")
                pred = model.predict(np.array([roi]))
                del model

                pred = pred[0]  ## pegando o vetor interno da classificação
                print("pred")
                print(pred)
                label = "NO MASK"

                if pred[0] >= pred[1]:
                    label = "NO MASK - " + str(np.round(pred[0], 2))
                    color = (0, 0, 255)
                else:
                    label = "MASK - " + str(np.round(pred[1], 2))
                    color = (0, 255, 0) 
                
                labels.append(label)           
                
                label_position = (x1, y1)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3
                )
        return img, labels

    st.error("Oops!  didn't find a face in the image.  Try again...")
    return False, False


def set_enhance(our_image):
    enhance_type = st.sidebar.radio(
        "Enhance Type", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"]
    )

    if enhance_type == "Gray-Scale":
        new_img = np.array(our_image.convert("RGB"))
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        new_img = cv2.cvtColor(new_img, 1)

    elif enhance_type == "Contrast":
        c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
        enhancer = ImageEnhance.Contrast(our_image)
        new_img = enhancer.enhance(c_rate)
        new_img = np.array(new_img.convert("RGB"))
        new_img = cv2.cvtColor(new_img, 1)

    elif enhance_type == "Brightness":
        c_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
        enhancer = ImageEnhance.Brightness(our_image)
        new_img = enhancer.enhance(c_rate)
        new_img = np.array(new_img.convert("RGB"))
        new_img = cv2.cvtColor(new_img, 1)

    # elif enhance_type == "Blurring":
    #     new_img = np.array(our_image.convert("RGB"))
    #     blur_rate = st.sidebar.slider("Blurring", 0.5, 3.5)
    #     img = cv2.cvtColor(new_img, 1)
    #     new_img = cv2.GaussianBlur(img, (11, 11), blur_rate)
    else:
        new_img = np.array(our_image.convert("RGB"))
        new_img = cv2.cvtColor(new_img, 1)

    return new_img

def show_result(result_img, result_labels):
    st.image(result_img, width=400)
    st.success("Found {} faces".format(len(result_labels)))
    for label in result_labels:
        st.success(str(label))
   

def set_detection(our_image):

    if st.sidebar.button("Apply Mask Detection"):
        try:
            result_img, result_labels = detect_mask(our_image)
            show_result(result_img, result_labels)

        except TypeError:
            print('image error - didnt find face')

def main():
    st.title("Mask Detection")
    st.text("Build with Streamlit, OpenCV and TensorFlow")

    activities = ["Detection", "About"]
    choice = st.sidebar.selectbox("Select Activty", activities)

    if choice == "Detection":

        image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        if image_file is not None:
            our_image = Image.open(image_file)
            del image_file
            st.subheader("Our Image")
            # st.image(our_image, width=200)

            new_image = set_enhance(our_image)
            del our_image
            if new_image is not None:
                st.image(new_image, width=200)
                set_detection(new_image)
        gc.collect()

    elif choice == "About":
        st.subheader("About Maks Detection App")
        st.markdown(
            "Built by [Luma](https://www.linkedin.com/in/luma-gallacio-4a90b4aa/)"
        )
        st.markdown(
            "Source code [here](https://github.com/lumagallacio/mask_detector)"
        )
        st.text("Luma Gallacio")


if __name__ == "__main__":
    main()

