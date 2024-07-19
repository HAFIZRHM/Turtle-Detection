import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
import os
import cv2

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()
    return model

model = load_model()

def make_prediction(img):
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img)
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], labels=prediction["labels"],
                                          colors=["red" if label=="dog" else "green" for label in prediction["labels"]], width=2)
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)
    return img_with_bboxes_np

def send_email(subject, message, from_addr, to_addr, password, img_path):
    msg = MIMEMultipart()
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Subject'] = subject

    body = message
    msg.attach(MIMEText(body, 'plain'))

    img_data = open(img_path, 'rb').read()
    image = MIMEImage(img_data, name=os.path.basename(img_path))
    msg.attach(image)

    try:
        server = smtplib.SMTP('smtp.naver.com', 587)
        server.starttls()
        server.login(from_addr, password)
        text = msg.as_string()
        server.sendmail(from_addr, to_addr, text)
        server.quit()
        print("Email sent successfully!")
    except smtplib.SMTPAuthenticationError:
        print("Error: Unable to authenticate with the SMTP server.")
    except smtplib.SMTPException as e:
        print(f"Error: {e}")
    except TimeoutError:
        print("Error: Connection timed out. Please try again.")
    except Exception as e:
        print(f"Error: {e}")

def detect_dog_in_frame(frame):
    img = Image.fromarray(frame)
    prediction = make_prediction(img)
    img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2, 0, 1), prediction)
    dog_detected = any(label == "dog" for label in prediction["labels"])
    return img_with_bbox, dog_detected

st.title("Dog Detector")

use_webcam = st.checkbox("Use Webcam")

if use_webcam:
    st.write("Webcam feed will be displayed below. Please wait for the camera to start.")
    camera = cv2.VideoCapture(0)
    st_frame = st.empty()
    while camera.isOpened():
        ret, frame = camera.read()
        if ret:
            img_with_bbox, dog_detected = detect_dog_in_frame(frame)
            st_frame.image(frame, channels = "BGR")

        else:
            break

        if dog_detected:
            st.image(img_with_bbox, channels="BGR")
            st.header("Dog Detected!")
            st.write("A dog has been detected in the image.")
            cv2.imwrite('image_with_bboxes.png', cv2.cvtColor(img_with_bbox, cv2.COLOR_RGB2BGR))
            image_rgb = cv2.imread('image_with_bboxes.png', cv2.IMREAD_UNCHANGED)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite('image_with_bboxes.png', image_bgr)
            send_email("Dog Detection Alert", "A dog has been detected in the image. Please find the image attached.", "send_email", "eceive_email", "send_email_password", 'image_with_bboxes.png')
            break

    camera.release()
    cv2.destroyAllWindows()

else:
    upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

    if upload:
        img = Image.open(upload)

        prediction = make_prediction(img)
        img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2, 0, 1), prediction)

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        plt.imshow(img_with_bbox)
        plt.xticks([], [])
        plt.yticks([], [])
        ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

        st.image(img_with_bbox)

        dog_detected = any(label == "dog" for label in prediction["labels"])

        if dog_detected:
            st.header("Dog Detected!")
            st.write("A dog has been detected in the image.")
        else:
            st.header("No Dog Detected")
            st.write("No dog has been detected in the image.")

        # Save the image with bounding boxes to a file
        plt.savefig('image_with_bboxes.png')

        # Send the email
        if dog_detected:
            send_email("Dog Detection Alert", "A dog has been detected in the image. Please find the image attached.", "send_email", "eceive_email", "send_email_password", 'image_with_bboxes.png')
