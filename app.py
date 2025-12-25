import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from PIL import Image
import os


st.title("حضور و غیاب با عکس کلاس")

# لیست دانش‌آموزان
students = [f.split(".")[0] for f in os.listdir("students")]
status = {name: "غایب" for name in students}

# آپلود عکس کلاس
uploaded = st.file_uploader("عکس کلاس را آپلود کن", type=["jpg","png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="عکس کلاس")

    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

    results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.detections:
        for i, det in enumerate(results.detections):
            if i < len(students):
                status[students[i]] = "حاضر"

    df = pd.DataFrame(status.items(), columns=["نام دانش‌آموز", "وضعیت"])
    st.table(df)
