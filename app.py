import streamlit as st
import cv2
import mediapipe as mp          # نسخه 0.9 یا 0.10 فرقی نمی‌کند
import pandas as pd
import numpy as np
from PIL import Image
import os

st.title("حضور و غیاب با عکس کلاس")

# لیست دانش‌آموزان
students = [f.split(".")[0] for f in os.listdir("students")]
status = {name: "غایب" for name in students}

uploaded = st.file_uploader("عکس کلاس را آپلود کن", type=["jpg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="عکس کلاس")
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # ---------- API قدیمی ----------
    mp_face = mp.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5)
    results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # --------------------------------

    if results.detections:
        for i, det in enumerate(results.detections):
            if i < len(students):
                status[students[i]] = "حاضر"

    present_students = [name for name, stt in status.items() if stt == "حاضر"]
    if present_students:
        st.subheader("✅ دانش‌آموزان حاضر:")
        st.write(", ".join(present_students))
    else:
        st.subheader("هیچ دانش‌آموزی حاضر نیست.")

    df = pd.DataFrame(status.items(), columns=["نام دانش‌آموز", "وضعیت"])
    st.table(df)
