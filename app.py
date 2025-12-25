import streamlit as st
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import os

# ---------- Import صحیح MediaPipe 0.10+ ----------
from mediapipe.python.solutions import face_detection as mp_face_detection
# --------------------------------------------------

st.title("حضور و غیاب با عکس کلاس")

# بارگذاری اسامی دانش‌آموزان از فولدر students
students = [f.split(".")[0] for f in os.listdir("students")]
status = {name: "غایب" for name in students}

# آپلود عکس کلاس
uploaded = st.file_uploader("عکس کلاس را آپلود کن", type=["jpg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="عکس کلاس")
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # ---------- MediaPipe Face Detection ----------
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
    results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # -----------------------------------------------

    # به‌روزرسانی وضعیت دانش‌آموزان
    if results.detections:
        for i, det in enumerate(results.detections):
            if i < len(students):
                status[students[i]] = "حاضر"

    # نمایش دانش‌آموزان حاضر
    present_students = [name for name, stt in status.items() if stt == "حاضر"]
    if present_students:
        st.subheader("✅ دانش‌آموزان حاضر:")
        st.write(", ".join(present_students))
    else:
        st.subheader("هیچ دانش‌آموزی حاضر نیست.")

    # نمایش جدول وضعیت
    df = pd.DataFrame(status.items(), columns=["نام دانش‌آموز", "وضعیت"])
    st.table(df)
