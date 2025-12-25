import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os

st.title("حضور و غیاب با عکس کلاس")

# بارگذاری دانش‌آموزان از فولدر students
students = [f.split(".")[0] for f in os.listdir("students")]
status = {name: "غایب" for name in students}

# آپلود عکس کلاس
uploaded = st.file_uploader("عکس کلاس را آپلود کن", type=["jpg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="عکس کلاس")
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # ---------- تشخیص چهره با OpenCV ----------
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # علامت گذاری چهره‌ها روی تصویر
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="چهره‌های شناسایی شده")

    # بروزرسانی وضعیت دانش‌آموزان
    for i, name in enumerate(students):
        if i < len(faces):
            status[name] = "حاضر"

    # نمایش دانش‌آموزان حاضر
    df = pd.DataFrame(status.items(), columns=["نام دانش‌آموز", "وضعیت"])
    st.table(df)
