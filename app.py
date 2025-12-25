import streamlit as st
import face_recognition
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import os

st.title("حضور و غیاب با عکس کلاس")

# بارگذاری دانش‌آموزان و عکس‌های آنها
students = [f.split(".")[0] for f in os.listdir("students")]
known_images = [face_recognition.load_image_file(f"students/{f}") for f in os.listdir("students")]
known_encodings = [face_recognition.face_encodings(img)[0] for img in known_images]

# وضعیت اولیه دانش‌آموزان
status = {name: "غایب" for name in students}

# آپلود عکس کلاس
uploaded = st.file_uploader("عکس کلاس را آپلود کن", type=["jpg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="عکس کلاس")
    img = np.array(image)

    # شناسایی چهره‌ها در عکس کلاس
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    # مقایسه با دانش‌آموزان
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        if True in matches:
            match_index = matches.index(True)
            status[students[match_index]] = "حاضر"

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
