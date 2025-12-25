# app.py
import streamlit as st
import face_recognition
from PIL import Image
import numpy as np
import os
import pandas as pd

st.title("حضور و غیاب با عکس کلاس")

STUDENTS_DIR = "students"

# بارگذاری embeddings دانش‌آموزان
student_names = []
student_encodings = []

for file in os.listdir(STUDENTS_DIR):
    if file.lower().endswith((".jpg", ".png")):
        name = os.path.splitext(file)[0]
        img_path = os.path.join(STUDENTS_DIR, file)
        img = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(img)
        if encoding:
            student_names.append(name)
            student_encodings.append(encoding[0])

uploaded_file = st.file_uploader("عکس کلاس را آپلود کن", type=["jpg", "png"])
if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="عکس کلاس")

    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    status = {name: "غایب" for name in student_names}

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(student_encodings, face_encoding, tolerance=0.4)
        for idx, match in enumerate(matches):
            if match:
                status[student_names[idx]] = "حاضر"

    # نمایش جدول حضور و غیاب
    df = pd.DataFrame(status.items(), columns=["نام دانش‌آموز", "وضعیت"])
    st.table(df)
