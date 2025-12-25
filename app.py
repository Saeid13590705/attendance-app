import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
from PIL import Image
import os

st.title("حضور و غیاب با عکس کلاس")

# -----------------------------
# 1. بارگذاری دانش‌آموزان و استخراج embedding
# -----------------------------
students_dir = "students"
student_images = []
student_names = []

for filename in os.listdir(students_dir):
    if filename.lower().endswith((".jpg", ".png")):
        path = os.path.join(students_dir, filename)
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)
        if encoding:  # مطمئن شویم چهره شناسایی شده
            student_images.append(encoding[0])
            student_names.append(os.path.splitext(filename)[0])

status = {name: "غایب" for name in student_names}

# -----------------------------
# 2. آپلود عکس کلاس
# -----------------------------
uploaded = st.file_uploader("عکس کلاس را آپلود کن", type=["jpg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="عکس کلاس")
    img = np.array(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # -----------------------------
    # 3. شناسایی چهره‌ها در عکس کلاس
    # -----------------------------
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    # -----------------------------
    # 4. تطبیق چهره‌ها با دانش‌آموزان
    # -----------------------------
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(student_images, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(student_images, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = student_names[best_match_index]
            status[name] = "حاضر"
            # رسم نام روی تصویر
            cv2.rectangle(img_rgb, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img_rgb, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # چهره ناشناس
            cv2.rectangle(img_rgb, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(img_rgb, "Unknown", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # نمایش تصویر با نام‌ها
    st.image(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB), caption="نتیجه شناسایی")

    # -----------------------------
    # 5. نمایش جدول حضور
    # -----------------------------
    df = pd.DataFrame(status.items(), columns=["نام دانش‌آموز", "وضعیت"])
    st.table(df)
