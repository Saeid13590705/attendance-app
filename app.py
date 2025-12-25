import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image
import os

st.title("حضور و غیاب با تطبیق چهره")

# بارگذاری دانش‌آموزان و استخراج embeddings اولیه
students = [f.split(".")[0] for f in os.listdir("students")]
student_embeddings = []

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

for f in os.listdir("students"):
    img = cv2.imread(f"students/{f}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        emb = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks]).flatten()
        student_embeddings.append(emb)
    else:
        student_embeddings.append(None)

status = {name: "غایب" for name in students}

# آپلود عکس کلاس
uploaded = st.file_uploader("عکس کلاس", type=["jpg", "png"])
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="عکس کلاس")
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10)

    results = mp_face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_emb = np.array([[lmk.x, lmk.y, lmk.z] for lmk in face_landmarks.landmark]).flatten()
            # مقایسه با embeddings دانش‌آموزان
            min_dist = float('inf')
            matched_idx = None
            for i, emb in enumerate(student_embeddings):
                if emb is not None:
                    dist = np.linalg.norm(face_emb - emb)
                    if dist < min_dist and dist < 0.1:  # threshold قابل تنظیم
                        min_dist = dist
                        matched_idx = i
            if matched_idx is not None:
                status[students[matched_idx]] = "حاضر"

    # نمایش جدول حضور
    df = pd.DataFrame(status.items(), columns=["نام دانش‌آموز", "وضعیت"])
    st.table(df)
