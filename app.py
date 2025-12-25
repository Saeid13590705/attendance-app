import streamlit as st
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import os
from deepface import DeepFace

st.title("حضور و غیاب با عکس کلاس (DeepFace)")

# -----------------------------
# 1. بارگذاری دانش‌آموزان و استخراج embedding
# -----------------------------
students_dir = "students"
student_embeddings = []
student_names = []

st.write("در حال بارگذاری دانش‌آموزان ...")
for filename in os.listdir(students_dir):
    if filename.lower().endswith((".jpg", ".png")):
        path = os.path.join(students_dir, filename)
        try:
            embedding = DeepFace.represent(img_path=path, model_name="Facenet")[0]["embedding"]
            student_embeddings.append(np.array(embedding))
            student_names.append(os.path.splitext(filename)[0])
        except:
            st.warning(f"چهره‌ای در {filename} شناسایی نشد.")

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
    st.write("در حال شناسایی چهره‌ها ...")
    detections = DeepFace.extract_faces(img_path=np.array(image), detector_backend="opencv")

    for face_data in detections:
        x, y, w, h = face_data["facial_area"].values()
        face_img = face_data["face"]

        try:
            face_embedding = DeepFace.represent(img_path=face_img, model_name="Facenet")[0]["embedding"]
            face_embedding = np.array(face_embedding)

            # -----------------------------
            # 4. تطبیق با دانش‌آموزان
            # -----------------------------
            distances = [np.linalg.norm(face_embedding - s_emb) for s_emb in student_embeddings]
            min_dist = min(distances)
            min_index = distances.index(min_dist)

            threshold = 0.7  # فاصله مجاز برای شناسایی
            if min_dist < threshold:
                name = student_names[min_index]
                status[name] = "حاضر"
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)

            # رسم مستطیل و نام روی تصویر
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_rgb, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        except:
            continue

    # نمایش تصویر با نام‌ها
    st.image(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB), caption="نتیجه شناسایی")

    # -----------------------------
    # 5. نمایش جدول حضور و غیاب
    # -----------------------------
    df = pd.DataFrame(status.items(), columns=["نام دانش‌آموز", "وضعیت"])
    st.table(df)
