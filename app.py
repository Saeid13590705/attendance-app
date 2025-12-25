# app.py
import streamlit as st
from deepface import DeepFace
from PIL import Image
import os
import pandas as pd
import numpy as np

st.title("حضور و غیاب با عکس کلاس")

# مسیر فولدر دانش‌آموزان
STUDENTS_DIR = "students"

# بارگذاری اسامی دانش‌آموزان و embedding‌ها
student_names = []
student_embeddings = []

for file in os.listdir(STUDENTS_DIR):
    if file.lower().endswith((".jpg", ".png")):
        name = os.path.splitext(file)[0]
        img_path = os.path.join(STUDENTS_DIR, file)
        student_names.append(name)
        embedding = DeepFace.represent(img_path=img_path, model_name="Facenet")[0]["embedding"]
        student_embeddings.append(np.array(embedding))

student_embeddings = np.array(student_embeddings)

# آپلود عکس کلاس
uploaded_file = st.file_uploader("عکس کلاس را آپلود کن", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="عکس کلاس")
    
    # شناسایی همه چهره‌ها و گرفتن embedding
    results = DeepFace.extract_faces(img_path=np.array(image), detector_backend="mtcnn", enforce_detection=False)
    
    status = {name: "غایب" for name in student_names}

    for face in results:
        face_img = face["face"]
        face_embedding = DeepFace.represent(img_path=face_img, model_name="Facenet")[0]["embedding"]
        face_embedding = np.array(face_embedding)

        # محاسبه فاصله با تمام دانش‌آموزان
        distances = np.linalg.norm(student_embeddings - face_embedding, axis=1)
        min_idx = np.argmin(distances)
        
        # اگر فاصله کمتر از threshold بود، دانش‌آموز حاضر است
        if distances[min_idx] < 0.4:
            status[student_names[min_idx]] = "حاضر"

    # نمایش دانش‌آموزان حاضر
    present = [name for name, stt in status.items() if stt == "حاضر"]
    if present:
        st.subheader("✅ دانش‌آموزان حاضر:")
        st.write(", ".join(present))
    else:
        st.subheader("هیچ دانش‌آموزی حاضر نیست.")

    # نمایش جدول حضور و غیاب
    df = pd.DataFrame(status.items(), columns=["نام دانش‌آموز", "وضعیت"])
    st.table(df)
