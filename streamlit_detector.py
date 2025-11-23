
# v2 - login/gallery added - {datetime.now()}

import streamlit as st
import cv2
import numpy as np
import joblib
from skimage import transform, exposure, feature
from PIL import Image
import io
import os
from datetime import datetime
import json
import hashlib  

st.set_page_config(layout="wide")


AGE_PROTOTXT_PATH = "deploy_age.prototxt"
AGE_MODEL_PATH = "age_net.caffemodel"
SMILE_PIPELINE_PATH = "full_smile_pipeline.pkl"
SMILE_THRESHOLD = 0.1
GALLERY_DIR = "smile_gallery"
USERS_FILE = "users.json"  
os.makedirs(GALLERY_DIR, exist_ok=True)


if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump({}, f) 

def load_users():
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


@st.cache_resource
def load_models():
    age_net = cv2.dnn.readNetFromCaffe(AGE_PROTOTXT_PATH, AGE_MODEL_PATH)
    smile_pipeline = joblib.load(SMILE_PIPELINE_PATH)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return age_net, smile_pipeline, face_cascade

age_net, smile_pipeline, face_cascade = load_models()

def preprocess_smile(face_crop):
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    gray = transform.resize(gray, (64, 64), anti_aliasing=True)
    gray = exposure.equalize_adapthist(gray)
    hog_features = feature.hog(
        gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True
    )
    return hog_features.reshape(1, -1)

def process_image(image):
    image = cv2.convertScaleAbs(image, alpha=1.2, beta=25)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    results = []
    annotated_image = image.copy()
    raw_age_buckets = [
        "(0-2)", "(4-6)", "(8-13)", "(15-20)",
        "(25-32)", "(38-43)", "(48-53)", "(60-100)"
    ]
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        if face.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227),
            (78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False
        )
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age_bucket_idx = age_preds[0].argmax()
        raw_range = raw_age_buckets[age_bucket_idx]

        min_age = int(raw_range[1:].split('-')[0])
        max_age = int(raw_range[1:].split('-')[1][:-1])
        approx_age = int((min_age + max_age) / 2)
        age_display = f"Age: {raw_range} (~{approx_age}yrs)"

        smile_input = preprocess_smile(face)
        smile_prob = smile_pipeline.predict_proba(smile_input)[0][1]

        color_box = (0, 255, 0) if smile_prob >= SMILE_THRESHOLD else (0, 0, 255)
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color_box, 2)
        cv2.putText(annotated_image, age_display, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated_image, f"Smile: {smile_prob:.2f}", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_box, 2)
        results.append({
            "age_range": raw_range,
            "approx_age": f"~{approx_age}yrs",
            "smile_prob": smile_prob,
            "is_smiling": smile_prob >= SMILE_THRESHOLD
        })
    return annotated_image, results


def save_to_gallery(annotated_image, results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{GALLERY_DIR}/capture_{timestamp}.jpg"
    cv2.imwrite(filename, annotated_image)
    return filename


def load_gallery():
    images = []
    for file in sorted(os.listdir(GALLERY_DIR), reverse=True):
        if file.endswith(".jpg"):
            path = os.path.join(GALLERY_DIR, file)
            img = Image.open(path)
            images.append((file, img))
    return images


st.markdown("""
    <style>
    .nav-bar {
        background-color: #333;
        padding: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: white;
    }
    .orange-button {
        background-color: #FF8C00 !important;
        color: white !important;
        border: none !important;
        padding: 8px 8px !important;
        margin-left: 5px !important;
    }
    .stButton > button {
        background-color: #FF8C00;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

def nav_bar():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<span class='title'>SmileAge.ai : AI Tool for Smile and Age Detection</span>", unsafe_allow_html=True)
    with col2:
        col_login, col_signup = st.columns(2)
        with col_login:
            if st.button("Login"):
                st.session_state['auth_mode'] = "login"
                st.rerun()
        with col_signup:
            if st.button("Sign Up"):
                st.session_state['auth_mode'] = "signup"
                st.rerun()

def login_form():
    st.markdown("<h4>Login to SmileAge.ai</h4>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    return username, password

def signup_form():
    st.markdown("<h4>Sign Up for SmileAge.ai</h4>", unsafe_allow_html=True)
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    return new_username, new_password, confirm_password

def main_app():
    #st.title("Smile and Age Detector")
    # Tabs: Detect + Gallery
    tab1, tab2 = st.tabs(["Detect", "Gallery"])
    with tab1:
        option = st.radio("Choose input method:", ("Upload Image", "Capture from Webcam"))
        image = None
        if option == "Upload Image":
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image_bytes = uploaded_file.read()
                image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        elif option == "Capture from Webcam":
            camera_image = st.camera_input("Take a picture")
            if camera_image is not None:
                bytes_data = camera_image.getvalue()
                image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            st.write("Processing image...")
            annotated_image, results = process_image(image)
            saved_path = save_to_gallery(annotated_image, results)
            st.success(f"Saved to gallery: {os.path.basename(saved_path)}")
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_image_rgb, caption="Processed Image", use_container_width=True)
            if results:
                for i, res in enumerate(results):
                    st.write(f"**Face {i+1}:**")
                    st.write(f"- Age Range: {res['age_range']}")
                    st.write(f"- Approximate Age: {res['approx_age']}")
                    st.write(f"- Smile Probability: {res['smile_prob']:.2f}")
                    st.write(f"- Smiling: {'Yes' if res['is_smiling'] else 'No'}")
            else:
                st.write("No faces detected.")
        else:
            st.write("Please upload an image or capture one from the webcam.")
    with tab2:
        st.header("Captured Gallery")
        gallery = load_gallery()
        if gallery:
            cols = st.columns(3)
            for idx, (name, img) in enumerate(gallery):
                with cols[idx % 3]:
                    st.image(img, caption=name, use_container_width=True)
        else:
            st.info("No captures yet. Try detecting a face!")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'auth_mode' not in st.session_state:
    st.session_state['auth_mode'] = "signup" 
nav_bar() 
if not st.session_state['logged_in']:
    if st.session_state['auth_mode'] == "login":
        username, password = login_form()
        if st.button("Submit Login", key="submit_login"):
            users = load_users()
            hashed_pw = hash_password(password)
            if username in users and users[username] == hashed_pw:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Incorrect username or password.")
    else:
        new_username, new_password, confirm_password = signup_form()
        if st.button("Submit Sign Up", key="submit_signup"):
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            elif not new_username or not new_password:
                st.error("Please fill in all fields.")
            else:
                users = load_users()
                if new_username in users:
                    st.error("Username already exists.")
                else:
                    hashed_pw = hash_password(new_password)
                    users[new_username] = hashed_pw
                    save_users(users)
                    st.success("Signed up successfully! Please log in.")
                    st.session_state['auth_mode'] = "login"
                    st.rerun()
else:
    main_app()