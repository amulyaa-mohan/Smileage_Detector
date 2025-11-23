import cv2
import numpy as np
import time
import joblib
from skimage import transform, color, exposure, feature

AGE_PROTOTXT_PATH = "deploy_age.prototxt"
AGE_MODEL_PATH = "age_net.caffemodel"
SMILE_PIPELINE_PATH = "full_smile_pipeline.pkl"
SMILE_THRESHOLD = 0.2
CAPTURE_DELAY = 0.5
SAVE_PATH = "smile_capture.jpg"

print("Loading models...")
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTOTXT_PATH, AGE_MODEL_PATH)
print("AgeNet model loaded successfully.")

smile_pipeline = joblib.load(SMILE_PIPELINE_PATH)
print("Smile pipeline loaded successfully.")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def preprocess_smile(face_crop):
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    gray = transform.resize(gray, (64, 64), anti_aliasing=True)
    gray = exposure.equalize_adapthist(gray)  # normalize lighting
    hog_features = feature.hog(
        gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True
    )
    return hog_features.reshape(1, -1)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not accessible.")
    exit()

smile_start = None
captured = False

print("Smile to capture! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=25)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
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
        confidence = float(age_preds[0][age_bucket_idx])

        bucket_centers = [1, 5, 10, 18, 25, 35, 45, 65]
        model_age = bucket_centers[age_bucket_idx]

        if model_age <= 25:
            approx_age = model_age - 4
        elif model_age <= 40:
            approx_age = model_age - 2
        else:
            approx_age = model_age + 8

        approx_age = max(1, approx_age)

        low = max(1, approx_age - 4)
        high = approx_age + 4

        age_range = f"{low}-{high}"
        approx_age_text = f"~{approx_age}yrs"

        smile_input = preprocess_smile(face)
        smile_prob = smile_pipeline.predict_proba(smile_input)[0][1]

        color_box = (0, 255, 0) if smile_prob >= SMILE_THRESHOLD else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color_box, 2)

        cv2.putText(
            frame, f"Age: {age_range} ({approx_age_text})",
            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )

        cv2.putText(
            frame, f"Smile: {smile_prob:.2f}",
            (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_box, 2
        )

        if smile_prob >= SMILE_THRESHOLD:
            if smile_start is None:
                smile_start = time.time()
                print(" Hold that smile...")
            elif time.time() - smile_start >= CAPTURE_DELAY:
                cv2.imwrite(SAVE_PATH, frame)
                print(f"CAPTURED! → Saved to {SAVE_PATH}")
                captured = True
                break
        else:
            smile_start = None

    cv2.imshow("Smart Smile+Age Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q") or captured:
        break

cap.release()
cv2.destroyAllWindows()
print("Session ended.")
