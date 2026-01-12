import cv2, os, numpy as np

try:
    from config import MODEL_PATH, IMAGES_DIR, LOGS_DIR
    from logger import log_message
except ImportError:
    from utils.config import MODEL_PATH, IMAGES_DIR, LOGS_DIR
    from utils.logger import log_message

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)

def save_face_snapshot(student: dict, frame, face_coords, timestamp):
    (x, y, w, h) = face_coords
    margin = 200
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
    face_img = frame[y1:y2, x1:x2]

    filename = f"{student['id']}-{timestamp.strftime('%Y%m%d%H%M%S')}.png"
    filepath = os.path.join(LOGS_DIR, filename)
    cv2.imwrite(filepath, face_img)

    log_message(f"📸 Snapshot saved for {student['nama']}: {filepath}")

def predict_student(gray_face, students, threshold=60):
    recognizer.read(MODEL_PATH)
    face_resized = preprocess_face(gray_face)

    if face_resized is None:
        return None, None

    id_pred, conf = recognizer.predict(face_resized)

    if conf < threshold:
        student = students.get(str(id_pred))
        return student, conf
    else:
        return None, conf

def preprocess_face(img):
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return cv2.resize(img[y:y+h, x:x+w], (200, 200))

def train_model(log_box=None):
    faces, labels = [], []
    log_message("🔄 Collecting faces for training...", log_box)

    for student_id in os.listdir(IMAGES_DIR):
        folder = os.path.join(IMAGES_DIR, student_id)
        if not os.path.isdir(folder):
            continue

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                log_message(f"⚠️ Could not read {img_path}", log_box)
                continue

            face = preprocess_face(img)
            if face is None:
                log_message(f"⚠️ No face detected in {img_name}", log_box)
                continue

            face = cv2.equalizeHist(face)
            faces.append(face)
            labels.append(int(student_id))

    if faces:
        recognizer.train(faces, np.array(labels))
        recognizer.save(MODEL_PATH)
        log_message(f"✅ Model trained with {len(faces)} samples and {len(set(labels))} students", log_box)
    else:
        log_message("❌ No valid images found, training aborted", log_box)

def save_faces(student_id, photo_paths, folder, log_box=None):
    os.makedirs(folder, exist_ok=True)
    existing = len([f for f in os.listdir(folder) if f.endswith((".jpg", ".png", ".jpeg"))])
    count = 0

    for i, photo_path in enumerate(photo_paths, start=existing + 1):
        img = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            log_message(f"⚠️ Invalid file {photo_path}", log_box)
            continue

        face = preprocess_face(img)
        if face is None:
            log_message(f"⚠️ No face in {photo_path}", log_box)
            continue

        save_path = os.path.join(folder, f"{student_id}_{i}.jpg")
        cv2.imwrite(save_path, face)
        count += 1
        log_message(f"✅ Saved face: {save_path}", log_box)

    return count