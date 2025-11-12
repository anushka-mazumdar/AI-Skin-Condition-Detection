import cv2
import mediapipe as mp
import os
import numpy as np

mp_face = mp.solutions.face_detection
mp_mesh = mp.solutions.face_mesh

RAW_DIR = "data/raw"

def estimate_yaw(landmarks, image_w, image_h):
    """
    Roughly estimates head yaw (horizontal rotation) using eye and nose bridge landmarks.
    Negative = left, Positive = right.
    """
    left_eye = landmarks[33]  # left eye outer corner
    right_eye = landmarks[263]  # right eye outer corner
    nose_tip = landmarks[1]    # nose tip

    left_eye_x = left_eye.x * image_w
    right_eye_x = right_eye.x * image_w
    nose_x = nose_tip.x * image_w

    mid_eye_x = (left_eye_x + right_eye_x) / 2
    yaw = (nose_x - mid_eye_x) / (right_eye_x - left_eye_x)
    return yaw * 100  # scaled to percentage of rotation


def verify_faces_and_pose(raw_dir=RAW_DIR, min_conf=0.5):
    """
    Verifies face presence and angle correctness for front, left, right images.
    Returns a dict: {filename: (face_ok, pose_ok, yaw_angle)}
    """
    verified = {}
    face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=min_conf)
    face_mesh = mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    for fname in os.listdir(raw_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(raw_dir, fname)
        img = cv2.imread(path)
        if img is None:
            verified[fname] = (False, False, 0)
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_results = face_detector.process(rgb)
        if not face_results.detections:
            verified[fname] = (False, False, 0)
            print(f"❌ {fname}: No face detected")
            continue

        mesh_results = face_mesh.process(rgb)
        if not mesh_results.multi_face_landmarks:
            verified[fname] = (True, False, 0)
            print(f"⚠️ {fname}: Face detected but no landmarks found")
            continue

        image_h, image_w, _ = img.shape
        landmarks = mesh_results.multi_face_landmarks[0].landmark
        yaw = estimate_yaw(landmarks, image_w, image_h)

        # Determine acceptable yaw based on expected filename
        expected = "front" if "front" in fname else "left" if "left" in fname else "right"
        pose_ok = (
            (-10 < yaw < 10 and expected == "front") or
            (yaw < -10 and expected == "left") or
            (yaw > 10 and expected == "right")
        )

        verified[fname] = (True, pose_ok, yaw)
        status = "✅" if pose_ok else "⚠️"
        print(f"{status} {fname}: Face OK | Yaw={yaw:.2f} | Pose {'OK' if pose_ok else 'Mismatch'}")

    face_detector.close()
    face_mesh.close()
    return verified


if __name__ == "__main__":
    results = verify_faces_and_pose()
    print("\nVerification summary:")
    for name, (face_ok, pose_ok, yaw) in results.items():
        print(f"{name}: Face={'✔️' if face_ok else '❌'}, Pose={'✔️' if pose_ok else '❌'}, Yaw={yaw:.2f}")
