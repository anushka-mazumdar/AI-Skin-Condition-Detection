import os
import cv2
from src.face_verification import verify_faces_and_pose
from src.enhancer import enhance_all_images
from src.condition_detector import predict_condition

RAW_DIR = "data/raw"
ENHANCED_DIR = "data/enhanced"

def capture_images():
    os.makedirs(RAW_DIR, exist_ok=True)
    cam = cv2.VideoCapture(0)
    print("\n🎥 Camera started. Capture sequence: FRONT → LEFT → RIGHT")
    print("Press ENTER in terminal when ready for each shot.\n")

    angles = ["front", "left", "right"]
    for angle in angles:
        input(f"Ready for {angle.upper()} view? Press ENTER to capture.")
        ret, frame = cam.read()
        if not ret:
            print(f"❌ Failed to capture {angle} view.")
            continue
        file_path = os.path.join(RAW_DIR, f"{angle}.jpg")
        cv2.imwrite(file_path, frame)
        print(f"✅ Captured {angle} view: {file_path}")

    cam.release()
    cv2.destroyAllWindows()


def main():
    print("=== 🧠 AI Skin Condition Detection System (PyTorch) ===")

    # 1. Capture user images
    capture_images()

    # 2. Face and Pose verification
    print("\n🔍 Verifying captured faces and poses...")
    verification_results = verify_faces_and_pose(RAW_DIR)
    invalid = [f for f, (face_ok, pose_ok, _) in verification_results.items() if not (face_ok and pose_ok)]
    if invalid:
        print("\n❌ Some captures are invalid (face missing or wrong angle). Please retake these:")
        for f in invalid:
            print(f"   -> {f}")
        return
    print("✅ All faces verified and poses correct.")

    # 3. Enhancement
    print("\n✨ Enhancing verified images...")
    enhance_all_images(RAW_DIR, ENHANCED_DIR)

    # 4. Skin condition detection (PyTorch)
    print("\n🩺 Detecting skin conditions...")
    for img_name in os.listdir(ENHANCED_DIR):
        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(ENHANCED_DIR, img_name)
            try:
                condition, confidence = predict_condition(path)
                print(f"   -> {img_name}: {condition} ({confidence*100:.2f}% confidence)")
            except Exception as e:
                print(f"   -> {img_name}: Prediction error: {e}")

    print("\n✅ Full process complete — from capture to condition detection.")


if __name__ == "__main__":
    main()
