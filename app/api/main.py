from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
from src.face_verification import verify_faces_and_pose
from src.enhancer import enhance_all_images
from src.condition_detector import predict_condition

app = FastAPI(title="AI Skin Condition Detection API (PyTorch)")

RAW_DIR = "data/raw"
ENHANCED_DIR = "data/enhanced"

def save_uploaded_images(files):
    os.makedirs(RAW_DIR, exist_ok=True)
    # Clear previous captures
    for old in os.listdir(RAW_DIR):
        os.remove(os.path.join(RAW_DIR, old))

    expected_angles = ["front", "left", "right"]
    if len(files) != 3:
        raise HTTPException(status_code=400, detail="Please upload exactly 3 images: front, left, and right.")

    for idx, file in enumerate(files):
        filename = f"{expected_angles[idx]}.jpg"
        with open(os.path.join(RAW_DIR, filename), "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    return RAW_DIR


@app.post("/analyze/")
async def analyze_skin(front: UploadFile = File(...), left: UploadFile = File(...), right: UploadFile = File(...)):
    try:
        save_uploaded_images([front, left, right])

        verification_results = verify_faces_and_pose(RAW_DIR)
        invalid = [f for f, (face_ok, pose_ok, _) in verification_results.items() if not (face_ok and pose_ok)]
        if invalid:
            return JSONResponse(status_code=400, content={"error": f"Invalid captures: {invalid}. Please retake with proper angles."})

        enhance_all_images(RAW_DIR, ENHANCED_DIR)

        results = []
        for img_name in os.listdir(ENHANCED_DIR):
            if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(ENHANCED_DIR, img_name)
                try:
                    condition, confidence = predict_condition(path)
                    results.append({"image": img_name, "condition": condition, "confidence": round(confidence * 100, 2)})
                except Exception as e:
                    results.append({"image": img_name, "error": str(e)})

        return {"status": "success", "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
