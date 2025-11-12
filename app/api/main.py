import os
import shutil
import time
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from src.face_verification import verify_faces_and_pose
from src.enhancer import enhance_all_images
from src.condition_detector import predict_condition

app = FastAPI(title="AI Skin Condition Detection API", version="2.0")

RAW_DIR = "data/raw"
ENHANCED_DIR = "data/enhanced"

# === Helper function: save uploaded images ===
def save_uploaded_images(files):
    os.makedirs(RAW_DIR, exist_ok=True)
    # Clear old captures
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


# === Background processing task ===
def process_images():
    """Runs the core AI pipeline asynchronously."""
    start_time = time.time()

    # Step 1: Verify face and pose
    verification_results = verify_faces_and_pose(RAW_DIR)
    invalid = [f for f, (face_ok, pose_ok, _) in verification_results.items() if not (face_ok and pose_ok)]
    if invalid:
        return {
            "status": "error",
            "message": f"Invalid captures: {invalid}. Retake with proper angles.",
            "time_taken": round(time.time() - start_time, 2)
        }

    # Step 2: Enhance images
    enhance_all_images(RAW_DIR, ENHANCED_DIR)

    # Step 3: Run condition detection
    results = []
    for img_name in os.listdir(ENHANCED_DIR):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(ENHANCED_DIR, img_name)
            condition, confidence = predict_condition(path)
            results.append({
                "image": img_name,
                "condition": condition,
                "confidence": round(confidence * 100, 2)
            })

    elapsed = round(time.time() - start_time, 2)
    return {"status": "success", "results": results, "time_taken": elapsed}


# === Route with async + background task support ===
@app.post("/analyze/")
async def analyze_skin(
    background_tasks: BackgroundTasks,
    front: UploadFile = File(...),
    left: UploadFile = File(...),
    right: UploadFile = File(...)
):
    try:
        # Save uploaded images
        save_uploaded_images([front, left, right])

        # Run processing asynchronously
        start_time = time.time()
        response = process_images()
        total_time = round(time.time() - start_time, 2)

        if response["status"] == "error":
            return JSONResponse(status_code=400, content=response)

        response["server_response_time"] = total_time
        return JSONResponse(status_code=200, content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
