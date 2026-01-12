from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, cv2, uvicorn
from pyngrok import ngrok

from utils.face_utils import predict_student
from utils.data_manager import update_attendance_record, load_students

app = FastAPI(
    title="Face Recognition & Attendance API",
    description="API to predict students and update attendance",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def decode_uploaded_image(image: UploadFile):
    contents = await image.read()
    np_img = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)


def error_response(message: str, status: int = 500):
    return JSONResponse({"success": False, "error": message}, status_code=status)


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        img = await decode_uploaded_image(image)
        students = load_students()
        student, confidence = predict_student(img, students)

        if student:
            return {"success": True, "student": student, "confidence": confidence}
        return JSONResponse({"success": False, "message": "No match found"}, status_code=404)

    except Exception as e:
        return error_response(str(e))


@app.post("/attendance/update")
async def attendance_update(
    student_id: str = Query(..., description="Student ID"),
    start_time: str = Query(..., description="Allowed start time (HH:MM)"),
    end_time: str = Query(..., description="Allowed end time (HH:MM)")
):
    try:
        students = load_students()
        student = students.get(student_id)

        if not student:
            return JSONResponse({"success": False, "message": "Student not found"}, status_code=404)

        updated, now = update_attendance_record(student, start_time, end_time)

        if updated:
            return {
                "success": True,
                "student": student,
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S")
            }
        return {"success": False, "message": "Attendance not updated (outside time window or duplicate)"}

    except Exception as e:
        return error_response(str(e))


@app.post("/recognize-and-update")
async def recognize_and_update(
    image: UploadFile = File(...),
    start_time: str = Query(..., description="Allowed start time (HH:MM)"),
    end_time: str = Query(..., description="Allowed end time (HH:MM)")
):
    try:
        img = await decode_uploaded_image(image)
        students = load_students()
        student, confidence = predict_student(img, students)

        if not student:
            return JSONResponse({"success": False, "message": "No match found"}, status_code=404)

        updated, now = update_attendance_record(student, start_time, end_time)

        return {
            "success": True,
            "student": student,
            "confidence": confidence,
            "attendance_updated": updated,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        return error_response(str(e))


@app.get("/students")
async def get_students():
    try:
        students = load_students()
        return {"success": True, "count": len(students), "students": students}
    except Exception as e:
        return error_response(str(e))


if __name__ == "__main__":
    # public_url = ngrok.connect(8000).public_url
    # print("Public API:", public_url)
    # print("Docs:", public_url + "/docs")

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)