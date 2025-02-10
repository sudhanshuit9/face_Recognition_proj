from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

from face_recognition import detect_and_encode, recognize_face, store_face

app = FastAPI()

# Store a Face
@app.post("/store_face/")
async def store_face_api(name: str = Form(...), file: UploadFile = File(...)):
    image_data = await file.read()
    image = np.array(Image.open(BytesIO(image_data)).convert("RGB"))
    
    encodings = detect_and_encode(image)
    if encodings:
        store_face(name, encodings[0])
        return {"message": f"Stored {name}'s face in database"}
    return {"error": "No face detected"}

# Recognize Face
@app.post("/recognize_face/")
async def recognize_face_api(file: UploadFile = File(...)):
    image_data = await file.read()
    image = np.array(Image.open(BytesIO(image_data)).convert("RGB"))

    test_encodings = detect_and_encode(image)
    if not test_encodings:
        return {"error": "No face detected"}

    recognized_names = recognize_face(test_encodings)
    return {"recognized_faces": recognized_names}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)