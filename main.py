from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import json
import tflite_runtime.interpreter as tflite

app = FastAPI()

# Load the labels
with open("labels.json", "r") as f:
    class_names = json.load(f)

# Load the lightweight TFLite model
interpreter = tflite.Interpreter(model_path="plant_doctor.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        # Read and format the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
        input_data = np.array(image, dtype=np.float32)

        # Add batch dimension: (1, 224, 224, 3)
        input_data = np.expand_dims(input_data, axis=0)

        # Normalize the image (Uncomment if you normalized during training)
        # input_data = input_data / 255.0

        # Feed the image to the model and run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Extract the prediction probabilities
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        return JSONResponse(content={
            "disease": class_names[predicted_index],
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})