from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import json

# Initialize the API
app = FastAPI(title="Plant Doctor AI Engine")

# 1. Load the fine-tuned MobileNetV2 model
# (This happens once when the server starts, so predictions are fast)
model = load_model("plant_doctor.h5")

# 2. Load the 15 disease labels
# Assuming you have a labels.json file like: ["Healthy", "Tomato___Early_blight", ...]
with open("labels.json", "r") as file:
    CLASS_NAMES = json.load(file)


# Helper function to format the image for MobileNetV2
def preprocess_image(image_bytes):
    # Open the image and ensure it is RGB
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Resize to the standard MobileNetV2 input size
    img = img.resize((224, 224))
    # Convert to NumPy array and normalize to [0, 1]
    img_array = np.array(img) / 255.0
    # Expand dimensions to match the model's expected shape: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# 3. The API Endpoint that Laravel will call
@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        # Read the image sent by Laravel
        image_data = await file.read()

        # Preprocess the image
        processed_image = preprocess_image(image_data)

        # Run the model inference
        predictions = model.predict(processed_image)

        # Extract the highest probability result
        predicted_index = np.argmax(predictions[0])
        confidence_score = float(np.max(predictions[0]) * 100)
        disease_label = CLASS_NAMES[predicted_index]

        # Return the exact JSON contract expected by the Laravel developer
        return {
            "disease": disease_label,
            "confidence": round(confidence_score, 2)
        }

    except Exception as e:
        return {"error": str(e)}


# Start the server on port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)