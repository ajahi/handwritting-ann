from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
import joblib

app = FastAPI()

# Load the pre-trained model
try:
    ann_model = joblib.load('handwritting.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Model loading failed")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("templates/index.html", "r") as file:
            return HTMLResponse(content=file.read())
    except Exception as e:
        print(f"Error reading HTML file: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/predict")
async def predict(request: Request):
    try:
        form_data = await request.form()
        pixels = form_data.get('pixels')

        if not pixels:
            raise ValueError("No pixel data received")

        # Convert the pixel data from string to a numpy array
        pixel_array = np.array(list(map(float, pixels.split(','))))
        print(f"Pixel array shape before reshape: {pixel_array.shape}")

        # Reshape the array to match the input shape expected by the model
        pixel_array = pixel_array.reshape(1, -1)
        print(f"Pixel array shape after reshape: {pixel_array.shape}")

        # Make a prediction using the model
        prediction = ann_model.predict(pixel_array)
        print(f"Prediction: {prediction}")

        # Ensure the prediction is a 1D array
        if prediction.ndim > 1:
            prediction = prediction.flatten()

        # Convert the prediction probabilities to a list and pair them with their indices
        prediction_list = prediction.tolist()
        prediction_with_indices = [(i, prob) for i, prob in enumerate(prediction_list)]

        # Return the predicted digit and the full probability distribution
        return JSONResponse(content={
            "predicted_digit": int(np.argmax(prediction)),
            "probabilities": prediction_with_indices
        })

    except ValueError as ve:
        print(f"ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        print(f"Exception error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
