from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()


# Definición de la clase InputData
class InputData(BaseModel):
    pregnancies: int
    glucose: int
    blood_pressure: int
    skin_thickness: int
    insulin: int
    bmi: float
    diabetes_pedigree_function: float
    age: int


# Cargar el modelo de archivo .pkl
model = joblib.load("diabetes_model.pkl")


# Configuración de CORS
origins = ["*"]  # Ajusta esto según tus necesidades de seguridad

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint para realizar la predicción y almacenar en la base de datos
@app.post("/predict")
async def predict_diabetes(data: InputData):
    try:
        # Convertir datos de entrada a un array NumPy
        input_features = np.array([[
            data.pregnancies, data.glucose, data.blood_pressure,
            data.skin_thickness, data.insulin, data.bmi,
            data.diabetes_pedigree_function, data.age
        ]])

        # Realizar la predicción
        prediction = model.predict(input_features)

        # Convertir la predicción a formato booleano (tiene diabetes o no)
        has_diabetes = int(prediction[0])

        return {"has_diabetes": bool(has_diabetes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

