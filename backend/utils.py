from pathlib import Path
import json
import joblib
import pandas as pd
from pydantic import BaseModel, Field

# Rutas: subo un nivel desde /backend hasta la raíz del proyecto y entro a /models
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "churn_model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
SCHEMA_PATH = MODELS_DIR / "model_columns.json"


def cargar_modelo():
    """Carga el pipeline entrenado (preprocesamiento + Random Forest) desde el .pkl."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No encontré el modelo en {MODEL_PATH}. "
            "Corré primero el notebook.ipynb para generar el .pkl."
        )
    return joblib.load(MODEL_PATH)


def cargar_metricas():
    """Lee las métricas guardadas por el notebook (para el endpoint /metrics)."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "metrics.json no encontrado. Ejecutá el notebook."}


def cargar_schema():
    """Lee el esquema de columnas (categorías válidas, orden de columnas)."""
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Esquema de ENTRADA de la API
# Nota: Pydantic valida automáticamente los datos que llegan. Si mandan
# un texto donde va un número, o falta un campo, la API responde un error claro
# (422) sin que yo tenga que chequear nada a mano. Los 'Field' con ejemplos también
# alimentan la documentación automática en /docs.
class ClienteInput(BaseModel):
    CreditScore: int = Field(..., ge=300, le=900, example=600)
    Geography: str = Field(..., example="Germany")          # France / Spain / Germany
    Gender: str = Field(..., example="Female")              # Male / Female
    Age: int = Field(..., ge=18, le=100, example=45)
    Tenure: int = Field(..., ge=0, le=10, example=3)
    Balance: float = Field(..., ge=0, example=120000.0)
    NumOfProducts: int = Field(..., ge=1, le=4, example=2)
    HasCrCard: int = Field(..., ge=0, le=1, example=1)
    IsActiveMember: int = Field(..., ge=0, le=1, example=0)
    EstimatedSalary: float = Field(..., ge=0, example=80000.0)


# Esquema de SALIDA
class PrediccionOutput(BaseModel):
    prediccion: int                 # 0 = se queda, 1 = se va
    etiqueta: str                   # texto legible
    probabilidad_churn: float       # 0..1
    mensaje: str


def predecir(modelo, datos: ClienteInput) -> PrediccionOutput:
    """
    Toma el modelo y un cliente validado, y devuelve la predicción.
    Nota: el pipeline del .pkl ya hace el escalado y el One-Hot solo,
    por eso le paso los datos CRUDOS en un DataFrame de una sola fila.
    """
    fila = pd.DataFrame([datos.dict()])
    pred = int(modelo.predict(fila)[0])
    proba = float(modelo.predict_proba(fila)[0, 1])

    if pred == 1:
        etiqueta = "SE VA (riesgo de churn)"
        mensaje = "Cliente en riesgo de abandono. Conviene una acción de retención."
    else:
        etiqueta = "SE QUEDA"
        mensaje = "Cliente con baja probabilidad de abandono."

    return PrediccionOutput(
        prediccion=pred,
        etiqueta=etiqueta,
        probabilidad_churn=round(proba, 4),
        mensaje=mensaje,
    )
