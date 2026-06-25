from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from utils import (cargar_modelo, cargar_metricas, cargar_schema,
                   ClienteInput, PrediccionOutput, predecir)

# Creo la app
app = FastAPI(
    title="API Predicción de Churn Bancario",
    description="Predice si un cliente del banco va a cerrar su cuenta. "
                "TP Final - Ruiz Diaz Dario Ezequiel.",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargo el modelo UNA sola vez, al arrancar el servidor
# Nota de Dario: cargarlo acá (y no dentro de /predict) hace que cada predicción sea
# rápida, porque el .pkl ya quedó en memoria. Cargarlo en cada request sería lentísimo.
modelo = cargar_modelo()


@app.get("/")
def raiz():
    """Endpoint de bienvenida para chequear que la API está viva."""
    return {
        "mensaje": "API de predicción de churn bancario funcionando.",
        "autor": "Ruiz Diaz Dario Ezequiel",
        "endpoints": {
            "POST /predict": "Envía datos de un cliente y devuelve la predicción.",
            "GET /metrics": "Métricas de rendimiento del modelo.",
            "GET /schema": "Esquema de entrada y categorías válidas.",
            "GET /health": "Estado del servicio.",
            "GET /docs": "Documentación interactiva (Swagger).",
        },
    }


@app.get("/health")
def health():
    """Healthcheck simple: confirma que el modelo está cargado."""
    return {"status": "ok", "modelo_cargado": modelo is not None}


@app.get("/metrics")
def metrics():
    """Devuelve las métricas del modelo (las usa el frontend para mostrar el rendimiento)."""
    return cargar_metricas()


@app.get("/schema")
def schema():
    """Categorías válidas y orden de columnas (le sirve al frontend para armar el form)."""
    return cargar_schema()


@app.post("/predict", response_model=PrediccionOutput)
def predict(cliente: ClienteInput):
    """
    Recibe los datos de un cliente (JSON) y devuelve si se va o se queda,
    junto con la probabilidad de churn.
    """
    try:
        return predecir(modelo, cliente)
    except Exception as e:
        # Nota: si algo raro pasa al predecir, devuelvo un 500 con el detalle
        # en vez de que la API se caiga en silencio.
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")
