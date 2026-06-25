# 🏦 Predicción de Abandono de Clientes Bancarios (Churn)

**Trabajo Práctico Final — Taller de Lenguajes de Programación III (Python para Ciencia de Datos)**

**Alumno:** Ruiz Diaz Dario Ezequiel

Solución de IA integral que cubre el ciclo de vida completo de un proyecto de datos: desde el análisis exploratorio hasta una interfaz web que consume el modelo a través de una API.

---

## 🎯 El problema

Un banco quiere detectar **qué clientes están por cerrar su cuenta** (*churn*) antes de que se vayan, para poder retenerlos. Es un problema de **clasificación binaria supervisada**: predecir la variable `Exited` (1 = se va, 0 = se queda).

**Dataset:** [Bank Customer Churn — Kaggle](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction) · 10.000 clientes, 14 columnas.

---

## 🧠 El modelo

- **Modelo final:** `RandomForestClassifier` **podado** (`max_depth=8`, `min_samples_leaf=20`, `min_samples_split=20`) para controlar el overfitting.
- **Baseline de comparación:** Regresión Logística.
- **Preprocesamiento** (dentro de un `Pipeline` de scikit-learn): escalado de variables numéricas (`StandardScaler`) + codificación One-Hot de categóricas (`Geography`, `Gender`).
- **Exportación:** el pipeline completo se serializa con `joblib` en `models/churn_model.pkl`.

| Métrica (test) | Random Forest podado | Reg. Logística (baseline) |
|---|---|---|
| Accuracy | 0.866 | 0.816 |
| Precisión (clase "se va") | 0.852 | 0.627 |
| Recall (clase "se va") | 0.410 | 0.231 |
| F1 (clase "se va") | 0.554 | 0.338 |
| ROC-AUC | 0.860 | 0.783 |

> El recall no es alto a propósito: el bosque está **podado** para no memorizar el train. Las métricas reportadas son sobre un conjunto de test que el modelo nunca vio, así que son honestas. Si el negocio priorizara maximizar el recall, se podría usar `class_weight='balanced'` o SMOTE (a costa de más falsos positivos).

---

## 🛠️ Stack tecnológico

| Capa | Tecnología |
|---|---|
| Análisis y modelado | Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn |
| Serialización del modelo | joblib |
| Backend / API | FastAPI + Uvicorn |
| Frontend | HTML + CSS + JavaScript (fetch) |

---

## 📁 Estructura del proyecto

```
churn-bancario/
├── data/
│   └── Churn_Modelling.csv        # Dataset original (Kaggle)
├── models/
│   ├── churn_model.pkl            # Pipeline entrenado (joblib)
│   ├── metrics.json               # Métricas del modelo (las consume el frontend)
│   └── model_columns.json         # Esquema de entrada / categorías válidas
├── backend/
│   ├── main.py                    # API FastAPI (endpoints)
│   └── utils.py                   # Carga del modelo + esquema Pydantic + predicción
├── frontend/
│   └── index.html                 # Interfaz web (formulario + métricas)
├── notebook.ipynb                 # Notebook de ciencia de datos (EDA → export)
├── requirements.txt
└── README.md
```

---

## 🚀 Cómo ejecutar el proyecto

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. (Opcional) Regenerar el modelo

El `.pkl` ya viene incluido. Si querés volver a entrenarlo, abrí y ejecutá `notebook.ipynb` de arriba a abajo. Eso regenera `models/churn_model.pkl`, `metrics.json` y `model_columns.json`.

### 3. Levantar la API (backend)

```bash
cd backend
uvicorn main:app --reload
```

La API queda en `http://127.0.0.1:8000`. La documentación interactiva (Swagger) está en `http://127.0.0.1:8000/docs`.

### 4. Abrir el frontend

Abrí `frontend/index.html` con doble clic en el navegador. Cargá los datos de un cliente y presioná **Predecir**. El panel de la derecha muestra el rendimiento del modelo (se lee de la API).

> **Importante:** la API tiene que estar corriendo (paso 3) para que el frontend pueda predecir y mostrar las métricas.

---

## 🔌 Endpoints de la API

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Mensaje de bienvenida y lista de endpoints |
| `GET` | `/health` | Estado del servicio (modelo cargado) |
| `POST` | `/predict` | Recibe los datos de un cliente (JSON) y devuelve la predicción |
| `GET` | `/metrics` | Métricas de rendimiento del modelo |
| `GET` | `/schema` | Esquema de entrada y categorías válidas |
| `GET` | `/docs` | Documentación interactiva (Swagger UI) |

**Ejemplo de request a `/predict`:**

```json
{
  "CreditScore": 600,
  "Geography": "Germany",
  "Gender": "Female",
  "Age": 45,
  "Tenure": 3,
  "Balance": 120000.0,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 0,
  "EstimatedSalary": 80000.0
}
```

**Respuesta:**

```json
{
  "prediccion": 0,
  "etiqueta": "SE QUEDA",
  "probabilidad_churn": 0.4872,
  "mensaje": "Cliente con baja probabilidad de abandono."
}
```

---

*Desarrollado por Ruiz Diaz Dario Ezequiel — Taller de Lenguajes de Programación III.*
