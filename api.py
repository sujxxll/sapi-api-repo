from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import os

app = FastAPI(title="SapiAgent - Mouse Dynamics AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load both models at startup (compile=False for Keras 3.x compatibility with legacy .h5 files)
print("Loading models...")
rnn_model = tf.keras.models.load_model("bidirectional_dx_dy_mse_supervised.h5", compile=False)
cnn_model = tf.keras.models.load_model("fcn_dx_dy_mse_supervised.h5", compile=False)
print("✅ Both models loaded!")


class MouseInput(BaseModel):
    actions: list[list[float]]


class ModelInfo(BaseModel):
    name: str
    type: str
    description: str
    input_shape: list
    output_shape: list


@app.get("/api/health")
def health():
    return {"status": "healthy", "models_loaded": True}


@app.get("/api/models")
def get_models():
    return {
        "models": [
            {
                "name": "Bidirectional RNN (GRU)",
                "key": "rnn",
                "endpoint": "/api/generate/rnn",
                "type": "Recurrent Neural Network",
                "description": "Bidirectional GRU-based autoencoder for mouse trajectory generation. Captures temporal dependencies in both forward and backward directions.",
                "input_shape": list(rnn_model.input_shape),
                "output_shape": list(rnn_model.output_shape),
                "parameters": int(rnn_model.count_params()),
            },
            {
                "name": "Fully Convolutional Network (FCN)",
                "key": "cnn",
                "endpoint": "/api/generate/cnn",
                "type": "Convolutional Neural Network",
                "description": "1D-CNN based autoencoder with bottleneck layer for mouse trajectory generation. Extracts spatial patterns from movement data.",
                "input_shape": list(cnn_model.input_shape),
                "output_shape": list(cnn_model.output_shape),
                "parameters": int(cnn_model.count_params()),
            },
        ]
    }


@app.post("/api/generate/rnn")
def generate_rnn(data: MouseInput):
    try:
        arr = np.array(data.actions, dtype=np.float32)
        inp = arr.reshape(1, arr.shape[0], arr.shape[1])
        output = rnn_model.predict(inp)
        # Compute reconstruction error
        mse = float(np.mean((inp - output) ** 2))
        return {
            "model": "bidirectional_rnn",
            "generated_actions": output[0].tolist(),
            "reconstruction_error": mse,
            "input_shape": list(arr.shape),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/generate/cnn")
def generate_cnn(data: MouseInput):
    try:
        arr = np.array(data.actions, dtype=np.float32)
        inp = arr.reshape(1, arr.shape[0], arr.shape[1])
        output = cnn_model.predict(inp)
        # Compute reconstruction error
        mse = float(np.mean((inp - output) ** 2))
        return {
            "model": "fcn_cnn",
            "generated_actions": output[0].tolist(),
            "reconstruction_error": mse,
            "input_shape": list(arr.shape),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/generate/both")
def generate_both(data: MouseInput):
    """Run both models and compare results."""
    try:
        arr = np.array(data.actions, dtype=np.float32)
        inp = arr.reshape(1, arr.shape[0], arr.shape[1])

        rnn_output = rnn_model.predict(inp)
        cnn_output = cnn_model.predict(inp)

        rnn_mse = float(np.mean((inp - rnn_output) ** 2))
        cnn_mse = float(np.mean((inp - cnn_output) ** 2))

        return {
            "rnn": {
                "model": "bidirectional_rnn",
                "generated_actions": rnn_output[0].tolist(),
                "reconstruction_error": rnn_mse,
            },
            "cnn": {
                "model": "fcn_cnn",
                "generated_actions": cnn_output[0].tolist(),
                "reconstruction_error": cnn_mse,
            },
            "input_shape": list(arr.shape),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Serve static frontend files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(static_dir, "index.html"))