# SapiAgent - Mouse Dynamics AI

A FastAPI-powered web application that uses deep learning models (Bidirectional RNN & FCN) to analyze and generate mouse movement trajectories. Built for research on human-like mouse dynamics.

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (tested with Python 3.12)

### 1. Clone the Repository

```bash
git clone https://github.com/sujxxll/sapi-api-repo.git
cd sapi-api-repo
```

### 2. Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirement.txt
```

### 4. Run the Application

```bash
uvicorn api:app --reload
```

The app will be available at **http://127.0.0.1:8000**

## 📁 Project Structure

```
sapi-api-repo/
├── api.py                                  # FastAPI backend
├── static/
│   └── index.html                          # Frontend UI
├── bidirectional_dx_dy_mse_supervised.h5    # RNN model
├── bidirectional_dx_dy_mse_unsupervised.h5 # RNN model (unsupervised)
├── fcn_dx_dy_mse_supervised.h5             # CNN model
├── fcn_dx_dy_mse_unsupervised.h5           # CNN model (unsupervised)
├── requirement.txt                         # Python dependencies
├── Procfile                                # Deployment config
└── README.md
```

## 🔌 API Endpoints

| Method | Endpoint             | Description                          |
|--------|----------------------|--------------------------------------|
| GET    | `/`                  | Serves the frontend UI               |
| GET    | `/api/health`        | Health check                         |
| GET    | `/api/models`        | List available models & their info   |
| POST   | `/api/generate/rnn`  | Generate trajectory using RNN model  |
| POST   | `/api/generate/cnn`  | Generate trajectory using CNN model  |
| POST   | `/api/generate/both` | Run both models & compare results    |

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/api/generate/rnn \
  -H "Content-Type: application/json" \
  -d '{"actions": [[0.1, 0.2], [0.3, 0.4], ...]}'
```

## 🛠️ Tech Stack

- **Backend:** FastAPI, Uvicorn
- **AI/ML:** TensorFlow, NumPy
- **Frontend:** HTML, CSS, JavaScript
