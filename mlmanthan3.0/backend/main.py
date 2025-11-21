from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import io
import json
import os

from data_loader import load_dataset, simulate_dataset, preprocess_retail_data
from causal_engine import estimate_causal_effect, get_causal_graph_image

app = FastAPI(title="Causal Inference Marketing Attribution")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend
# Check if running from backend dir or root
if os.path.exists("../frontend"):
    app.mount("/app", StaticFiles(directory="../frontend", html=True), name="static")
elif os.path.exists("frontend"):
    app.mount("/app", StaticFiles(directory="frontend", html=True), name="static")


# Global state for the loaded dataframe
# In a production app, this would be a database or session-based storage
state = {
    "df": None
}
TEMP_DATA_FILE = "temp_data.pkl"
DEFAULT_DATA_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Online Retail.csv"))

def save_state(df: pd.DataFrame):
    state["df"] = df
    try:
        df.to_pickle(TEMP_DATA_FILE)
    except Exception as e:
        print(f"Warning: Could not persist data: {e}")

def load_state():
    if state["df"] is not None:
        return state["df"]
    
    if os.path.exists(TEMP_DATA_FILE):
        try:
            df = pd.read_pickle(TEMP_DATA_FILE)
            state["df"] = df
            return df
        except Exception as e:
            print(f"Warning: Could not reload data: {e}")

    # Fallback: auto-load bundled Online Retail dataset if available
    if os.path.exists(DEFAULT_DATA_FILE):
        try:
            with open(DEFAULT_DATA_FILE, "rb") as f:
                content = f.read()
            df = load_dataset(content, os.path.basename(DEFAULT_DATA_FILE))
            df = preprocess_retail_data(df)
            save_state(df)
            return df
        except Exception as e:
            print(f"Warning: Could not load default dataset: {e}")
    return None

class AnalysisRequest(BaseModel):
    treatment: str
    outcome: str
    confounders: List[str]

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = load_dataset(content, file.filename)
        
        # If it's the specific retail dataset, preprocess it automatically
        if "Retail" in file.filename or "retail" in file.filename:
            df = preprocess_retail_data(df)
            
        save_state(df)
        return {"message": "File uploaded successfully", "columns": df.columns.tolist(), "rows": len(df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/simulate")
async def trigger_simulation():
    try:
        df = simulate_dataset()
        save_state(df)
        return {"message": "Simulation data generated", "columns": df.columns.tolist(), "rows": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/columns")
async def get_columns():
    df = load_state()
    if df is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    return {"columns": df.columns.tolist()}

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    df = load_state()
    if df is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        result = estimate_causal_effect(
            df,
            request.treatment,
            request.outcome,
            request.confounders
        )
        
        graph_image = get_causal_graph_image(
            df,
            request.treatment,
            request.outcome,
            request.confounders
        )
        
        return {
            "result": result,
            "graph_image": graph_image
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Causal Inference API is running"}
