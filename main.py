# main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import pandas as pd
import shutil
import base64
import os

from eda_utils import get_missing_values, plot_distributions, plot_correlation, plot_churn_by_feature, plot_timeline, generate_pdf_report
from survival_utils import run_kaplan_meier, run_cox_model, get_risk_scores
from ml_utils import run_complete_ml_pipeline
from causal_utils import run_causal_analysis
from rl_utils import train_rl_agent, get_rl_recommendation

class ReportRequest(BaseModel):
    plots: list[str]

# This Pydantic model now correctly matches your data and the frontend form
class UserState(BaseModel):
    age: int
    monthly_fee: float
    watch_hours: float

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

app.mount("/plots", StaticFiles(directory="plots"), name="plots")

@app.get("/")
async def root():
    return {"message": "ChurnSense AI Backend is running!"}

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        df = pd.read_csv(file_path)

        churn_column = None
        possible_churn_cols = ['churned', 'churn']
        for col in df.columns:
            if col.lower() in possible_churn_cols:
                churn_column = col
                break
        
        if not churn_column:
            raise ValueError("A 'churned' or 'churn' column is required for analysis but was not found.")

        time_column = None
        possible_time_cols = ['tenure', 'time', 'duration', 'last_login_days']
        for col in df.columns:
            if col.lower() in possible_time_cols and pd.api.types.is_numeric_dtype(df[col]):
                time_column = col
                break 
        
        if not time_column:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]) and col != churn_column:
                    time_column = col
                    break

        return JSONResponse(content={
            "churn_column": churn_column, 
            "time_column": time_column, 
            "file_path": file_path,
            "columns": list(df.columns),
            "shape": df.shape
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/run-eda/")
async def run_eda(payload: dict):
    try:
        file_data = base64.b64decode(payload["file"])
        temp_path = os.path.join(UPLOAD_DIR, "temp_eda.csv")
        with open(temp_path, "wb") as f:
            f.write(file_data)

        df = pd.read_csv(temp_path)
        churn_col = payload["churn_column"]
        time_col = payload.get("time_column")

        results = {
            "missing": get_missing_values(df),
            "distribution_plots": plot_distributions(df, churn_col),
            "correlation_plot": plot_correlation(df),
            "churn_by_feature_plots": plot_churn_by_feature(df, churn_col),
            "timeline_plot": plot_timeline(df, time_col, churn_col) if time_col else None
        }
        
        for key in results:
            if isinstance(results[key], str) and key.endswith('_plot'):
                results[key] = "/" + results[key].replace("\\", "/")
            elif isinstance(results[key], list) and key.endswith('_plots'):
                results[key] = ["/" + p.replace("\\", "/") for p in results[key]]
        
        return JSONResponse(content=results)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/run-survival/")
async def run_survival(payload: dict):
    try:
        file_data = base64.b64decode(payload["file"])
        temp_path = os.path.join(UPLOAD_DIR, "temp_survival.csv")
        with open(temp_path, "wb") as f:
            f.write(file_data)

        df = pd.read_csv(temp_path)
        churn_col = payload["churn_column"]
        time_col = payload["time_column"]

        if not time_col:
            return JSONResponse(content={"error": "Time column is required for survival analysis"}, status_code=400)

        km_plot = run_kaplan_meier(df, time_col, churn_col)
        cox_plot, cph = run_cox_model(df, time_col, churn_col)
        risk_scores = get_risk_scores(cph, df, time_col, churn_col)

        return JSONResponse(content={
            "kaplan_meier_plot": "/" + km_plot.replace("\\", "/"),
            "cox_model_plot": "/" + cox_plot.replace("\\", "/"),
            "risk_scores": risk_scores
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/run-ml/")
async def run_ml(payload: dict):
    try:
        file_data = base64.b64decode(payload["file"])
        temp_path = os.path.join(UPLOAD_DIR, "temp_ml.csv")
        with open(temp_path, "wb") as f:
            f.write(file_data)

        df = pd.read_csv(temp_path)
        churn_col = payload["churn_column"]

        ml_results = run_complete_ml_pipeline(df, churn_col)
        
        for key in ml_results.get('plots', {}):
            if ml_results['plots'].get(key):
                ml_results['plots'][key] = "/" + ml_results['plots'][key].replace("\\", "/")
        
        return JSONResponse(content=ml_results)
        
    except Exception as e:
        return JSONResponse(content={"error": f"ML Pipeline failed: {e}"}, status_code=500)


@app.post("/run-causal/")
async def run_causal(payload: dict):
    try:
        file_data = base64.b64decode(payload["file"])
        temp_path = os.path.join(UPLOAD_DIR, "temp_causal.csv")
        with open(temp_path, "wb") as f:
            f.write(file_data)

        df = pd.read_csv(temp_path)
        
        treatment = 'monthly_fee'
        outcome = payload["churn_column"]
        
        common_causes = [col for col in df.columns if col not in [treatment, outcome, 'customer_id', 'last_login_days']]
        
        causal_results = run_causal_analysis(df, treatment, outcome, common_causes)
        
        if "error" in causal_results:
             return JSONResponse(content=causal_results, status_code=500)

        return JSONResponse(content=causal_results)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/generate-report/")
async def generate_report(request: ReportRequest):
    try:
        corrected_plots = [p.lstrip('/') for p in request.plots if p]
        pdf_path = generate_pdf_report(corrected_plots)
        return JSONResponse(content={"pdf_url": "/" + pdf_path.replace("\\", "/")})
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to generate report: {e}"}, status_code=500)


@app.get("/train-rl/")
async def stream_rl_training():
    return StreamingResponse(train_rl_agent(), media_type="text/event-stream")

@app.post("/get-rl-recommendation/")
async def get_recommendation(user_state: UserState):
    recommendation = get_rl_recommendation(user_state.dict())
    return {"recommendation": recommendation}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
