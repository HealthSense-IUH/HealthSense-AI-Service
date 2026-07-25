"""Router cho API dự đoán rung tâm nhĩ (AFib)."""

import io
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File

from app.schemas.health_data import (
    SensorDataRequest,
    PredictionResponse,
    HealthCheckResponse,
    HRVFeatures,
)
from app.services.preprocessing import apply_physiological_filter
from app.services.feature_engineering import extract_hrv_features, extract_features_from_csv_data
from app.services.prediction import prediction_service

router = APIRouter(prefix="/api/v1", tags=["Prediction"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="ok",
        model_loaded=prediction_service.is_model_loaded,
        model_version=prediction_service.model_version,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: SensorDataRequest):
    if len(request.rr_intervals) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Dữ liệu quá ngắn ({len(request.rr_intervals)} nhịp). Cần tối thiểu 10 nhịp tim để phân tích.",
        )

    try:
        valid_rr = apply_physiological_filter(request.rr_intervals)

        if len(valid_rr) < 10:
            raise HTTPException(
                status_code=422,
                detail=f"Sau khi lọc nhiễu, chỉ còn {len(valid_rr)} nhịp tim hợp lệ. Cần tối thiểu 10 nhịp.",
            )

        features = extract_hrv_features(valid_rr)
        prediction_label, confidence = prediction_service.predict(features)

        return PredictionResponse(
            prediction=prediction_label,
            confidence=round(confidence, 4),
            features=HRVFeatures(**features),
            model_version=prediction_service.model_version,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi xử lý dữ liệu: {str(e)}",
        )

@router.post("/predict-csv", response_model=PredictionResponse)
async def predict_from_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Vui lòng tải lên file định dạng CSV.")
        
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Hỗ trợ cả Time(ms) và time, IR hoặc ppg
        time_col = 'Time(ms)' if 'Time(ms)' in df.columns else 'time'
        ppg_col = 'IR' if 'IR' in df.columns else 'ppg'
        
        if time_col not in df.columns or ppg_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"File CSV phải có cột thời gian và tín hiệu (VD: '{time_col}' và '{ppg_col}')")
            
        time_ms_array = df[time_col].values
        ppg_array = df[ppg_col].values
        
        # Nếu cột time nhỏ (ví dụ tính bằng giây), chuyển sang ms
        if np.max(time_ms_array) < 100000 and np.mean(np.diff(time_ms_array)) < 1.0:
            time_ms_array = time_ms_array * 1000.0
            
        # Tính tần số lấy mẫu (fs) tự động
        fs = 125.0 # Default cho MIMIC
        if len(time_ms_array) >= 2:
            dt = np.mean(np.diff(time_ms_array[:min(100, len(time_ms_array))])) / 1000.0
            if dt > 0:
                fs = 1.0 / dt
                
        # Trích xuất đặc trưng
        features = extract_features_from_csv_data(time_ms_array, ppg_array, fs=fs)
        
        # Gọi model
        prediction_label, confidence = prediction_service.predict(features)
        
        return PredictionResponse(
            prediction=prediction_label,
            confidence=round(confidence, 4),
            features=HRVFeatures(**features),
            model_version=prediction_service.model_version,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý dữ liệu: {str(e)}")
