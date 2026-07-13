"""Router cho API dự đoán rung tâm nhĩ (AFib)."""

from fastapi import APIRouter, HTTPException

from app.schemas.health_data import (
    SensorDataRequest,
    PredictionResponse,
    HealthCheckResponse,
    HRVFeatures,
)
from app.services.preprocessing import apply_physiological_filter
from app.services.feature_engineering import extract_hrv_features
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
