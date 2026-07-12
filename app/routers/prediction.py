"""Router cho API dự đoán sức khỏe.

Tương tự @RestController trong Spring Boot.
"""

import numpy as np
from fastapi import APIRouter, HTTPException

from app.schemas.health_data import (
    SensorDataRequest,
    PredictionResponse,
    HealthCheckResponse,
    HRVFeatures,
)
from app.services.preprocessing import preprocess_signal, detect_peaks
from app.services.feature_engineering import calculate_rr_intervals, extract_all_features
from app.services.prediction import prediction_service

router = APIRouter(prefix="/api/v1", tags=["Prediction"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Kiểm tra trạng thái hoạt động của server.

    Tương tự @GetMapping("/health") trong Spring Boot.
    """
    return HealthCheckResponse(
        status="ok",
        model_loaded=prediction_service.is_model_loaded,
        model_version=prediction_service.model_version,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: SensorDataRequest):
    """Nhận dữ liệu cảm biến, trích xuất đặc trưng và trả về dự đoán.

    Tương tự @PostMapping("/predict") trong Spring Boot.

    Quy trình:
        1. Nhận mảng IR, RED, SpO2, timestamps từ request
        2. Tiền xử lý: Bandpass Filter
        3. Phát hiện đỉnh nhịp tim
        4. Tính R-R Intervals
        5. Trích xuất 16 features HRV
        6. Gọi model dự đoán
        7. Trả về kết quả
    """
    # Validate dữ liệu đầu vào
    if len(request.ir) < 3000:  # Tối thiểu 30 giây @ 100Hz
        raise HTTPException(
            status_code=400,
            detail=f"Dữ liệu quá ngắn ({len(request.ir)} samples). "
                   f"Cần tối thiểu 3000 samples (30 giây @ 100Hz).",
        )

    if len(request.ir) != len(request.timestamps):
        raise HTTPException(
            status_code=400,
            detail="Số lượng phần tử trong 'ir' và 'timestamps' phải bằng nhau.",
        )

    try:
        ir_data = np.array(request.ir)
        spo2_data = np.array(request.spo2)
        timestamps = np.array(request.timestamps)

        # 1. Tiền xử lý
        ir_filtered = preprocess_signal(ir_data, request.sampling_rate)

        # 2. Phát hiện đỉnh
        peaks = detect_peaks(ir_filtered, request.sampling_rate)

        if len(peaks) < 5:
            raise HTTPException(
                status_code=422,
                detail="Không phát hiện đủ nhịp tim trong dữ liệu. "
                       "Vui lòng kiểm tra lại cảm biến.",
            )

        # 3. Tính R-R Intervals
        rr_intervals = calculate_rr_intervals(peaks, timestamps)

        if len(rr_intervals) < 10:
            raise HTTPException(
                status_code=422,
                detail=f"Chỉ phát hiện {len(rr_intervals)} R-R Intervals hợp lệ. "
                       f"Cần tối thiểu 10.",
            )

        # 4. Trích xuất 16 features
        features = extract_all_features(rr_intervals, spo2_data)

        # 5. Dự đoán
        prediction, confidence = prediction_service.predict(features)

        # 6. Trả về kết quả
        return PredictionResponse(
            prediction=prediction,
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
