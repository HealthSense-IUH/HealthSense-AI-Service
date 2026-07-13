from pydantic import BaseModel, Field
from typing import Optional


class SensorDataRequest(BaseModel):
    """Dữ liệu cảm biến gửi từ Web React hoặc ESP32.
    
    Thay vì gửi tín hiệu quang học khổng lồ, Client tự phát hiện đỉnh nhịp tim 
    và chỉ gửi mảng khoảng cách thời gian giữa các nhịp (tính bằng mili-giây).
    """
    rr_intervals: list[float] = Field(..., description="Mảng R-R intervals (tính bằng mili-giây)")


class HRVFeatures(BaseModel):
    """10 đặc trưng HRV được trích xuất bằng toán học (Khớp 100% với MIT-BIH)."""
    
    mean_nni: Optional[float] = Field(None, description="Khoảng R-R trung bình (ms)")
    sdnn: Optional[float] = Field(None, description="Độ lệch chuẩn R-R (ms)")
    rmssd: Optional[float] = Field(None, description="RMSSD (ms)")
    pnn50: Optional[float] = Field(None, description="Tỷ lệ cặp R-R chênh > 50ms (%)")
    cvnni: Optional[float] = Field(None, description="Hệ số biến thiên nhịp")
    cvsd: Optional[float] = Field(None, description="Hệ số biến thiên RMSSD")
    lf: Optional[float] = Field(None, description="Năng lượng phổ tần số thấp")
    hf: Optional[float] = Field(None, description="Năng lượng phổ tần số cao")
    lf_hf_ratio: Optional[float] = Field(None, description="Tỷ lệ LF/HF")
    total_power: Optional[float] = Field(None, description="Tổng năng lượng phổ")


class PredictionResponse(BaseModel):
    """Kết quả dự đoán trạng thái sức khỏe (Rung tâm nhĩ)."""
    prediction: str = Field(..., description="Trạng thái dự đoán (AFib hoặc Normal)")
    confidence: float = Field(..., description="Độ tin cậy (0.0 - 1.0)")
    features: HRVFeatures = Field(..., description="Các đặc trưng HRV đã tính")
    model_version: str = Field(default="dummy", description="Phiên bản model đang sử dụng")


class HealthCheckResponse(BaseModel):
    """Trạng thái hoạt động của server."""
    status: str = Field(default="ok")
    model_loaded: bool = Field(default=False, description="Model đã được load chưa")
    model_version: str = Field(default="none")
