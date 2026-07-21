from pydantic import BaseModel, Field
from typing import Optional


class SensorDataRequest(BaseModel):
    """Dữ liệu cảm biến gửi từ Web React hoặc ESP32.
    
    Thay vì gửi tín hiệu quang học khổng lồ, Client tự phát hiện đỉnh nhịp tim 
    và chỉ gửi mảng khoảng cách thời gian giữa các nhịp (tính bằng mili-giây).
    """
    rr_intervals: list[float] = Field(..., description="Mảng R-R intervals (tính bằng mili-giây)")


class HRVFeatures(BaseModel):
    """13 đặc trưng được trích xuất bằng toán học (Khớp 100% với mô hình Random Forest)."""
    
    HR_mean: Optional[float] = Field(None, description="Nhịp tim trung bình (BPM)")
    Mean_NN: Optional[float] = Field(None, description="Khoảng R-R trung bình (ms)")
    SDNN: Optional[float] = Field(None, description="Độ lệch chuẩn R-R (ms)")
    RMSSD: Optional[float] = Field(None, description="RMSSD (ms)")
    pNN50: Optional[float] = Field(None, description="Tỷ lệ cặp R-R chênh > 50ms (%)")
    NN50: Optional[float] = Field(None, description="Số lượng cặp R-R chênh > 50ms")
    CV: Optional[float] = Field(None, description="Hệ số biến thiên (%)")
    LF: Optional[float] = Field(None, description="Năng lượng phổ tần số thấp")
    HF: Optional[float] = Field(None, description="Năng lượng phổ tần số cao")
    LF_HF_Ratio: Optional[float] = Field(None, description="Tỷ lệ LF/HF")
    LF_norm: Optional[float] = Field(None, description="Năng lượng phổ LF chuẩn hóa (%)")
    HF_norm: Optional[float] = Field(None, description="Năng lượng phổ HF chuẩn hóa (%)")
    Total_Power: Optional[float] = Field(None, description="Tổng năng lượng phổ")


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
