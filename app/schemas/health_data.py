from pydantic import BaseModel, Field
from typing import Optional


class SensorDataRequest(BaseModel):
    """Dữ liệu cảm biến gửi từ ESP32 hoặc Spring Boot Backend.

    Chứa mảng tín hiệu PPG thô thu thập trong một khoảng thời gian
    (tối thiểu 5 phút để tính HRV theo chuẩn Task Force 1996).
    """
    ir: list[float] = Field(..., description="Mảng giá trị IR từ MAX30102")
    red: list[float] = Field(..., description="Mảng giá trị RED từ MAX30102")
    spo2: list[float] = Field(..., description="Mảng giá trị SpO2")
    timestamps: list[float] = Field(..., description="Mảng timestamp (ms)")
    sampling_rate: float = Field(default=100.0, description="Tần số lấy mẫu (Hz)")


class HRVFeatures(BaseModel):
    """16 đặc trưng HRV được trích xuất từ tín hiệu PPG."""

    # Time-Domain (6 features)
    mean_nn: Optional[float] = Field(None, description="Khoảng R-R trung bình (ms)")
    sdnn: Optional[float] = Field(None, description="Độ lệch chuẩn R-R (ms)")
    rmssd: Optional[float] = Field(None, description="RMSSD (ms)")
    pnn50: Optional[float] = Field(None, description="Tỷ lệ cặp R-R chênh > 50ms (%)")
    nn50: Optional[int] = Field(None, description="Số cặp R-R chênh > 50ms")
    cv: Optional[float] = Field(None, description="Hệ số biến thiên (%)")

    # Heart Rate (1 feature)
    hr_mean: Optional[float] = Field(None, description="Nhịp tim trung bình (BPM)")

    # Frequency-Domain (6 features)
    lf: Optional[float] = Field(None, description="Năng lượng phổ tần số thấp (ms2)")
    hf: Optional[float] = Field(None, description="Năng lượng phổ tần số cao (ms2)")
    lf_hf_ratio: Optional[float] = Field(None, description="Tỷ lệ LF/HF")
    lf_norm: Optional[float] = Field(None, description="LF chuẩn hóa (%)")
    hf_norm: Optional[float] = Field(None, description="HF chuẩn hóa (%)")
    total_power: Optional[float] = Field(None, description="Tổng năng lượng phổ (ms2)")

    # SpO2 (3 features)
    spo2_mean: Optional[float] = Field(None, description="SpO2 trung bình (%)")
    spo2_std: Optional[float] = Field(None, description="Độ dao động SpO2")
    spo2_min: Optional[float] = Field(None, description="SpO2 thấp nhất (%)")


class PredictionResponse(BaseModel):
    """Kết quả dự đoán trạng thái sức khỏe."""
    prediction: str = Field(..., description="Trạng thái dự đoán (Sitting, Walking, ...)")
    confidence: float = Field(..., description="Độ tin cậy (0.0 - 1.0)")
    features: HRVFeatures = Field(..., description="Các đặc trưng HRV đã tính")
    model_version: str = Field(default="dummy", description="Phiên bản model đang sử dụng")


class HealthCheckResponse(BaseModel):
    """Trạng thái hoạt động của server."""
    status: str = Field(default="ok")
    model_loaded: bool = Field(default=False, description="Model đã được load chưa")
    model_version: str = Field(default="none")
