"""HealthSense AI Service - FastAPI Application.

Microservice chuyên phục vụ ML inference cho dự án HealthSense.
Nhận dữ liệu cảm biến PPG, trích xuất đặc trưng HRV, và trả về dự đoán trạng thái sức khỏe.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import prediction

app = FastAPI(
    title="HealthSense AI Service",
    description=(
        "Microservice dự đoán trạng thái sức khỏe từ tín hiệu PPG (nhịp tim). "
        "Nhận dữ liệu cảm biến MAX30102, trích xuất 16 đặc trưng HRV, "
        "và sử dụng model Random Forest để phân loại trạng thái."
    ),
    version="0.1.0",
)

# CORS - cho phép Spring Boot Backend và ESP32 gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký Router (tương tự @ComponentScan trong Spring Boot)
app.include_router(prediction.router)


@app.get("/", tags=["Root"])
async def root():
    """Trang chủ API."""
    return {
        "service": "HealthSense AI Service",
        "version": "0.1.0",
        "docs": "/docs",
    }
