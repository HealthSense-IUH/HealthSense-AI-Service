"""HealthSense AI Service - FastAPI Application.

Microservice chuyên phục vụ ML inference cho dự án HealthSense.
Nhận dữ liệu cảm biến PPG, trích xuất đặc trưng HRV, và trả về dự đoán trạng thái sức khỏe.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import prediction
from app.services.consumer import consumer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Khởi động RabbitMQ consumer
    await consumer.start()
    yield
    # Shutdown: Đóng kết nối RabbitMQ
    await consumer.stop()


app = FastAPI(
    title="HealthSense AI Service",
    description=(
        "Microservice phát hiện Rung Nhĩ (AFib) từ tín hiệu PPG (nhịp tim). "
        "Nhận dữ liệu cảm biến MAX30102, trích xuất 16 đặc trưng HRV (pipeline v4, "
        "có SQI kiểm soát chất lượng tín hiệu), và phân loại bằng model XGBoost "
        "đã kiểm định LOSO + cross-dataset (xem app/models/model_card.json)."
    ),
    version="0.3.0-v4-model",
    lifespan=lifespan,
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
        "version": "0.3.0-v4-model",
        "docs": "/docs",
    }
