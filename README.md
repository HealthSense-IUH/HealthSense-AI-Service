# HealthSense AI Service

## Tiếng Việt

**HealthSense AI Service** là microservice dự đoán trạng thái sức khỏe từ tín hiệu PPG (nhịp tim), được phát triển trên nền tảng FastAPI (Python).

### Chức năng chính
- Nhận dữ liệu cảm biến PPG thô từ ESP32 hoặc Spring Boot Backend qua REST API.
- Tiền xử lý tín hiệu: loại bỏ Baseline Wander bằng Butterworth Bandpass Filter.
- Trích xuất 16 đặc trưng HRV (Heart Rate Variability) theo chuẩn Task Force 1996.
- Dự đoán trạng thái sức khỏe (Sitting, Walking, ...) bằng model Random Forest.

### Công nghệ
- **Framework:** FastAPI 0.115+
- **Ngôn ngữ:** Python 3.12+
- **Thư viện xử lý tín hiệu:** SciPy, NumPy
- **Thư viện ML:** Scikit-learn, Joblib
- **Validation:** Pydantic v2

### Cấu trúc dự án
Dự án được tổ chức theo mô hình Router-Schema-Service (tương tự Controller-DTO-Service trong Spring Boot):
- `routers/prediction.py`: Định nghĩa các API endpoint (POST /predict, GET /health).
- `schemas/health_data.py`: Định nghĩa cấu trúc dữ liệu Request/Response (Pydantic).
- `services/preprocessing.py`: Xử lý lọc tín hiệu PPG (Bandpass Filter, Peak Detection).
- `services/feature_engineering.py`: Trích xuất 16 đặc trưng HRV (SDNN, RMSSD, LF/HF, ...).
- `services/prediction.py`: Load model và thực hiện dự đoán.

### Cài đặt và Sử dụng
1. Tạo môi trường ảo và cài đặt thư viện:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Chạy server:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
3. Mở Swagger UI tại `http://localhost:8000/docs` để xem và test API.

### API Endpoints

| Method | Path | Mô tả |
|--------|------|-------|
| GET | `/` | Trang chủ |
| GET | `/api/v1/health` | Kiểm tra trạng thái server |
| POST | `/api/v1/predict` | Nhận dữ liệu cảm biến, trả về dự đoán |

---

## English

**HealthSense AI Service** is a microservice that predicts health status from PPG (Photoplethysmography) signals, built with FastAPI (Python).

### Key Features
- Receives raw PPG sensor data from ESP32 or Spring Boot Backend via REST API.
- Signal preprocessing: removes Baseline Wander using Butterworth Bandpass Filter.
- Extracts 16 HRV (Heart Rate Variability) features following the Task Force 1996 standard.
- Predicts health status (Sitting, Walking, ...) using a Random Forest model.

### Tech Stack
- **Framework:** FastAPI 0.115+
- **Language:** Python 3.12+
- **Signal Processing:** SciPy, NumPy
- **Machine Learning:** Scikit-learn, Joblib
- **Validation:** Pydantic v2

### Project Structure
Organized following the Router-Schema-Service pattern (similar to Controller-DTO-Service in Spring Boot):
- `routers/prediction.py`: API endpoint definitions (POST /predict, GET /health).
- `schemas/health_data.py`: Request/Response data structures (Pydantic models).
- `services/preprocessing.py`: PPG signal filtering (Bandpass Filter, Peak Detection).
- `services/feature_engineering.py`: 16 HRV feature extraction (SDNN, RMSSD, LF/HF, ...).
- `services/prediction.py`: Model loading and inference.

### Installation and Usage
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
3. Open Swagger UI at `http://localhost:8000/docs` to explore and test the API.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root page |
| GET | `/api/v1/health` | Server health check |
| POST | `/api/v1/predict` | Receive sensor data, return prediction |
