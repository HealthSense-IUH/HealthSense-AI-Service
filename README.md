# HealthSense AI Service

## Tiếng Việt

**HealthSense AI Service** là microservice dự đoán trạng thái sức khỏe từ tín hiệu PPG (nhịp tim), được phát triển trên nền tảng FastAPI (Python).

### Chức năng chính
- Nhận dữ liệu cảm biến PPG thô từ ESP32 hoặc Spring Boot Backend (REST API + RabbitMQ/S3).
- Xử lý tín hiệu pipeline v4 (đồng bộ 100% với repo HealthSense-ML, có parity test): Butterworth bandpass 0.5–8 Hz, dò nhịp theo prominence, lọc sinh lý NN 250–2000 ms.
- Trích xuất 16 đặc trưng HRV chuẩn Task Force 1996 (SampEn thật, phổ Welch) + **SQI** — tín hiệu kém trả "chất lượng không đủ" thay vì đoán bừa.
- Phát hiện Rung Nhĩ (AFib) bằng model **`healthsense_afib_pipeline.pkl`** (XGBoost + Scaler, huấn luyện 60 bệnh nhân MIMIC + AFDB, kiểm định LOSO + cross-dataset không data leakage — chi tiết: `app/models/model_card.json`).

### Công nghệ
- **Framework:** FastAPI 0.115+
- **Ngôn ngữ:** Python 3.12+
- **Thư viện xử lý tín hiệu:** SciPy, NumPy
- **Thư viện ML:** Scikit-learn, XGBoost, Joblib
- **Validation:** Pydantic v2

### Cấu trúc dự án
Dự án được tổ chức theo mô hình Router-Schema-Service (tương tự Controller-DTO-Service trong Spring Boot):
- `routers/prediction.py`: Định nghĩa các API endpoint (POST /predict, GET /health).
- `schemas/health_data.py`: Định nghĩa cấu trúc dữ liệu Request/Response (Pydantic).
- `services/hrv_v4.py`: Xử lý tín hiệu + 16 đặc trưng HRV + SQI (bản đồng bộ với ML Lab — sửa phải chạy `tests/parity_check.py`).
- `services/feature_engineering.py`: Trích xuất 16 đặc trưng HRV (SDNN, RMSSD, LF/HF, ...).
- `services/prediction.py`: Load model và thực hiện dự đoán. Model được **kiểm tra trước khi nạp**: phải là sklearn Pipeline có bước tiền xử lý đi kèm, và chỉ đòi những đặc trưng service tính được. Không đạt thì bị từ chối và model đang chạy giữ nguyên.

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
| GET | `/api/health` | Kiểm tra trạng thái server, model đang dùng và số đặc trưng |
| GET | `/api/models` | Liệt kê toàn bộ các file model (.pkl, .joblib) có trong `app/models/` kèm metadata |
| POST | `/api/models/active` | Chuyển đổi nóng model AI đang chạy mà không cần restart server (model mới phải qua kiểm tra, xem bên dưới) |

> **Chỉ có một model được triển khai: `healthsense_afib_pipeline.pkl`.**
> Trước đây thư mục `app/models/` còn giữ 2 model đời cũ — `mimic_afib_pipeline.pkl`
> (đời v1/v2) và `best_model_8165.pkl` (đời v3, huấn luyện trên pipeline bị
> data leakage). Cả hai đã được gỡ.
>
> `best_model_8165.pkl` đặc biệt nguy hiểm: nó là `MLPClassifier` **trần, không
> kèm scaler**, huấn luyện trên dữ liệu đã chuẩn hóa toàn cục từ trước. Nạp nó
> rồi đưa đặc trưng thô vào thì kết quả sai hoàn toàn **mà không có lỗi nào báo
> ra**. Vì vậy `load_model()` nay từ chối mọi model không đóng gói tiền xử lý
> bên trong Pipeline.
>
> Lấy lại 2 file cũ từ lịch sử git nếu cần đối chiếu: `git checkout 2e25aa9 -- app/models`
| POST | `/api/predict` | Nhận mảng RR intervals, trả về dự đoán AFib |
| POST | `/api/predict-csv` | Tải lên file CSV chứa tín hiệu PPG thô để dự đoán |

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
