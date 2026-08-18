import asyncio
import json
import logging
import io
import pandas as pd
import numpy as np
import httpx
import aio_pika
from app.config import settings
from app.services.s3_client import s3_client
from app.services.feature_engineering import extract_features_from_csv_data
from app.services.prediction import prediction_service

logger = logging.getLogger("uvicorn.error")


class RabbitMQConsumer:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.queue = None
        self._is_running = False

    async def start(self):
        """Kết nối RabbitMQ và bắt đầu lắng nghe hàng đợi."""
        try:
            logger.info(f"Connecting to RabbitMQ at {settings.RABBITMQ_URL}...")
            self.connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)
            self.channel = await self.connection.channel()
            
            # Khai báo queue (durable=True để khớp với cấu hình Queue(..., true) bên Spring Boot)
            self.queue = await self.channel.declare_queue(
                settings.RABBITMQ_QUEUE, durable=True
            )
            
            self._is_running = True
            logger.info(f"Connected to RabbitMQ! Listening on queue '{settings.RABBITMQ_QUEUE}'...")
            await self.queue.consume(self.on_message, no_ack=False)
        except Exception as e:
            logger.error(f"Failed to start RabbitMQ consumer: {str(e)}")
            # Không raise để ứng dụng vẫn khởi động được khi local chưa mở RabbitMQ

    async def stop(self):
        """Đóng kết nối RabbitMQ khi ứng dụng tắt."""
        self._is_running = False
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            logger.info("RabbitMQ connection closed.")

    async def on_message(self, message: aio_pika.abc.AbstractIncomingMessage):
        """Xử lý mỗi khi có tin nhắn mới từ hàng đợi."""
        async with message.process():
            try:
                payload_str = message.body.decode("utf-8")
                logger.info(f"Received message from RabbitMQ: {payload_str}")
                data = json.loads(payload_str)
                
                record_id = data.get("recordId")
                s3_key = data.get("s3Key")
                user_id = data.get("userId")
                
                if not record_id or not s3_key:
                    logger.error("Message missing required fields 'recordId' or 's3Key'. Discarding.")
                    return

                # Chạy tải S3 và tính toán trong thread pool để không block asyncio loop
                await asyncio.to_thread(self._process_record_sync, record_id, s3_key, user_id)
                
            except Exception as e:
                logger.error(f"Error processing message {message.body}: {str(e)}", exc_info=True)

    def _process_record_sync(self, record_id: int, s3_key: str, user_id: int):
        """Hàm đồng bộ chạy trong thread pool: tải S3, dự đoán AFib, gửi callback."""
        try:
            # 1. Tải file CSV từ S3
            logger.info(f"[Record {record_id}] Downloading CSV from S3 key: {s3_key}")
            content = s3_client.download_file_as_bytes(s3_key)
            
            # 2. Phân tích CSV
            df = pd.read_csv(io.BytesIO(content))
            time_col = 'Time(ms)' if 'Time(ms)' in df.columns else 'time'
            ppg_col = 'IR' if 'IR' in df.columns else 'ppg'
            
            if time_col not in df.columns or ppg_col not in df.columns:
                raise ValueError(f"File CSV thiếu cột '{time_col}' hoặc '{ppg_col}'")
                
            time_ms_array = np.asarray(df[time_col].values, dtype=float)
            ppg_array = np.asarray(df[ppg_col].values, dtype=float)
            
            if np.max(time_ms_array) < 100000 and np.mean(np.diff(time_ms_array)) < 1.0:
                time_ms_array = time_ms_array * 1000.0
                
            fs = 125.0
            if len(time_ms_array) >= 2:
                dt = np.mean(np.diff(time_ms_array[:min(100, len(time_ms_array))])) / 1000.0
                if dt > 0:
                    fs = 1.0 / dt

            # Bỏ 2 giây dữ liệu đầu tiên (2000ms) để loại bỏ nhiễu khởi động cảm biến / transient
            skip_ms = 2000.0
            if len(time_ms_array) > 0:
                start_time = time_ms_array[0]
                valid_mask = time_ms_array >= (start_time + skip_ms)
                if np.sum(valid_mask) >= 100:
                    time_ms_array = time_ms_array[valid_mask]
                    ppg_array = ppg_array[valid_mask]
                else:
                    cut_samples = int(2.0 * fs)
                    if len(time_ms_array) > cut_samples + 50:
                        time_ms_array = time_ms_array[cut_samples:]
                        ppg_array = ppg_array[cut_samples:]

            logger.info(f"[Record {record_id}] Extracting HRV features (fs={fs:.2f}Hz, samples after 2s trim={len(ppg_array)})...")
            features = extract_features_from_csv_data(time_ms_array, ppg_array, fs=fs)
            
            # 3. Chạy model dự đoán
            logger.info(f"[Record {record_id}] Running AFib prediction...")
            _, afib_probability = prediction_service.predict(features)
            
            if afib_probability < 0.30:
                callback_label = "NORMAL"
            elif afib_probability < 0.50:
                callback_label = "UNCERTAIN"
            elif afib_probability < 0.70:
                callback_label = "AFIB_SUSPECTED"
            else:
                callback_label = "AFIB"
                
            logger.info(f"[Record {record_id}] Prediction done: prob={afib_probability:.4f}, label={callback_label}")

            # 4. Gửi HTTP PATCH callback về Core Service
            callback_payload = {
                "recordId": record_id,
                "predictionLabel": callback_label,
                "confidence": round(afib_probability, 4),
                "hrvFeaturesJson": json.dumps(features)
            }
            
            logger.info(f"[Record {record_id}] Sending callback to Core Service: {settings.CORE_CALLBACK_URL}")
            with httpx.Client(timeout=10.0) as client:
                response = client.patch(settings.CORE_CALLBACK_URL, json=callback_payload)
                if response.status_code in [200, 204]:
                    logger.info(f"[Record {record_id}] Callback successful! Core Service updated.")
                else:
                    logger.error(f"[Record {record_id}] Callback failed with status {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"[Record {record_id}] Failed during processing or callback: {str(e)}", exc_info=True)
            try:
                fail_url = f"{settings.CORE_CALLBACK_URL}/fail"
                logger.info(f"[Record {record_id}] Sending fail callback to {fail_url}")
                with httpx.Client(timeout=10.0) as client:
                    resp = client.patch(fail_url, json={
                        "recordId": record_id,
                        "errorReason": str(e)
                    })
                    if resp.status_code in [200, 204]:
                        logger.info(f"[Record {record_id}] FAIL callback successful.")
                    else:
                        logger.error(f"[Record {record_id}] FAIL callback failed with status {resp.status_code}: {resp.text}")
            except Exception as cb_e:
                logger.error(f"[Record {record_id}] Could not send FAIL callback: {str(cb_e)}")


consumer = RabbitMQConsumer()
