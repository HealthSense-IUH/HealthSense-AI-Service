"""Service dự đoán trạng thái sức khỏe.

Load model đã train (model.pkl) và thực hiện inference.
Nếu chưa có model thật, sử dụng dummy model trả về kết quả giả.
"""

import os
import logging
from pathlib import Path

import numpy as np
import joblib

logger = logging.getLogger(__name__)

# Đường dẫn mặc định tới file model
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"


class PredictionService:
    """Service quản lý model và thực hiện dự đoán.

    Tương tự @Service trong Spring Boot.
    """

    def __init__(self):
        self.model = None
        self.model_version = "none"
        self._load_model()

    def _load_model(self):
        """Tải model từ file .pkl nếu tồn tại."""
        if MODEL_PATH.exists():
            try:
                self.model = joblib.load(MODEL_PATH)
                self.model_version = "v1.0"
                logger.info(f"Đã load model từ {MODEL_PATH}")
            except Exception as e:
                logger.error(f"Lỗi khi load model: {e}")
                self.model = None
                self.model_version = "error"
        else:
            logger.warning(
                f"Chưa có file model tại {MODEL_PATH}. "
                "Sử dụng dummy model."
            )
            self.model = None
            self.model_version = "dummy"

    @property
    def is_model_loaded(self) -> bool:
        """Kiểm tra model đã được load chưa."""
        return self.model is not None

    def predict(self, features: dict) -> tuple[str, float]:
        """Dự đoán trạng thái sức khỏe từ 16 features.

        Parameters:
            features: dict chứa 16 đặc trưng HRV

        Returns:
            tuple: (prediction_label, confidence)
        """
        if self.model is not None:
            return self._predict_real(features)
        else:
            return self._predict_dummy(features)

    def _predict_real(self, features: dict) -> tuple[str, float]:
        """Dự đoán bằng model thật (Random Forest)."""
        # Sắp xếp features theo đúng thứ tự model đã train
        feature_order = [
            "mean_nn", "sdnn", "rmssd", "pnn50", "nn50", "cv",
            "hr_mean", "lf", "hf", "lf_hf_ratio", "lf_norm", "hf_norm",
            "total_power", "spo2_mean", "spo2_std", "spo2_min",
        ]

        X = np.array([[features.get(f, 0) or 0 for f in feature_order]])

        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = float(np.max(probabilities))

        return str(prediction), confidence

    def _predict_dummy(self, features: dict) -> tuple[str, float]:
        """Dự đoán giả khi chưa có model thật.

        Sử dụng logic đơn giản dựa trên HR_mean để demo.
        """
        hr = features.get("hr_mean")

        if hr is None:
            return "Unknown", 0.0

        if hr < 80:
            return "Sitting", 0.75
        elif hr < 110:
            return "Walking", 0.65
        else:
            return "Running", 0.60


# Singleton instance (giống @Component trong Spring Boot)
prediction_service = PredictionService()
