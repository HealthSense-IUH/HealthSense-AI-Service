import os
import glob
import logging
import joblib
import numpy as np
import pandas as pd
from app.config import settings

logger = logging.getLogger("uvicorn.error")


class PredictionService:
    """Service quản lý mô hình AI, nạp động và thực hiện suy luận (Inference)."""

    def __init__(self):
        self.model = None
        self.is_model_loaded = False
        self.active_model_file = settings.MODEL_FILE
        self.model_version = ""
        self.expected_features: list[str] = []
        self._models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )
        self.load_model(self.active_model_file)

    def _extract_feature_names(self, model) -> list[str]:
        """Tự động trích xuất danh sách đặc trưng mà mô hình yêu cầu."""
        # 1. Trực tiếp từ thuộc tính feature_names_in_
        if hasattr(model, "feature_names_in_"):
            return [str(f) for f in model.feature_names_in_]

        # 2. Nếu là Pipeline của scikit-learn, kiểm tra step đầu tiên hoặc cuối
        if hasattr(model, "steps") and len(model.steps) > 0:
            first_step = model.steps[0][1]
            if hasattr(first_step, "feature_names_in_"):
                return [str(f) for f in first_step.feature_names_in_]
            last_step = model.steps[-1][1]
            if hasattr(last_step, "feature_names_in_"):
                return [str(f) for f in last_step.feature_names_in_]

        # 3. Fallback mặc định 13 đặc trưng chuẩn cho MIMIC
        return [
            "HR_mean", "Mean_NN", "SDNN", "RMSSD", "pNN50", "NN50", "CV",
            "LF", "HF", "LF_HF_Ratio", "LF_norm", "HF_norm", "Total_Power"
        ]

    def load_model(self, model_filename: str) -> bool:
        """Nạp hoặc chuyển đổi mô hình đang hoạt động trong bộ nhớ."""
        model_path = os.path.join(self._models_dir, model_filename)

        try:
            if not os.path.exists(model_path):
                logger.error(f"[ERROR] Không tìm thấy file mô hình tại: {model_path}")
                self.is_model_loaded = False
                return False

            loaded = joblib.load(model_path)
            self.model = loaded
            self.active_model_file = model_filename
            self.model_version = f"v2.0-{os.path.splitext(model_filename)[0]}"
            self.expected_features = self._extract_feature_names(loaded)
            self.is_model_loaded = True

            logger.info(
                f"[OK] Đã nạp thành công AI Model '{model_filename}' "
                f"(Yêu cầu {len(self.expected_features)} đặc trưng: {self.expected_features})"
            )
            return True
        except Exception as e:
            logger.error(f"[ERROR] Lỗi khi tải mô hình '{model_filename}': {e}", exc_info=True)
            self.is_model_loaded = False
            return False

    def list_available_models(self) -> list[dict]:
        """Quét toàn bộ thư mục app/models và trả về danh sách kèm metadata."""
        os.makedirs(self._models_dir, exist_ok=True)
        files = glob.glob(os.path.join(self._models_dir, "*.pkl")) + glob.glob(
            os.path.join(self._models_dir, "*.joblib")
        )

        result = []
        for filepath in sorted(files):
            fname = os.path.basename(filepath)
            size_kb = round(os.path.getsize(filepath) / 1024.0, 2)
            is_active = (fname == self.active_model_file and self.is_model_loaded)

            feat_list = None
            model_type = None
            if is_active and self.model is not None:
                feat_list = self.expected_features
                model_type = type(self.model).__name__
            else:
                try:
                    tmp_m = joblib.load(filepath)
                    feat_list = self._extract_feature_names(tmp_m)
                    model_type = type(tmp_m).__name__
                except Exception:
                    pass

            result.append({
                "filename": fname,
                "is_active": is_active,
                "size_kb": size_kb,
                "feature_count": len(feat_list) if feat_list else None,
                "expected_features": feat_list,
                "model_type": model_type,
            })
        return result

    def predict(self, features_dict: dict) -> tuple[str, float]:
        """Thực hiện dự đoán an toàn, tự động căn chỉnh đúng đặc trưng của mô hình."""
        if not self.is_model_loaded or self.model is None:
            raise RuntimeError(
                f"Mô hình '{self.active_model_file}' chưa được tải thành công. Vui lòng kiểm tra lại file."
            )

        # Căn chỉnh thứ tự và danh sách đặc trưng khớp chính xác với mô hình
        feature_columns = self.expected_features
        aligned_data = {}
        for col in feature_columns:
            aligned_data[col] = features_dict.get(col, 0.0)

        input_df = pd.DataFrame([aligned_data], columns=feature_columns)

        # Tính toán xác suất dự đoán
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(input_df)[0]
            classes = getattr(self.model, "classes_", [0, 1])
            predicted_idx = int(np.argmax(probabilities))
            predicted_label = classes[predicted_idx]
            confidence = float(probabilities[predicted_idx])
        else:
            pred = self.model.predict(input_df)[0]
            predicted_label = pred
            confidence = 1.0

        # Chuẩn hóa nhãn văn bản hiển thị
        pred_str = str(predicted_label).strip().lower()
        if pred_str in ("0", "0.0", "normal", "false"):
            label_display = "Normal (Bình thường)"
        elif pred_str in ("1", "1.0", "afib", "true"):
            label_display = "AFib (Rung tâm nhĩ)"
        else:
            label_display = str(predicted_label)

        return label_display, round(confidence, 4)


prediction_service = PredictionService()
