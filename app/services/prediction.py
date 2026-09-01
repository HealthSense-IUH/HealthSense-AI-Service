import glob
import json
import logging
import os

import joblib
import pandas as pd

from app.config import settings
from app.services.hrv_v4 import HRV_FEATURE_NAMES

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
        """Danh sách đặc trưng mô hình yêu cầu. Trả về [] nếu không xác định được.

        KHÔNG đoán mò: trước đây hàm này rơi về một danh sách 13 đặc trưng mặc
        định khi không đọc được tên cột. Với một model lạ, làm vậy nghĩa là đưa
        sai đặc trưng vào mà không ai biết. Nay thà trả rỗng để `_validate`
        từ chối nạp còn hơn đoán.
        """
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

        return []

    def _read_card(self, model_filename: str) -> dict:
        """Thẻ model đi kèm: ưu tiên sidecar `<tên>.json`, sau đó `model_card.json`.

        `model_card.json` chỉ được dùng cho ĐÚNG model mà nó mô tả (model mặc
        định trong cấu hình) — nếu không sẽ gán nhầm phiên bản của model này
        cho model khác.
        """
        base = os.path.splitext(model_filename)[0]
        candidates = [f"{base}.json"]
        if model_filename == settings.MODEL_FILE:
            candidates.append("model_card.json")

        for name in candidates:
            path = os.path.join(self._models_dir, name)
            if not os.path.exists(path):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[CARD] Không đọc được thẻ model '{name}': {e}")
        return {}

    def _validate(self, model, filename: str) -> tuple[bool, str]:
        """Kiểm tra model có an toàn để phục vụ không. (ok, lý do nếu không).

        Hai điều kiện, đều rút ra từ sự cố có thật trong chính dự án này:

        1. PHẢI là sklearn Pipeline có bước tiền xử lý đi kèm.
           File `best_model_8165.pkl` (đời v3) từng nằm trong thư mục này là
           một MLPClassifier TRẦN, huấn luyện trên dữ liệu đã chuẩn hóa toàn
           cục từ trước. Nạp nó rồi đưa đặc trưng THÔ vào thì mạng nơ-ron nhận
           sai hoàn toàn thang đo — và vẫn trả về một xác suất trông hợp lý.
           Hỏng âm thầm, không có lỗi nào báo ra.

        2. PHẢI khai báo tên đặc trưng, và mọi đặc trưng đó service phải tính
           được. Không khai tên thì không có cách nào căn cột cho đúng.
        """
        if not (hasattr(model, "steps") and len(getattr(model, "steps", [])) >= 2):
            return False, (
                f"'{filename}' không phải sklearn Pipeline có bước tiền xử lý. "
                f"Model triển khai bắt buộc đóng gói scaler cùng bộ phân loại, "
                f"nếu không sẽ nhận sai thang đo mà không báo lỗi."
            )

        names = self._extract_feature_names(model)
        if not names:
            return False, f"'{filename}' không khai báo tên đặc trưng (feature_names_in_)."

        unknown = [n for n in names if n not in HRV_FEATURE_NAMES]
        if unknown:
            return False, (
                f"'{filename}' đòi các đặc trưng service không tính được: {unknown}. "
                f"Service chỉ sinh {len(HRV_FEATURE_NAMES)} đặc trưng HRV chuẩn."
            )
        return True, ""

    def load_model(self, model_filename: str) -> bool:
        """Nạp hoặc chuyển đổi mô hình đang hoạt động trong bộ nhớ.

        Model mới chỉ được thay vào SAU KHI qua `_validate`. Nếu không đạt,
        model đang chạy được giữ nguyên — một lần đổi model hỏng không được
        phép làm sập service đang phục vụ.
        """
        model_path = os.path.join(self._models_dir, model_filename)

        try:
            if not os.path.exists(model_path):
                logger.error(f"[ERROR] Không tìm thấy file mô hình tại: {model_path}")
                return self.is_model_loaded

            loaded = joblib.load(model_path)

            ok, reason = self._validate(loaded, model_filename)
            if not ok:
                logger.error(f"[REJECT] Từ chối nạp model: {reason}")
                if self.is_model_loaded:
                    logger.info(f"[KEEP] Giữ nguyên model đang chạy '{self.active_model_file}'.")
                return False

            base = os.path.splitext(model_filename)[0]
            # Phiên bản đọc từ thẻ model, KHÔNG hardcode. Trước đây dòng này là
            # f"v2.0-{base}" cố định, nên API luôn báo "v2.0" kể cả khi đang
            # chạy model 4.1.0 — ai gọi /api/health cũng tưởng service dùng
            # model đời v2.
            card = self._read_card(model_filename)
            version = str(card.get("version") or "").strip()

            self.model = loaded
            self.active_model_file = model_filename
            self.model_version = f"v{version}-{base}" if version else base
            self.expected_features = self._extract_feature_names(loaded)
            self.is_model_loaded = True

            logger.info(
                f"[OK] Đã nạp thành công AI Model '{model_filename}' "
                f"(phiên bản {self.model_version}, "
                f"yêu cầu {len(self.expected_features)} đặc trưng: {self.expected_features})"
            )
            return True
        except Exception as e:
            logger.error(f"[ERROR] Lỗi khi tải mô hình '{model_filename}': {e}", exc_info=True)
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
            is_active = fname == self.active_model_file and self.is_model_loaded

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

            result.append(
                {
                    "filename": fname,
                    "is_active": is_active,
                    "size_kb": size_kb,
                    "feature_count": len(feat_list) if feat_list else None,
                    "expected_features": feat_list,
                    "model_type": model_type,
                }
            )
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
        missing = []
        for col in feature_columns:
            if col not in features_dict:
                missing.append(col)
            aligned_data[col] = features_dict.get(col, 0.0)
        if missing:
            # Điền 0.0 âm thầm sẽ làm sai dự đoán mà không ai biết — phải cảnh báo to
            logger.warning(
                f"[PARITY] Thiếu đặc trưng {missing} so với model "
                f"'{self.active_model_file}' — đã điền 0.0, kết quả có thể sai!"
            )

        input_df = pd.DataFrame([aligned_data], columns=feature_columns)

        # Tính toán xác suất dự đoán
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(input_df)[0]
            classes = getattr(self.model, "classes_", [0, 1])

            # Extract probability specifically for AFib class
            afib_prob = 0.0
            found_afib = False
            for idx, c in enumerate(classes):
                c_str = str(c).strip().lower()
                if c_str in ("1", "1.0", "afib", "true"):
                    afib_prob = float(probabilities[idx])
                    found_afib = True
                    break

            # If we couldn't identify the AFib class, fallback to index 1 or max prob
            if not found_afib:
                if len(probabilities) > 1:
                    afib_prob = float(probabilities[1])
                else:
                    afib_prob = float(probabilities[0])
        else:
            # Mô hình không hỗ trợ predict_proba (chỉ trả về nhãn cứng)
            pred = self.model.predict(input_df)[0]
            pred_str = str(pred).strip().lower()
            if pred_str in ("1", "1.0", "afib", "true"):
                afib_prob = 1.0
            else:
                afib_prob = 0.0

        return "AFib_Probability", round(afib_prob, 4)


prediction_service = PredictionService()
