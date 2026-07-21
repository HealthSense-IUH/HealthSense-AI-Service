import os
import joblib
import numpy as np
import pandas as pd

class PredictionService:
    def __init__(self):
        self.model = None
        self.is_model_loaded = False
        self.model_version = "v1.0.2-AFib-RF"
        self._load_model()

    def _load_model(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "afib_rf_model.pkl")

        try:
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                self.is_model_loaded = True
                print(f"[OK] Da tai thanh cong AI Model ({self.model_version}) tu {model_path}")
            else:
                print(f"[ERROR] Khong tim thay file model tai {model_path}. Can tai tu GitHub Release bo vao day.")
        except Exception as e:
            print(f"[ERROR] Loi khi tai mo hinh: {e}")

    def predict(self, features_dict: dict) -> tuple[str, float]:
        if not self.is_model_loaded:
            raise RuntimeError("Mô hình chưa được tải thành công. Vui lòng kiểm tra lại file .pkl.")

        # Phải khớp 100% với tên cột lúc huấn luyện bằng RandomForest
        feature_columns = [
            'HR_mean', 'Mean_NN', 'SDNN', 'RMSSD', 'pNN50', 'NN50', 'CV',
            'LF', 'HF', 'LF_HF_Ratio', 'LF_norm', 'HF_norm', 'Total_Power'
        ]
        
        input_df = pd.DataFrame([features_dict], columns=feature_columns)
        
        # Mô hình RandomForest mới không dùng scaler, có thể feed thẳng dataframe
        probabilities = self.model.predict_proba(input_df)[0]
        classes = self.model.classes_
        
        predicted_idx = np.argmax(probabilities)
        predicted_label = classes[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        # Ánh xạ từ nhãn 0/1 sang văn bản dễ đọc nếu cần (Mô hình trả về int 0/1)
        if str(predicted_label) == '0':
            predicted_label_str = "Normal (Bình thường)"
        elif str(predicted_label) == '1':
            predicted_label_str = "AFib (Rung tâm nhĩ)"
        else:
            predicted_label_str = str(predicted_label)
            
        return predicted_label_str, float(confidence)

prediction_service = PredictionService()
