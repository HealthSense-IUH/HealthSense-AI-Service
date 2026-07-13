import os
import joblib
import numpy as np
import pandas as pd

class PredictionService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_model_loaded = False
        self.model_version = "v1.0-ML-Final"
        self._load_model()

    def _load_model(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "random_forest_afib.pkl")
        scaler_path = os.path.join(base_dir, "models", "scaler.pkl")

        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_model_loaded = True
                print(f"[OK] Da tai thanh cong AI Model va Scaler ({self.model_version})")
            else:
                print("[ERROR] Khong tim thay file model. Can copy tu thu muc ML sang app/models/")
        except Exception as e:
            print(f"[ERROR] Loi khi tai mo hinh: {e}")

    def predict(self, features_dict: dict) -> tuple[str, float]:
        if not self.is_model_loaded:
            raise RuntimeError("Mô hình chưa được tải thành công. Vui lòng kiểm tra lại file .pkl.")

        feature_columns = [
            'mean_nni', 'sdnn', 'rmssd', 'pnn50', 
            'cvnni', 'cvsd', 'lf', 'hf', 
            'lf_hf_ratio', 'total_power'
        ]
        
        input_df = pd.DataFrame([features_dict], columns=feature_columns)
        input_scaled = self.scaler.transform(input_df)
        probabilities = self.model.predict_proba(input_scaled)[0]
        classes = self.model.classes_
        
        predicted_idx = np.argmax(probabilities)
        predicted_label = classes[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        return predicted_label, float(confidence)

prediction_service = PredictionService()
