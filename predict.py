import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import traceback
import numpy as np
import pandas as pd
import cv2
import pickle
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from model_utils import safe_load_model, apply_efficientnet_preprocess

# Paths (aligned with app.py)
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "selected_features.pkl"
VOICE_MODEL_PATH = "xgboost_pd_speech.pkl"
CNN_MODEL_PATH = os.path.join("models", "efficientnet_model.keras")
FUSION_MODEL_PATH = os.path.join("models", "parkinson_fusion_model.keras")
CNN_THRESHOLD_PATH = os.path.join("models", "cnn_decision_threshold.json")
FUSION_THRESHOLD_PATH = os.path.join("outputs", "fusion_decision_threshold.json")

# Global variables
scaler = None
EXACT_20_FEATURES = None
voice_model = None
voice_threshold = 0.590
cnn_model = None
cnn_threshold = 0.295
fusion_model = None
fusion_threshold = 0.415
cnn_feature_extractor = None

def load_all():
    global scaler, EXACT_20_FEATURES, voice_model, voice_threshold
    global cnn_model, cnn_threshold, fusion_model, fusion_threshold
    global cnn_feature_extractor

    if not os.path.exists(FEATURES_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(VOICE_MODEL_PATH):
        raise FileNotFoundError("Required model, scaler, or feature list files are missing from root directory.")

    with open(FEATURES_PATH, "rb") as f:
        EXACT_20_FEATURES = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
        
    with open(VOICE_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    if isinstance(bundle, dict):
        voice_model = bundle["model"]
        voice_threshold = bundle.get("threshold", 0.590)
    else:
        voice_model = bundle
        voice_threshold = 0.590

    cnn_model = safe_load_model(CNN_MODEL_PATH)
    if os.path.exists(CNN_THRESHOLD_PATH):
        try:
            with open(CNN_THRESHOLD_PATH, "r") as f:
                cnn_threshold = json.load(f).get("threshold", 0.295)
        except:
            cnn_threshold = 0.295
    else:
        cnn_threshold = 0.295

    # Setup CNN feature extractor
    try:
        pool_layer = cnn_model.get_layer("global_avg_pool")
        cnn_output = pool_layer.output
    except ValueError:
        cnn_output = cnn_model.layers[-2].output
    cnn_feature_extractor = tf.keras.models.Model(
        inputs=cnn_model.inputs,
        outputs=cnn_output,
        name="cnn_feature_extractor"
    )

    fusion_model = safe_load_model(FUSION_MODEL_PATH)
    if os.path.exists(FUSION_THRESHOLD_PATH):
        try:
            with open(FUSION_THRESHOLD_PATH, "r") as f:
                fusion_threshold = json.load(f).get("threshold", 0.415)
        except:
            fusion_threshold = 0.415
    else:
        fusion_threshold = 0.415

def prepare_image(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (300, 300), interpolation=cv2.INTER_AREA)
    img_preprocessed = apply_efficientnet_preprocess(img_resized.astype(np.float32))
    return np.expand_dims(img_preprocessed, axis=0), img_resized

def load_voice_row(csv_path: str, row_index: int):
    df = pd.read_csv(csv_path, header=1)
    df.columns = df.columns.astype(str).str.strip()
    df = df.drop(columns=["id", "ID", "Id"], errors="ignore")
    # Impute missing values with median if needed
    feature_cols = [c for c in df.columns if c != "class"]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.median(numeric_only=True))
    sample = df.iloc[[row_index]]
    return sample

def main():
    parser = argparse.ArgumentParser(description="Parkinson diagnostic CLI prediction tool.")
    parser.add_argument("--image", help="Path to a spiral drawing image file.")
    parser.add_argument("--voice_csv", help="Path to voice feature CSV file.")
    parser.add_argument("--voice_row", type=int, default=0, help="Row index for voice sample selection (default: 0).")
    parser.add_argument("--output_dir", default="outputs", help="Directory where outputs are saved.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        load_all()
    except Exception as exc:
        print(f"[ERROR] Could not initialize models/scalers: {exc}")
        traceback.print_exc()
        return

    if args.image is None and args.voice_csv is None:
        print("[ERROR] Please provide --image and/or --voice_csv.")
        return

    voice_prob = None
    voice_pred = None
    scaled_features = None

    if args.voice_csv is not None:
        try:
            sample_df = load_voice_row(args.voice_csv, args.voice_row)
            # Reorder features
            ordered_vals = [float(sample_df.iloc[0][feat]) for feat in EXACT_20_FEATURES]
            features_arr = np.array(ordered_vals).reshape(1, -1)
            scaled_features = scaler.transform(features_arr)
            voice_prob = float(voice_model.predict_proba(scaled_features)[0, 1])
            voice_pred = 1 if voice_prob >= voice_threshold else 0
            
            print(f"Voice Prediction: {'Parkinson' if voice_pred == 1 else 'Healthy'} (probability={voice_prob:.4f}, threshold={voice_threshold:.3f})")
        except Exception as e:
            print(f"[ERROR] Voice prediction failed: {e}")
            traceback.print_exc()
            return

    cnn_prob = None
    cnn_pred = None
    cnn_embedding = None
    img_input = None
    img_resized = None

    if args.image is not None:
        try:
            img_input, img_resized = prepare_image(args.image)
            cnn_prob = float(cnn_model.predict(img_input, verbose=0)[0, 0])
            cnn_pred = 1 if cnn_prob >= cnn_threshold else 0
            cnn_embedding = cnn_feature_extractor.predict(img_input, verbose=0)
            
            print(f"CNN Drawing Prediction: {'Parkinson' if cnn_pred == 1 else 'Healthy'} (probability={cnn_prob:.4f}, threshold={cnn_threshold:.3f})")
        except Exception as e:
            print(f"[ERROR] CNN prediction failed: {e}")
            traceback.print_exc()
            return

    if args.image is not None and args.voice_csv is not None:
        try:
            fusion_inputs = [
                scaled_features.astype(np.float32),
                np.array([[voice_prob]], dtype=np.float32),
                cnn_embedding.astype(np.float32),
                np.array([[cnn_prob]], dtype=np.float32)
            ]
            fusion_prob = float(fusion_model.predict(fusion_inputs, verbose=0)[0, 0])
            fusion_pred = 1 if fusion_prob >= fusion_threshold else 0
            print(f"Fusion Prediction: {'Parkinson' if fusion_pred == 1 else 'Healthy'} (probability={fusion_prob:.4f}, threshold={fusion_threshold:.3f})")
        except Exception as e:
            print(f"[ERROR] Fusion prediction failed: {e}")
            traceback.print_exc()
            return

    print("[SUCCESS] Diagnostics run completed.")

if __name__ == "__main__":
    main()
