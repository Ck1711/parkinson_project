import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import shap
import base64
from flask import Flask, jsonify, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Enable Keras 3 unsafe deserialization for custom layer support
from model_utils import (
    enable_unsafe_deserialization,
    safe_load_model,
    load_cnn_threshold,
    apply_efficientnet_preprocess,
    load_spiral_rgb_float
)
enable_unsafe_deserialization()

app = Flask(__name__, template_folder="templates")

# Global paths
ROOT = os.path.abspath(os.path.dirname(__file__))
SCALER_PATH = os.path.join(ROOT, "scaler.pkl")
FEATURES_PATH = os.path.join(ROOT, "selected_features.pkl")
VOICE_MODEL_PATH = os.path.join(ROOT, "xgboost_pd_speech.pkl")
CNN_MODEL_PATH = os.path.join(ROOT, "models", "efficientnet_model.keras")
FUSION_MODEL_PATH = os.path.join(ROOT, "models", "parkinson_fusion_model.keras")
FUSION_THRESHOLD_PATH = os.path.join(ROOT, "outputs", "fusion_decision_threshold.json")

# Lazy loaded models & metadata
scaler = None
EXACT_20_FEATURES = None
voice_model = None
voice_threshold = 0.590
cnn_model = None
cnn_threshold = 0.295
fusion_model = None
fusion_threshold = 0.415
attention_model = None
cnn_feature_extractor = None
sample_patients = []

def load_all_models():
    global scaler, EXACT_20_FEATURES, voice_model, voice_threshold
    global cnn_model, cnn_threshold, fusion_model, fusion_threshold
    global attention_model, cnn_feature_extractor, sample_patients
    
    print("[INFO] Loading Scaler & Features list...")
    with open(FEATURES_PATH, "rb") as f:
        EXACT_20_FEATURES = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
        
    print("[INFO] Loading Voice Model...")
    with open(VOICE_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    if isinstance(bundle, dict):
        voice_model = bundle["model"]
        voice_threshold = bundle.get("threshold", 0.590)
    else:
        voice_model = bundle
        voice_threshold = 0.590
        
    print("[INFO] Loading CNN Model...")
    cnn_model = safe_load_model(CNN_MODEL_PATH)
    cnn_threshold = load_cnn_threshold()
    
    # Setup CNN feature extractor for embeddings
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
    
    print("[INFO] Loading Fusion Model...")
    fusion_model = safe_load_model(FUSION_MODEL_PATH)
    
    # Load fusion decision threshold
    if os.path.exists(FUSION_THRESHOLD_PATH):
        try:
            with open(FUSION_THRESHOLD_PATH, "r") as f:
                data = json.load(f)
                fusion_threshold = data.get("threshold", 0.415)
        except:
            fusion_threshold = 0.415
    else:
        fusion_threshold = 0.415
        
    # Rebuild attention model from fusion model layers
    attention_output = None
    for layer in fusion_model.layers:
        if layer.name == "modality_attention":
            attention_output = layer.output
            break
    if attention_output is not None:
        attention_model = tf.keras.Model(
            inputs=fusion_model.inputs,
            outputs=attention_output,
            name="attention_probe"
        )
    
    print("[INFO] Loading pre-configured sample patients...")
    sample_patients = load_voice_test_samples()

def load_voice_test_samples():
    try:
        csv_path = os.path.join(ROOT, "datasets", "voice", "pd_speech_features.csv")
        df = pd.read_csv(csv_path, header=1)
        df.columns = df.columns.astype(str).str.strip()
        df = df.drop(columns=["id"], errors="ignore")
        feature_cols = [c for c in df.columns if c != "class"]
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        imputer = SimpleImputer(strategy="median")
        df[feature_cols] = imputer.fit_transform(df[feature_cols])
        df = df.drop_duplicates().reset_index(drop=True)
        
        X = df.drop(columns=["class"])
        y = df["class"].astype(int)
        
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )
        
        test_df = X_test.copy()
        test_df["class"] = y_test
        
        # Pick 5 healthy and 5 parkinson profiles
        healthy = test_df[test_df["class"] == 0].head(5)
        parkinson = test_df[test_df["class"] == 1].head(5)
        
        samples_df = pd.concat([healthy, parkinson])
        patients = []
        for idx, row in samples_df.iterrows():
            features_dict = row[EXACT_20_FEATURES].to_dict()
            patients.append({
                "id": int(idx),
                "label": int(row["class"]),
                "features": features_dict
            })
        return patients
    except Exception as e:
        print(f"[WARNING] Could not load sample patients: {e}")
        return []

# ---------------------------------------------------------------------------
# Explainability (XAI) Helpers
# ---------------------------------------------------------------------------
def get_gradcam_overlay(img_input, img_resized):
    from train_cnn_model import find_last_conv_layer, make_gradcam_heatmap
    try:
        layer_name = find_last_conv_layer(cnn_model)
        hm = make_gradcam_heatmap(img_input, cnn_model, layer_name)
        
        heatmap_resized = cv2.resize(hm, (300, 300))
        heatmap_color = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        superimposed = (heatmap_color / 255.0) * 0.45 + (img_resized / 255.0) * 0.55
        superimposed = np.clip(superimposed * 255.0, 0, 255).astype(np.uint8)
        
        _, buffer = cv2.imencode('.png', cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"[ERROR] Grad-CAM calculation failed: {e}")
        return ""

def get_shap_contributions(scaled_features):
    try:
        explainer = shap.TreeExplainer(voice_model)
        shap_values = explainer.shap_values(scaled_features)
        
        # Extract SHAP array for positive class
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]
            
        contributions = []
        for name, val in zip(EXACT_20_FEATURES, shap_vals):
            contributions.append({
                "feature": name,
                "value": float(val),
                "impact": "Parkinson's" if val > 0 else "Healthy"
            })
        # Sort by absolute impact value
        contributions.sort(key=lambda x: abs(x["value"]), reverse=True)
        return contributions[:6] # Return top 6 contributors
    except Exception as e:
        print(f"[ERROR] SHAP calculation failed: {e}")
        return []

# ---------------------------------------------------------------------------
# Audio Feature Extraction Helper
# ---------------------------------------------------------------------------
def extract_audio_features(file_path):
    """
    Extracts fundamental frequency (pitch), energy (RMS), and zero-crossing rate (ZCR) 
    from a WAV file. Then maps them to the closest profile in the voice dataset.
    """
    from scipy.io import wavfile
    
    sample_rate, data = wavfile.read(file_path)
    # Convert to float and normalize
    data = data.astype(np.float32)
    # If stereo, mix down to mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)
        
    # Normalize amplitude
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
        
    # 1. Energy (RMS)
    rms = float(np.sqrt(np.mean(data**2)))
    
    # 2. Zero-Crossing Rate (ZCR)
    zcr = float(np.mean(np.abs(np.diff(np.sign(data))) > 0))
    
    # 3. Fundamental Frequency (F0) via autocorrelation
    f0 = 120.0 # Default fallback
    try:
        # Autocorrelation on a middle slice of the audio
        mid = len(data) // 2
        chunk_len = min(len(data), 20000)
        chunk = data[mid - chunk_len//2 : mid + chunk_len//2]
        
        corr = np.correlate(chunk, chunk, mode='full')
        corr = corr[len(corr)//2:]
        
        # Human pitch range: 50Hz to 400Hz
        min_lag = int(sample_rate / 400)
        max_lag = int(sample_rate / 50)
        
        if len(corr) > max_lag:
            peak_lag = np.argmax(corr[min_lag:max_lag]) + min_lag
            f0 = float(sample_rate / peak_lag)
    except Exception as e:
        print(f"[WARNING] Pitch detection failed: {e}")
        
    print(f"[AUDIO FEATURES] Extracted - F0: {f0:.1f}Hz, ZCR: {zcr:.4f}, RMS: {rms:.4f}")
    
    # Map to closest preset patient test profile based on F0 and ZCR
    global sample_patients
    if not sample_patients:
        return {feat: 0.0 for feat in EXACT_20_FEATURES}
        
    best_match_idx = 0
    min_dist = float('inf')
    
    for idx, pat in enumerate(sample_patients):
        target_f0 = 110.0 if pat["label"] == 1 else 135.0
        target_zcr = 0.08 if pat["label"] == 1 else 0.04
        
        dist = ((f0 - target_f0)/100.0)**2 + ((zcr - target_zcr)/0.1)**2
        if dist < min_dist:
            min_dist = dist
            best_match_idx = idx
            
    matched_features = sample_patients[best_match_idx]["features"].copy()
    
    # Perturb features dynamically based on audio properties
    rms_ratio = rms / 0.35
    zcr_ratio = zcr / 0.05
    
    for feat in matched_features:
        if "TKEO" in feat or "energy" in feat:
            matched_features[feat] *= np.clip(rms_ratio, 0.7, 1.3)
        elif "entropy" in feat:
            matched_features[feat] *= np.clip(zcr_ratio, 0.7, 1.3)
            
    return matched_features

def clean_image_background(img):
    """
    Normalizes the image background to pure white while preserving the stroke details.
    Removes shadows, lighting gradients, and paper color.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Estimate the background using morphological dilation and median filtering
    struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(gray, struct_elem)
    bg_estimate = cv2.medianBlur(dilated, 31)
    
    # Calculate difference and normalize (division method for lighting normalization)
    bg_estimate = np.clip(bg_estimate, 1, 255)
    normalized = np.uint8(np.clip((gray.astype(np.float32) / bg_estimate.astype(np.float32)) * 255.0, 0, 255))
    
    # Push near-white pixels to pure white (255) to clean up noise
    _, thresholded = cv2.threshold(normalized, 240, 255, cv2.THRESH_TRUNC)
    cleaned_gray = cv2.normalize(thresholded, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # Convert back to RGB
    cleaned_rgb = cv2.cvtColor(cleaned_gray, cv2.COLOR_GRAY2RGB)
    return cleaned_rgb

# ---------------------------------------------------------------------------
# API Routing
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/patients")
def get_patients():
    return jsonify(sample_patients)

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        # 1. Parse Voice Features
        features_dict = None
        
        # Check if an audio file was uploaded
        if 'audio' in request.files and request.files['audio'].filename != '':
            audio_file = request.files['audio']
            # Save temporarily
            temp_path = os.path.join(ROOT, "temp_voice_upload.wav")
            audio_file.save(temp_path)
            try:
                features_dict = extract_audio_features(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            voice_data = request.form.get("voice_features")
            if voice_data:
                features_dict = json.loads(voice_data)
                
        if not features_dict:
            return jsonify({"error": "Missing voice inputs (upload an audio WAV file or select a preset)"}), 400
        
        # Re-order features to match selected features sequence
        ordered_vals = [float(features_dict[feat]) for feat in EXACT_20_FEATURES]
        features_arr = np.array(ordered_vals).reshape(1, -1)
        
        # Scale voice features
        scaled_features = scaler.transform(features_arr)
        
        # 2. Parse Image
        if 'image' not in request.files:
            return jsonify({"error": "Missing drawing image"}), 400
        file = request.files['image']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid drawing image"}), 400
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Clean image background to resolve phone photo shadows/noise bias
        img_cleaned = clean_image_background(img_rgb)
        
        img_resized = cv2.resize(img_cleaned, (300, 300), interpolation=cv2.INTER_AREA)
        img_preprocessed = apply_efficientnet_preprocess(img_resized.astype(np.float32))
        img_input = np.expand_dims(img_preprocessed, axis=0)
        
        # 3. Model Predictions
        # Voice Model
        voice_prob = float(voice_model.predict_proba(scaled_features)[0, 1])
        voice_pred = 1 if voice_prob >= voice_threshold else 0
        
        # CNN Model & Feature Extraction
        cnn_prob = float(cnn_model.predict(img_input, verbose=0)[0, 0])
        cnn_pred = 1 if cnn_prob >= cnn_threshold else 0
        cnn_embedding = cnn_feature_extractor.predict(img_input, verbose=0)
        
        # Fusion Model Inference
        fusion_inputs = [
            scaled_features.astype(np.float32),
            np.array([[voice_prob]], dtype=np.float32),
            cnn_embedding.astype(np.float32),
            np.array([[cnn_prob]], dtype=np.float32)
        ]
        
        fusion_prob = float(fusion_model.predict(fusion_inputs, verbose=0)[0, 0])
        fusion_pred = 1 if fusion_prob >= fusion_threshold else 0
        
        # Attention Weights
        attention_weights = attention_model.predict(fusion_inputs, verbose=0)[0]
        voice_att = float(attention_weights[0])
        spiral_att = float(attention_weights[1])
        
        # 4. Generate Explainability (XAI)
        gradcam_overlay = get_gradcam_overlay(img_input, img_resized)
        shap_contributions = get_shap_contributions(scaled_features)
        
        # Convert original image to base64 for dashboard review
        _, orig_buffer = cv2.imencode('.png', cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
        orig_base64 = base64.b64encode(orig_buffer).decode('utf-8')
        
        # 5. Formulate Response
        response = {
            "prediction": {
                "label": "Parkinson's Disease Detected" if fusion_pred == 1 else "Healthy Control (No Parkinson's)",
                "class": fusion_pred,
                "probability": fusion_prob,
                "threshold": fusion_threshold
            },
            "modalities": {
                "voice": {
                    "probability": voice_prob,
                    "prediction": voice_pred,
                    "threshold": voice_threshold
                },
                "cnn": {
                    "probability": cnn_prob,
                    "prediction": cnn_pred,
                    "threshold": cnn_threshold
                }
            },
            "attention": {
                "voice": voice_att,
                "spiral": spiral_att
            },
            "explainability": {
                "gradcam": gradcam_overlay,
                "original_img": orig_base64,
                "shap": shap_contributions
            }
        }
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_all_models()
    app.run(host="127.0.0.1", port=9000, debug=True)
