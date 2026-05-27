import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import traceback
import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

BASE_DIR = 'datasets'
VOICE_CSV_PATH = os.path.join(BASE_DIR, 'voice', 'current', 'pd_speech_features.csv')
XGB_MODEL_PATH = os.path.join('models', 'voice_xgb_model.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')
SELECTOR_PATH = os.path.join('models', 'feature_selector.pkl')
CNN_MODEL_PATH = os.path.join('models', 'efficientnet_model.keras')
FUSION_MODEL_PATH = os.path.join('models', 'parkinson_fusion_model.keras')


def enable_unsafe_deserialization():
    if hasattr(tf.keras.config, 'enable_unsafe_deserialization'):
        try:
            tf.keras.config.enable_unsafe_deserialization()
            return
        except Exception as ex:
            print(f"[WARNING] enable_unsafe_deserialization failed: {ex}")
    if hasattr(tf.keras.utils, 'enable_unsafe_deserialization'):
        try:
            tf.keras.utils.enable_unsafe_deserialization()
        except Exception as ex:
            print(f"[WARNING] enable_unsafe_deserialization failed: {ex}")


def safe_load_model(path):
    enable_unsafe_deserialization()
    try:
        return tf.keras.models.load_model(path, compile=False, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(path, compile=False)


def load_voice_dataframe(csv_path: str):
    df = pd.read_csv(csv_path, header=1)
    id_columns = ['id', 'ID', 'Id']
    df = df.drop(columns=[col for col in id_columns if col in df.columns], errors='ignore')
    df = df.select_dtypes(include=[np.number])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(df.mean())
    return df


def infer_target_column(df: pd.DataFrame):
    target_candidates = ['class', 'status', 'diagnosis']
    for col in df.columns:
        if col.lower() in target_candidates:
            return col
    return None


def prepare_voice_features(df: pd.DataFrame, selector, scaler):
    from patient_data import resolve_voice_feature_columns, transform_voice_inference

    pruned_path = os.path.join('models', 'voice_pruned_columns.pkl')
    impute_path = os.path.join('models', 'voice_impute_means.pkl')
    if not os.path.isfile(pruned_path):
        raise FileNotFoundError(
            'voice_pruned_columns.pkl missing — retrain with train_dl_model.py'
        )
    pruned_cols = joblib.load(pruned_path)
    impute_means = joblib.load(impute_path) if os.path.isfile(impute_path) else None

    numeric = df.select_dtypes(include=[np.number]).copy()
    missing = [c for c in pruned_cols if c not in numeric.columns]
    if missing:
        raise ValueError(f'Voice sample missing {len(missing)} pruned feature column(s).')

    x_voice = transform_voice_inference(
        numeric, pruned_cols, scaler, selector, impute_means
    )
    feature_cols = resolve_voice_feature_columns(numeric, 'id', 'class')
    return x_voice, feature_cols


def load_selector_and_scaler():
    if not os.path.exists(SELECTOR_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError('Required selector or scaler files are missing.')
    selector = joblib.load(SELECTOR_PATH)
    scaler = joblib.load(SCALER_PATH)
    return selector, scaler


def load_voice_decision_threshold(default: float = 0.5) -> float:
    path = os.path.join('models', 'voice_decision_threshold.json')
    if os.path.isfile(path):
        import json
        with open(path, encoding='utf-8') as f:
            return float(json.load(f).get('threshold', default))
    return default


def prepare_image(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Image file not found: {image_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if hasattr(layer, 'output_shape') and isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = tf.reduce_sum(last_conv_layer_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def save_gradcam_overlay(img_array, model, save_path, title=None):
    last_conv_layer_name = find_last_conv_layer(model)
    if last_conv_layer_name is None:
        raise ValueError('No Conv2D layer found for Grad-CAM.')

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    img = img_array[0].astype(np.uint8)
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.4)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def predict_voice(sample: pd.DataFrame):
    selector, scaler = load_selector_and_scaler()
    x_voice, selected_columns = prepare_voice_features(sample, selector, scaler)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    threshold = load_voice_decision_threshold()
    probabilities = xgb_model.predict_proba(x_voice)[:, 1]
    labels = (probabilities >= threshold).astype(int)
    return probabilities, labels, selected_columns, threshold


def predict_image(image_path: str, cnn_model):
    img = prepare_image(image_path)
    prediction = cnn_model.predict(img, verbose=0).flatten()[0]
    return prediction, img


def predict_fusion(sample: pd.DataFrame, image_path: str, cnn_model, fusion_model):
    from model_utils import extract_single_cnn_features

    selector, scaler = load_selector_and_scaler()
    x_voice, _ = prepare_voice_features(sample, selector, scaler)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    voice_prob = xgb_model.predict_proba(x_voice)[:, 1].reshape(-1, 1)
    cnn_embed, cnn_prob = extract_single_cnn_features(cnn_model, image_path)
    fusion_prob = fusion_model.predict(
        [x_voice, voice_prob, cnn_embed, cnn_prob], verbose=0
    ).flatten()[0]
    return fusion_prob, voice_prob[0, 0], cnn_embed


def load_models():
    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError(f'CNN model not found: {CNN_MODEL_PATH}')
    if not os.path.exists(FUSION_MODEL_PATH):
        raise FileNotFoundError(f'Fusion model not found: {FUSION_MODEL_PATH}')

    cnn_model = safe_load_model(CNN_MODEL_PATH)
    fusion_model = safe_load_model(FUSION_MODEL_PATH)
    return cnn_model, fusion_model


def main():
    parser = argparse.ArgumentParser(description='Parkinson prediction utility for voice, spiral image, and fusion models.')
    parser.add_argument('--image', help='Path to a spiral drawing image file.')
    parser.add_argument('--voice_csv', help='Path to voice feature CSV file.')
    parser.add_argument('--voice_row', type=int, default=0, help='Row index for voice sample selection when using a CSV file.')
    parser.add_argument('--output_dir', default='outputs', help='Directory where output visualizations are saved.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        cnn_model, fusion_model = load_models()
    except Exception as exc:
        print(f'[ERROR] Could not load models: {exc}')
        traceback.print_exc()
        return

    if args.voice_csv is None and args.image is None:
        print('[ERROR] Please provide --image and/or --voice_csv to make predictions.')
        return

    if args.voice_csv is not None:
        voice_df = load_voice_dataframe(args.voice_csv)
        if voice_df.shape[0] == 0:
            print('[ERROR] Voice CSV did not contain valid numeric feature rows.')
            return
        if args.voice_row >= len(voice_df):
            print(f'[ERROR] voice_row {args.voice_row} is out of range. CSV has {len(voice_df)} rows.')
            return
        sample = voice_df.iloc[[args.voice_row]]
        voice_prob, voice_labels, selected_columns, threshold = predict_voice(sample)
        print(f'Voice prediction probability (Parkinson): {voice_prob[0]:.4f}')
        print(f'Voice label (threshold={threshold:.3f}): {int(voice_labels[0])}')
        print(f'Selected voice feature columns used: {selected_columns}')

    if args.image is not None:
        image_prob, image_tensor = predict_image(args.image, cnn_model)
        print(f'CNN image prediction probability (Parkinson): {image_prob:.4f}')

    if args.image is not None and args.voice_csv is not None:
        fusion_prob, voice_score, image_tensor = predict_fusion(sample, args.image, cnn_model, fusion_model)
        print(f'Fusion prediction probability (Parkinson): {fusion_prob:.4f}')
        print(f'Voice probability input to fusion: {voice_score:.4f}')

        gradcam_path = os.path.join(args.output_dir, 'predict_gradcam.png')
        save_gradcam_overlay(image_tensor, cnn_model, gradcam_path, title=f'Fusion prediction {fusion_prob:.4f}')
        print(f'Grad-CAM saved to {gradcam_path}')

    print('[SUCCESS] Prediction completed.')


if __name__ == '__main__':
    main()
