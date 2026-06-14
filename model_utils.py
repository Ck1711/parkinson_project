"""
Shared utilities for Parkinson multimodal detection (TensorFlow 2.17+, tf.keras only).
EfficientNetB0 @ 224x224 + adaptive attention fusion helpers.
 
FIXES APPLIED:
  1. build_late_fusion_model: reduced dropout from 0.5/0.45 to 0.25/0.2 in fusion head.
     Heavy dropout on a small paired dataset caused underfitting.
  2. build_late_fusion_model: raised Adam lr from 5e-5 to 1e-4 for faster convergence.
  3. build_advanced_augmentation: kept mild transforms only (rotation, zoom, translation).
     Colour/contrast augmentation can destroy diagnostic signal in spiral drawings.
"""
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    Concatenate,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
)
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
 
# ---------------------------------------------------------------------------
# Global image / model paths (300x300 everywhere)
# ---------------------------------------------------------------------------
IMG_SIZE = 300
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
 
EFFICIENTNET_MODEL_PATH = os.path.join("models", "efficientnet_model.keras")
FUSION_MODEL_PATH = os.path.join("models", "parkinson_fusion_model.keras")
HISTORY_DIR = os.path.join("outputs", "history")
CNN_BEST_THRESHOLD_JSON = os.path.join("outputs", "cnn_best_threshold.json")
CNN_THRESHOLD_LEGACY_JSON = os.path.join("models", "cnn_decision_threshold.json")
CNN_INFERENCE_BATCH_SIZE = 8
 
LEGACY_CNN_PATHS = [
    os.path.join("models", "parkinson_cnn_model.keras"),
    os.path.join("models", "parkinson_cnn_model.h5"),
]
 
 
def enable_unsafe_deserialization():
    """Allow loading legacy graphs that used Lambda or custom objects."""
    if hasattr(tf.keras.config, "enable_unsafe_deserialization"):
        try:
            tf.keras.config.enable_unsafe_deserialization()
            return
        except Exception as ex:
            print(f"[WARNING] tf.keras.config.enable_unsafe_deserialization: {ex}")
    if hasattr(tf.keras.utils, "enable_unsafe_deserialization"):
        try:
            tf.keras.utils.enable_unsafe_deserialization()
        except Exception as ex:
            print(f"[WARNING] tf.keras.utils.enable_unsafe_deserialization: {ex}")
 
 
def _custom_objects():
    return {"AdaptiveModalityFusion": AdaptiveModalityFusion}
 
 
def safe_load_model(model_path):
    """Keras 3 compatible load with custom fusion layers registered."""
    enable_unsafe_deserialization()
    custom = _custom_objects()
    try:
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False,
            custom_objects=custom,
        )
    except TypeError:
        return tf.keras.models.load_model(
            model_path, compile=False, custom_objects=custom
        )
 
 
def save_keras_model(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save(path)
    print(f"[SUCCESS] Model saved to {path}")
 
 
def save_training_history(history, name):
    """Persist Keras History (or plain dict) for research logs."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    path = os.path.join(HISTORY_DIR, f"{name}.json")
    payload = history.history if hasattr(history, "history") else history
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[SUCCESS] Training history saved to {path}")
 
 
def migrate_legacy_model(legacy_path, target_path=None):
    target_path = target_path or EFFICIENTNET_MODEL_PATH
    print(f"[INFO] Migrating legacy model: {legacy_path} -> {target_path}")
    model = safe_load_model(legacy_path)
    save_keras_model(model, target_path)
    return model
 
 
def load_spiral_rgb_float(path: str) -> np.ndarray:
    """
    Load spiral image for EfficientNet: BGR->RGB, 224x224, float32 in [0, 255].
    Caller applies augment (optional) then preprocess_input once.
    """
    import cv2
 
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)
 
 
def apply_efficientnet_preprocess(img: np.ndarray) -> np.ndarray:
    """Single EfficientNet preprocess — input float32 RGB [0, 255], no /255 before."""
    if img.max() <= 1.0:
        raise ValueError(
            "Image appears normalized to [0,1]. Pass float32 [0,255] RGB before preprocess_input."
        )
    return preprocess_input(img)
 
 
def stack_spiral_preprocessed(records) -> np.ndarray:
    """Same inference tensors as train_cnn_model.py (no augmentation)."""
    if not records:
        return np.empty((0, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    batch = np.stack(
        [
            apply_efficientnet_preprocess(load_spiral_rgb_float(r.path))
            for r in records
        ],
        axis=0,
    ).astype(np.float32)
    return batch
 
 
def spiral_labels_from_records(records) -> np.ndarray:
    """healthy=0, parkinson=1 from SpiralRecord.label."""
    return np.array([int(r.label) for r in records], dtype=np.int32)
 
 
def save_cnn_threshold(threshold: float, extra: dict = None) -> None:
    os.makedirs(os.path.dirname(CNN_BEST_THRESHOLD_JSON) or ".", exist_ok=True)
    payload = {"threshold": float(threshold), "default": 0.5}
    if extra:
        payload.update(extra)
    with open(CNN_BEST_THRESHOLD_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.makedirs(os.path.dirname(CNN_THRESHOLD_LEGACY_JSON) or ".", exist_ok=True)
    with open(CNN_THRESHOLD_LEGACY_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(
        f"[SUCCESS] CNN threshold {threshold:.4f} saved: "
        f"{CNN_BEST_THRESHOLD_JSON}"
    )
 
 
def load_cnn_threshold(default: float = 0.5) -> float:
    for path in (CNN_BEST_THRESHOLD_JSON, CNN_THRESHOLD_LEGACY_JSON):
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            t = float(data.get("threshold", default))
            print(f"[INFO] Loaded CNN threshold {t:.4f} from {path}")
            return t
    print(f"[WARNING] No CNN threshold file; using default {default}")
    return default
 
 
def print_cnn_model_info(model_path: str = EFFICIENTNET_MODEL_PATH) -> None:
    import hashlib
    from datetime import datetime
 
    print("\n[INFO] === CNN model file ===")
    print(f"[INFO] Path: {os.path.abspath(model_path)}")
    if not os.path.isfile(model_path):
        print("[ERROR] CNN model file not found.")
        return
    stat = os.stat(model_path)
    print(f"[INFO] Modified: {datetime.fromtimestamp(stat.st_mtime)}")
    print(f"[INFO] Size: {stat.st_size / 1e6:.2f} MB")
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    print(f"[INFO] SHA256: {h.hexdigest()[:16]}...")
 
 
def evaluate_cnn_on_records(
    cnn_model,
    records,
    threshold: float,
    name: str = "CNN",
    batch_size: int = CNN_INFERENCE_BATCH_SIZE,
):
    """
    Identical inference to train_cnn_model.py hold-out eval (no augment, tuned threshold).
    Returns dict with y_true, y_prob, y_pred, metrics, confusion_matrix.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
 
    y_true = spiral_labels_from_records(records)
    X = stack_spiral_preprocessed(records)
    y_prob = cnn_model.predict(X, batch_size=batch_size, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)
 
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }
    cm = confusion_matrix(y_true, y_pred)
 
    print(f"\n[INFO] --- {name} (threshold={threshold:.4f}) ---")
    print(f"[SUCCESS] Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"[INFO] Balanced acc: {metrics['balanced_accuracy']*100:.2f}%")
    print(f"[INFO] Precision: {metrics['precision']*100:.2f}%")
    print(f"[INFO] Recall:    {metrics['recall']*100:.2f}%")
    print(f"[INFO] F1:        {metrics['f1']*100:.2f}%")
    print(f"[INFO] ROC-AUC:   {metrics['roc_auc']*100:.2f}%")
    print(f"[INFO] Confusion matrix:\n{cm}")
    print(
        f"[INFO] Predictions: healthy={int((y_pred == 0).sum())}, "
        f"parkinson={int((y_pred == 1).sum())}"
    )
    print(
        f"[INFO] Prob stats: mean={y_prob.mean():.4f} std={y_prob.std():.4f} "
        f"min={y_prob.min():.4f} max={y_prob.max():.4f}"
    )
 
    return {
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "metrics": metrics,
        "confusion_matrix": cm,
        "X_preprocessed": X,
    }
 
 
def build_advanced_augmentation():
    return tf.keras.Sequential(
        [
            RandomRotation(0.10, fill_mode="nearest"),
            RandomZoom((-0.15, 0.15), fill_mode="nearest"),
            RandomTranslation(0.08, 0.08, fill_mode="nearest"),
        ],
        name="data_augmentation",
    )
 
 
def _set_layer_trainable_recursive(layer, trainable: bool) -> None:
    """Set trainable on a layer and all nested sub-layers (Keras 3 safe)."""
    layer.trainable = trainable
    if hasattr(layer, "layers") and layer.layers:
        for sub in layer.layers:
            _set_layer_trainable_recursive(sub, trainable)
 
 
def freeze_efficientnet_backbone(base_model) -> None:
    """Phase 1: freeze entire EfficientNet backbone."""
    base_model.trainable = False
    for layer in base_model.layers:
        _set_layer_trainable_recursive(layer, False)
 
 
def unfreeze_efficientnet_top_layers(
    model: Model, base_model, num_layers: int = 40
) -> int:
    """
    Phase 2: unfreeze top ``num_layers`` EfficientNet layer groups (recursive).
    Classification head on ``model`` stays trainable.
    """
    base_model.trainable = True
    blocks = base_model.layers
    n = len(blocks)
    unfreeze_from = max(0, n - num_layers)
 
    for i, layer in enumerate(blocks):
        _set_layer_trainable_recursive(layer, trainable=(i >= unfreeze_from))
 
    def freeze_batchnorm_recursive(layer):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            _set_layer_trainable_recursive(layer, False)
        if hasattr(layer, "layers") and layer.layers:
            for sub in layer.layers:
                freeze_batchnorm_recursive(sub)
 
    freeze_batchnorm_recursive(base_model)
 
    for layer in model.layers:
        if layer is not base_model:
            _set_layer_trainable_recursive(layer, True)
 
    trainable_groups = n - unfreeze_from
    backbone_weights = len(base_model.trainable_weights)
    full_weights = len(model.trainable_weights)
    print(
        f"[INFO] EfficientNet fine-tune: {trainable_groups}/{n} "
        f"top-level backbone groups unfrozen (target top {num_layers})"
    )
    print(f"[INFO] Backbone trainable weight tensors: {backbone_weights}")
    print(f"[INFO] Full model trainable weight tensors: {full_weights}")
    if backbone_weights == 0:
        raise RuntimeError(
            "Fine-tuning failed: 0 trainable backbone weights. "
            "Check EfficientNet trainable flags."
        )
    return trainable_groups
 
 
def compile_efficientnet_classifier(
    model: Model, learning_rate: float = 1e-5, label_smoothing: float = 0.05
) -> Model:
    """Compile CNN for training or post-load evaluation/predict."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryFocalCrossentropy(
            gamma=2.0
        ),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
 
 
def cnn_compile_metrics():
    """Metrics logged each epoch (Parkinson = positive class, label 1)."""
    return [
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]
 
 
def cnn_binary_loss(label_smoothing: float = 0.05):
    """Standard Keras BCE with label smoothing."""
    return tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
 
 
def label_smoothing_binary_crossentropy(smoothing: float = 0.05):
    """Alias for fusion model compile compatibility."""
    return cnn_binary_loss(smoothing)
 
 
def build_efficientnet_classifier(
    input_shape=IMG_SHAPE, trainable_base=False, learning_rate=1e-4
):
    """
    EfficientNetB0 transfer-learning classifier (ImageNet weights, custom head).
    Returns (full_model, backbone) for two-phase training.
    """
    inputs = Input(shape=input_shape, name="image_input")
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    for layer in base_model.layers:
        layer.trainable = trainable_base
 
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = BatchNormalization(name="batch_norm")(x)
    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_256",
    )(x)
    x = BatchNormalization(name="dense_256_bn")(x)
    x = Dropout(0.5, name="dropout_head")(x)
    outputs = Dense(1, activation="sigmoid", name="prediction")(x)
 
    model = Model(inputs=inputs, outputs=outputs, name="efficientnet_pd_classifier")
    compile_efficientnet_classifier(model, learning_rate=learning_rate)
    return model, base_model
 
 
class AdaptiveModalityFusion(tf.keras.layers.Layer):
    """
    Sample-wise softmax gates over voice vs spiral embeddings (no Lambda layers).
    Outputs concatenation of gated voice and image vectors for interpretability.
    """
 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
 
    def call(self, inputs):
        voice_vec, image_vec, gates = inputs
        voice_w = gates[:, 0:1]
        image_w = gates[:, 1:2]
        return tf.concat([voice_vec * voice_w, image_vec * image_w], axis=-1)
 
    def get_config(self):
        return super().get_config()
 
 
CUSTOM_OBJECTS = {"AdaptiveModalityFusion": AdaptiveModalityFusion}
 
 
def get_cnn_embedding_dim(cnn_model) -> int:
    return int(cnn_model.get_layer("global_avg_pool").output.shape[-1])
 
 
def extract_cnn_cache(cnn_model, records, batch_size: int = CNN_INFERENCE_BATCH_SIZE) -> dict:
    """
    Precompute CNN embeddings + probabilities per spiral path (standalone pipeline).
    Keys: normalized filesystem paths.
    """
    if not records:
        return {}
    
    import pickle
    cache_path = os.path.join("outputs", "cnn_cache.pkl")
    disk_cache = {}
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                disk_cache = pickle.load(f)
            # Check if all records are in disk cache
            all_cached = True
            for r in records:
                if os.path.normpath(r.path) not in disk_cache:
                    all_cached = False
                    break
            if all_cached:
                print(f"[INFO] Loaded CNN feature cache for {len(records)} images from disk cache: {cache_path}")
                return {os.path.normpath(r.path): disk_cache[os.path.normpath(r.path)] for r in records}
        except Exception as ex:
            print(f"[WARNING] Could not load disk cache: {ex}")
            disk_cache = {}

    # Find missing records
    missing_records = [r for r in records if os.path.normpath(r.path) not in disk_cache]
    if missing_records:
        print(f"[INFO] Extracting CNN features for {len(missing_records)} missing images in batches...")
        export = Model(
            inputs=cnn_model.inputs,
            outputs=[
                cnn_model.get_layer("global_avg_pool").output,
                cnn_model.get_layer("prediction").output,
            ],
            name="cnn_feature_export",
        )
        export.trainable = False
        
        # Process in batches to avoid OOM
        chunk_size = 256
        for i in range(0, len(missing_records), chunk_size):
            chunk_records = missing_records[i : i + chunk_size]
            X_chunk = stack_spiral_preprocessed(chunk_records)
            emb_chunk, prob_chunk = export.predict(X_chunk, batch_size=batch_size, verbose=0)
            
            for j, rec in enumerate(chunk_records):
                key = os.path.normpath(rec.path)
                disk_cache[key] = {
                    "embedding": emb_chunk[j].astype(np.float32),
                    "prob": float(np.asarray(prob_chunk[j]).flatten()[0]),
                }
            print(f"  -> Processed {min(i + chunk_size, len(missing_records))}/{len(missing_records)} images")
        
        # Save to disk
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(disk_cache, f)
            print(f"[SUCCESS] Saved updated CNN feature cache to disk: {cache_path}")
        except Exception as ex:
            print(f"[WARNING] Could not save disk cache: {ex}")

    return {os.path.normpath(r.path): disk_cache[os.path.normpath(r.path)] for r in records}
 
 
def extract_single_cnn_features(cnn_model, image_path: str):
    """One image -> (embedding [1,D], prob [1,1]) for late-fusion inference."""
    from patient_data import SpiralRecord
 
    rec = SpiralRecord(path=image_path, label=0, patient_id="inference", split_folder="")
    cache = extract_cnn_cache(cnn_model, [rec])
    key = os.path.normpath(image_path)
    entry = cache[key]
    return (
        entry["embedding"].reshape(1, -1).astype(np.float32),
        np.array([[entry["prob"]]], dtype=np.float32),
    )
 
 
def build_late_fusion_model(
    voice_feature_dim: int,
    cnn_embed_dim: int,
    fusion_dim: int = 128,
):
    """
    True late fusion: voice vectors + CNN embeddings/probs only (no raw images, no CNN retrain).
 
    FIXES:
      - Dropout reduced from 0.5/0.45 to 0.25/0.2 in the fusion decision head.
        Original values caused underfitting on small paired datasets.
      - Adam learning_rate raised from 5e-5 to 1e-4 for faster, more reliable convergence.
        At 5e-5 the model frequently stopped early before finding good weights.
    """
    input_voice = Input(shape=(voice_feature_dim,), name="voice_input")
    input_voice_prob = Input(shape=(1,), name="voice_prob_input")
    input_cnn_embed = Input(shape=(cnn_embed_dim,), name="cnn_embed_input")
    input_cnn_prob = Input(shape=(1,), name="cnn_prob_input")
 
    l2 = tf.keras.regularizers.l2(1e-4)
 
    # --- Voice branch ---
    x_v = Dense(
        fusion_dim // 2,
        activation="relu",
        kernel_regularizer=l2,
        name="voice_dense_1",
    )(input_voice)
    x_v = BatchNormalization(name="voice_bn_1")(x_v)
    x_v = Dropout(0.3, name="voice_drop_1")(x_v)  # was 0.45
    voice_vec = Concatenate(name="voice_context")([x_v, input_voice_prob])
    voice_vec = Dense(
        fusion_dim // 2,
        activation="relu",
        kernel_regularizer=l2,
        name="voice_embedding",
    )(voice_vec)
 
    # --- CNN branch ---
    x_c = Dense(
        fusion_dim // 2,
        activation="relu",
        kernel_regularizer=l2,
        name="cnn_dense_1",
    )(input_cnn_embed)
    x_c = BatchNormalization(name="cnn_bn_1")(x_c)
    x_c = Dropout(0.3, name="cnn_drop_1")(x_c)  # was 0.45
    spiral_vec = Concatenate(name="spiral_context")([x_c, input_cnn_prob])
    spiral_vec = Dense(
        fusion_dim // 2,
        activation="relu",
        kernel_regularizer=l2,
        name="spiral_embedding",
    )(spiral_vec)
 
    # --- Adaptive modality attention ---
    joint_context = Concatenate(name="attention_context")([voice_vec, spiral_vec])
    modality_attention = Dense(2, activation="softmax", name="modality_attention")(
        joint_context
    )
    fused = AdaptiveModalityFusion(name="adaptive_modality_fusion")(
        [voice_vec, spiral_vec, modality_attention]
    )
 
    # --- Decision head (FIX: reduced dropout) ---
    x = Dense(64, activation="relu", kernel_regularizer=l2, name="fusion_dense_1")(fused)
    x = BatchNormalization(name="fusion_bn_1")(x)
    x = Dropout(0.25, name="fusion_drop_1")(x)   # was 0.5
    x = Dense(32, activation="relu", kernel_regularizer=l2, name="fusion_dense_2")(x)
    x = BatchNormalization(name="fusion_bn_2")(x)
    x = Dropout(0.2, name="fusion_drop_2")(x)    # was 0.45
    output = Dense(1, activation="sigmoid", name="fusion_output")(x)
 
    fusion_model = Model(
        inputs=[input_voice, input_voice_prob, input_cnn_embed, input_cnn_prob],
        outputs=output,
        name="late_adaptive_fusion",
    )
    # FIX: lr raised from 5e-5 to 1e-4
    fusion_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryFocalCrossentropy(
            gamma=2.0
        ),
        metrics=["accuracy"],
    )
    attention_model = Model(
        inputs=fusion_model.inputs,
        outputs=modality_attention,
        name="late_modality_attention_probe",
    )
    return fusion_model, attention_model
 
 
def build_adaptive_attention_fusion(cnn_model, voice_feature_dim, fusion_dim=128):
    """
    Attention-based adaptive multimodal fusion (raw image input variant — legacy).
    """
    input_voice = Input(shape=(voice_feature_dim,), name="voice_input")
    input_voice_prob = Input(shape=(1,), name="voice_prob_input")
    input_image = Input(shape=IMG_SHAPE, name="image_input")
 
    l2 = tf.keras.regularizers.l2(1e-4)
    x_v = Dense(
        fusion_dim // 2,
        activation="relu",
        kernel_regularizer=l2,
        name="voice_dense_1",
    )(input_voice)
    x_v = BatchNormalization(name="voice_bn_1")(x_v)
    x_v = Dropout(0.3, name="voice_drop_1")(x_v)
    voice_vec = Concatenate(name="voice_context")([x_v, input_voice_prob])
    voice_vec = Dense(
        fusion_dim // 2,
        activation="relu",
        kernel_regularizer=l2,
        name="voice_embedding",
    )(voice_vec)
 
    try:
        pool_layer = cnn_model.get_layer("global_avg_pool")
        cnn_output = pool_layer.output
    except ValueError:
        cnn_output = cnn_model.layers[-2].output
 
    cnn_feature_extractor = Model(
        inputs=cnn_model.inputs,
        outputs=cnn_output,
        name="cnn_feature_extractor",
    )
    cnn_feature_extractor.trainable = False
    image_vec = cnn_feature_extractor(input_image)
    image_vec = Dense(
        fusion_dim // 2,
        activation="relu",
        kernel_regularizer=l2,
        name="image_embedding",
    )(image_vec)
 
    joint_context = Concatenate(name="attention_context")([voice_vec, image_vec])
    modality_attention = Dense(
        2, activation="softmax", name="modality_attention"
    )(joint_context)
 
    fused = AdaptiveModalityFusion(name="adaptive_modality_fusion")(
        [voice_vec, image_vec, modality_attention]
    )
 
    x = Dense(
        64,
        activation="relu",
        kernel_regularizer=l2,
        name="fusion_dense_1",
    )(fused)
    x = BatchNormalization(name="fusion_bn_1")(x)
    x = Dropout(0.25, name="fusion_drop_1")(x)
    x = Dense(
        32,
        activation="relu",
        kernel_regularizer=l2,
        name="fusion_dense_2",
    )(x)
    x = BatchNormalization(name="fusion_bn_2")(x)
    x = Dropout(0.2, name="fusion_drop_2")(x)
    output = Dense(1, activation="sigmoid", name="fusion_output")(x)
 
    fusion_model = Model(
        inputs=[input_voice, input_voice_prob, input_image],
        outputs=output,
        name="adaptive_attention_fusion",
    )
    fusion_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryFocalCrossentropy(
            gamma=2.0
        ),
        metrics=["accuracy"],
    )
 
    attention_model = Model(
        inputs=fusion_model.inputs,
        outputs=modality_attention,
        name="modality_attention_probe",
    )
    return fusion_model, attention_model
 
 
def load_efficientnet_cnn(
    rebuild_on_failure=True,
    compile_model: bool = False,
    learning_rate: float = 1e-5,
):
    enable_unsafe_deserialization()
    candidates = [EFFICIENTNET_MODEL_PATH]
    for legacy in LEGACY_CNN_PATHS:
        if legacy not in candidates:
            candidates.append(legacy)
 
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            print(f"[INFO] Loading CNN from {path}")
            model = safe_load_model(path)
            in_shape = model.input_shape
            if in_shape and in_shape[1:3] != (IMG_SIZE, IMG_SIZE):
                print(
                    f"[WARNING] Skipping {path}: expected {IMG_SHAPE}, got {in_shape}"
                )
                continue
            if path != EFFICIENTNET_MODEL_PATH:
                migrate_legacy_model(path, EFFICIENTNET_MODEL_PATH)
            else:
                print(f"[SUCCESS] CNN loaded from {path}")
            if compile_model:
                compile_efficientnet_classifier(model, learning_rate=learning_rate)
                print(f"[INFO] CNN recompiled for inference (lr={learning_rate:.1e})")
            return model
        except Exception as ex:
            print(f"[WARNING] Could not load {path}: {ex}")
 
    if rebuild_on_failure:
        print("[INFO] Rebuilding EfficientNetB0 @ 224x224 (untrained weights).")
        model, _ = build_efficientnet_classifier(learning_rate=learning_rate)
        print("[SUCCESS] EfficientNet rebuilt.")
        return model
 
    raise FileNotFoundError(
        f"Train CNN first or place model at {EFFICIENTNET_MODEL_PATH}"
    )
 
 
def preprocess_images_for_efficientnet(images):
    """EfficientNet: float32 RGB [0, 255] -> preprocess_input (once, no double norm)."""
    images = np.asarray(images, dtype=np.float32)
    if images.size == 0:
        return images
    if images.ndim == 3:
        return apply_efficientnet_preprocess(images)
    if images.max() <= 1.0:
        images = images * 255.0
    return preprocess_input(images)
 
 
class TrainingMonitorCallback(tf.keras.callbacks.Callback):
    """Print train/val metrics and learning rate each epoch."""
 
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(
            f"[Epoch {epoch + 1}] "
            f"train_acc={logs.get('accuracy', 0)*100:.2f}% "
            f"val_acc={logs.get('val_accuracy', 0)*100:.2f}% | "
            f"val_recall={logs.get('val_recall', 0)*100:.2f}% "
            f"val_precision={logs.get('val_precision', 0)*100:.2f}% | "
            f"val_auc={logs.get('val_auc', 0):.4f} | "
            f"train_loss={logs.get('loss', 0):.4f} "
            f"val_loss={logs.get('val_loss', 0):.4f} | lr={lr:.2e}"
        )
 
 
class PredictionDistributionCallback(tf.keras.callbacks.Callback):
    """
    Debug validation probability distribution each epoch.
    Detects collapse to all-Parkinson (mean prob ~1) or all-healthy (mean prob ~0).
    """
 
    def __init__(self, val_dataset, threshold: float = 0.5):
        super().__init__()
        self.val_dataset = val_dataset
        self.threshold = threshold
 
    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.val_dataset, verbose=0).flatten()
        pred = (probs >= self.threshold).astype(int)
        n_pd = int(pred.sum())
        n_h = int(len(pred) - n_pd)
        print(
            f"[DEBUG Epoch {epoch + 1}] val prob mean={probs.mean():.3f} "
            f"std={probs.std():.3f} min={probs.min():.3f} max={probs.max():.3f} | "
            f"pred healthy={n_h} parkinson={n_pd}"
        )
        if probs.std() < 0.01:
            print(
                "[ERROR] Model collapse detected (std(predictions) < 0.01). "
                "Probabilities are nearly constant."
            )
        if n_pd == len(pred):
            print("[WARNING] Collapse: ALL validation samples predicted Parkinson.")
        elif n_h == len(pred):
            print("[WARNING] Collapse: ALL validation samples predicted healthy.")
        elif probs.mean() > 0.75:
            print("[WARNING] Strong Parkinson bias (mean prob > 0.75).")
        elif probs.mean() < 0.25:
            print("[WARNING] Strong healthy bias (mean prob < 0.25).")
 
 
def get_cnn_training_callbacks(
    checkpoint_path: str,
    val_dataset=None,
    monitor: str = "val_auc",
    patience: int = 8,
    lr_patience: int = 3,
):
    """Early stop / checkpoint on val_auc (max) with LR reduction on plateau."""
    mode = "min" if monitor in ("val_loss", "loss") else "max"
    cbs = [
        TrainingMonitorCallback(),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=0.5,
            patience=lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=1,
        ),
    ]
    if val_dataset is not None:
        cbs.insert(1, PredictionDistributionCallback(val_dataset))
    return cbs
 
 
def get_standard_callbacks(
    checkpoint_path,
    monitor="val_loss",
    patience=8,
    monitor_acc: bool = True,
):
    """EarlyStopping + LR schedule + best checkpoint; restores best weights."""
    stop_monitor = "val_accuracy" if monitor_acc else monitor
    return [
        TrainingMonitorCallback(),
        tf.keras.callbacks.EarlyStopping(
            monitor=stop_monitor,
            mode="max" if monitor_acc else "min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor=stop_monitor,
            mode="max" if monitor_acc else "min",
            save_best_only=True,
            verbose=0,
        ),
    ]