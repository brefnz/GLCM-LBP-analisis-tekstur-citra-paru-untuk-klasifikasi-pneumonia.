"""
Training Utility
================
Trains three classifiers on GLCM + LBP + Statistical features:
  - SVM (RBF kernel)
  - Random Forest
  - Gradient Boosting

Automatically selects the best model by 5-fold cross-validation accuracy,
saves it together with the fitted scaler and evaluation metrics.
Only summary output is printed – no per-file logging.
"""

import os
import json
import time
import joblib
import threading
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

from features.extractor import extract_features
from config import (
    DATASET_DIR, MODEL_DIR, MODEL_PATH, SCALER_PATH, METRICS_PATH, ANOMALY_PATH,
    CLASSES, CLASS_LABELS, TEST_SIZE, RANDOM_STATE, MAX_SAMPLES_PER_CLASS,
)

# ── Shared state (polled by Flask via /api/train-status) ──────────────────────

training_status: dict = {
    "running": False,
    "progress": 0,
    "stage": "",
    "message": "",
    "complete": False,
    "error": None,
    "metrics": None,
}

_lock = threading.Lock()


def _update(progress: int, stage: str, message: str):
    with _lock:
        training_status["progress"] = progress
        training_status["stage"] = stage
        training_status["message"] = message


def _fmt(secs: float) -> str:
    return f"{secs:.1f}s"


# ── Dataset helpers ────────────────────────────────────────────────────────────

def _collect_paths(dataset_dir: str, max_per_class: int | None):
    """Return (X_paths, y_labels) with optional per-class cap."""
    X, y = [], []
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = os.path.join(dataset_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        files = sorted([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if max_per_class:
            np.random.seed(RANDOM_STATE)
            np.random.shuffle(files)
            files = files[:max_per_class]
        for f in files:
            X.append(os.path.join(cls_dir, f))
            y.append(cls_idx)
    return X, y


def _balance(X: list, y: list):
    """Undersample majority class so class counts are equal."""
    classes, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())
    Xb, yb = [], []
    np.random.seed(RANDOM_STATE)
    for cls in classes:
        idxs = np.where(np.array(y) == cls)[0]
        chosen = np.random.choice(idxs, size=min_count, replace=False)
        Xb.extend([X[i] for i in chosen])
        yb.extend([y[i] for i in chosen])
    return Xb, yb


# ── Main training entry point ──────────────────────────────────────────────────

def train_model(dataset_dir: str | None = None, max_per_class: int | None = None):
    """
    Full training pipeline.  Meant to run in a background thread.
    Updates *training_status* throughout so the UI can poll progress.
    """
    global training_status

    with _lock:
        training_status.update({
            "running": True,
            "progress": 0,
            "stage": "init",
            "message": "Mempersiapkan pipeline...",
            "complete": False,
            "error": None,
            "metrics": None,
        })

    if dataset_dir is None:
        dataset_dir = DATASET_DIR

    SEP = "=" * 62

    try:
        print(f"\n{SEP}")
        print("  LUNG DISEASE CLASSIFICATION  –  MODEL TRAINING")
        print(SEP)

        # ── 1. Load paths ──────────────────────────────────────────────────
        _update(2, "loading", "Memuat daftar dataset...")
        cap = max_per_class or MAX_SAMPLES_PER_CLASS
        X_paths, y_raw = _collect_paths(dataset_dir, cap)

        if not X_paths:
            raise FileNotFoundError(
                "Dataset tidak ditemukan. Pastikan folder dataset/NORMAL dan "
                "dataset/PNEUMONIA sudah terisi gambar."
            )

        X_paths, y_raw = _balance(X_paths, y_raw)

        _, counts = np.unique(y_raw, return_counts=True)
        print(f"\n[INFO] Dataset loaded (balanced):")
        for cls, cnt in zip(CLASSES, counts):
            print(f"       {cls:12s} : {cnt} images")
        print(f"       {'Total':12s} : {sum(counts)} images")

        # ── 2. Feature extraction ──────────────────────────────────────────
        total = len(X_paths)
        _update(5, "features", f"Ekstraksi fitur (0 / {total})...")
        print(f"\n[INFO] Extracting features from {total} images …")
        print("       Methods : GLCM + LBP + Statistical")

        X_feat, y_final = [], []
        failed = 0
        t_feat_start = time.time()

        for idx, (path, label) in enumerate(zip(X_paths, y_raw)):
            try:
                X_feat.append(extract_features(path))
                y_final.append(label)
            except Exception:
                failed += 1

            # Update every 5 %
            if (idx + 1) % max(1, total // 20) == 0 or (idx + 1) == total:
                pct = int((idx + 1) / total * 100)
                prog = 5 + int((idx + 1) / total * 50)
                _update(prog, "features", f"Ekstraksi fitur: {idx+1} / {total} gambar")
                print(f"       {pct:3d}%  –  {idx+1}/{total} images processed")

        t_feat = time.time() - t_feat_start
        X = np.array(X_feat, dtype=np.float32)
        y = np.array(y_final, dtype=int)

        if failed:
            print(f"\n[WARN] {failed} image(s) skipped due to read/processing errors.")

        print(f"\n[INFO] Feature extraction complete in {_fmt(t_feat)}")
        print(f"       Feature vector size : {X.shape[1]} dimensions")

        # ── 3. Train / test split + scaling ───────────────────────────────
        _update(57, "split", "Memisahkan data train/test & scaling...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        print(f"\n[INFO] Split  :  {len(X_train)} train  |  {len(X_test)} test")
        print(f"       Scaler :  StandardScaler (zero-mean, unit-variance)")

        # ── 4. Train classifiers ───────────────────────────────────────────
        classifiers = {
            "SVM (RBF)": SVC(
                kernel="rbf", C=10, gamma="scale",
                probability=True, random_state=RANDOM_STATE,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_features="sqrt",
                n_jobs=-1, random_state=RANDOM_STATE,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1,
                max_depth=4, random_state=RANDOM_STATE,
            ),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        best_model, best_name, best_cv_score = None, "", 0.0
        all_cv: dict[str, float] = {}

        print(f"\n[INFO] Training {len(classifiers)} classifiers with 5-fold CV …")

        for i, (name, clf) in enumerate(classifiers.items()):
            base_prog = 60 + i * 10
            _update(base_prog, "training", f"Training {name}...")
            print(f"\n       [{i+1}/{len(classifiers)}]  {name} …", end=" ", flush=True)

            t0 = time.time()
            clf.fit(X_train_s, y_train)
            elapsed = time.time() - t0

            cv_scores = cross_val_score(clf, X_train_s, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
            mean_cv = float(cv_scores.mean())
            all_cv[name] = mean_cv

            print(f"done ({_fmt(elapsed)})  |  CV Accuracy = {mean_cv*100:.2f}%")

            if mean_cv > best_cv_score:
                best_cv_score = mean_cv
                best_model = clf
                best_name = name

        # ── 5. Final evaluation on hold-out test set ───────────────────────
        _update(92, "evaluation", "Evaluasi model terbaik...")
        print(f"\n[INFO] Best model : {best_name}  (CV Acc = {best_cv_score*100:.2f}%)")
        print("[INFO] Running final evaluation on hold-out test set …")

        y_pred = best_model.predict(X_test_s)
        y_proba = best_model.predict_proba(X_test_s)[:, 1]

        metrics = {
            "model_name": best_name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=CLASSES, output_dict=True, zero_division=0
            ),
            "cv_scores": {k: round(v * 100, 2) for k, v in all_cv.items()},
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "n_features": int(X.shape[1]),
            "classes": CLASSES,
        }

        print(f"\n{SEP}")
        print("  HASIL EVALUASI")
        print(SEP)
        print(f"  Model       : {metrics['model_name']}")
        print(f"  Accuracy    : {metrics['accuracy']*100:.2f} %")
        print(f"  Precision   : {metrics['precision']*100:.2f} %")
        print(f"  Recall      : {metrics['recall']*100:.2f} %")
        print(f"  F1-Score    : {metrics['f1_score']*100:.2f} %")
        print(f"  ROC-AUC     : {metrics['roc_auc']*100:.2f} %")
        print(SEP)

        cm = np.array(metrics["confusion_matrix"])
        print("\n  Confusion Matrix:")
        for i, row in enumerate(cm):
            print(f"    {CLASSES[i]:12s} :  {row}")

        print(SEP)

        # ── 6. Train Isolation Forest anomaly detector ─────────────
        _update(94, "anomaly", "Melatih anomaly detector (Isolation Forest)...")
        print("\n[INFO] Training Isolation Forest anomaly detector …")
        print("       Learns what 'lung X-ray features' look like.")
        print("       Non-lung images will be rejected as outliers.")

        iso_forest = IsolationForest(
            n_estimators=300,
            contamination=0.05,   # assume ≤5% of training data could be noisy
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        # Train on ALL scaled lung features (both classes)
        iso_forest.fit(X_train_s)

        # Quick self-test: inlier rate on training data should be ~95%
        train_preds = iso_forest.predict(X_train_s)
        inlier_rate = float(np.sum(train_preds == 1) / len(train_preds))
        print(f"       Inlier rate on train set: {inlier_rate*100:.1f}%  (expected ~95%)")

        # ── 7. Persist model ───────────────────────────────────────────────
        _update(98, "saving", "Menyimpan model...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(iso_forest, ANOMALY_PATH)
        with open(METRICS_PATH, "w") as fh:
            json.dump(metrics, fh, indent=2)

        print(f"\n[INFO] Classifier saved     →  {MODEL_PATH}")
        print(f"[INFO] Anomaly detector saved →  {ANOMALY_PATH}")
        print("[INFO] Training complete!\n")

        with _lock:
            training_status.update({
                "running": False,
                "progress": 100,
                "stage": "done",
                "message": "Training selesai!",
                "complete": True,
                "error": None,
                "metrics": metrics,
            })

        return metrics

    except Exception as exc:
        with _lock:
            training_status.update({
                "running": False,
                "progress": 0,
                "stage": "error",
                "message": f"Error: {exc}",
                "complete": False,
                "error": str(exc),
            })
        import traceback
        print(f"\n[ERROR] Training failed:\n{traceback.format_exc()}")
        raise
