import os
import json
import threading
import numpy as np
import cv2
import joblib

from flask import (
    Flask, render_template, request, jsonify, redirect, url_for
)
from werkzeug.utils import secure_filename

from config import (
    MODEL_PATH, SCALER_PATH, METRICS_PATH, ANOMALY_PATH,
    UPLOAD_DIR, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH,
    CONFIDENCE_THRESHOLD, ANOMALY_THRESHOLD,
    CLASS_LABELS, CLASSES,
)
from features.extractor import extract_features
from utils.trainer import train_model, training_status

# ─── App setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "lungclassify-secret-2024")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Make Python builtins available in Jinja2 templates
app.jinja_env.globals.update(enumerate=enumerate, zip=zip, len=len)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def model_is_trained() -> bool:
    return (os.path.exists(MODEL_PATH) and
            os.path.exists(SCALER_PATH) and
            os.path.exists(ANOMALY_PATH))


def load_model_and_scaler():
    if model_is_trained():
        return (joblib.load(MODEL_PATH),
                joblib.load(SCALER_PATH),
                joblib.load(ANOMALY_PATH))
    return None, None, None


def get_saved_metrics() -> dict | None:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as fh:
            return json.load(fh)
    return None


PREDICTION_DESCRIPTIONS = {
    "NORMAL": (
        "Citra paru-paru menunjukkan pola normal. "
        "Tidak ditemukan indikasi Pneumonia pada gambar ini."
    ),
    "PNEUMONIA": (
        "Terdeteksi indikasi Pneumonia pada citra paru-paru. "
        "Temuan ini perlu dikonfirmasi oleh tenaga medis profesional."
    ),
}


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    trained = model_is_trained()
    metrics = get_saved_metrics() if trained else None
    return render_template("dashboard.html", trained=trained, metrics=metrics)


@app.route("/prediksi", methods=["GET"])
def prediksi_page():
    return render_template("predict.html", trained=model_is_trained())


@app.route("/prediksi", methods=["POST"])
def prediksi_api():
    """Receive an uploaded image and return a JSON prediction."""
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Tidak ada file yang dipilih."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Format file tidak didukung. Gunakan PNG, JPG, atau JPEG."
        }), 400

    model, scaler, iso_forest = load_model_and_scaler()
    if model is None:
        return jsonify({
            "error": "Model belum dilatih. Lakukan training terlebih dahulu."
        }), 503

    # Save temporarily
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    fname = secure_filename(file.filename)
    fpath = os.path.join(UPLOAD_DIR, fname)
    file.save(fpath)

    try:
        features = extract_features(fpath)
        features_scaled = scaler.transform([features])

        # ── Layer 1: Anomaly detection ────────────────────────────
        # Isolation Forest score: positive = inlier, negative = outlier
        anomaly_score = float(iso_forest.decision_function(features_scaled)[0])
        iso_pred = iso_forest.predict(features_scaled)[0]   # 1=inlier, -1=outlier

        if iso_pred == -1 or anomaly_score < ANOMALY_THRESHOLD:
            return jsonify({
                "status": "unknown",
                "prediction": "TIDAK DIKENALI",
                "label_display": "Gambar Tidak Dikenali",
                "description": (
                    "Gambar tidak terdeteksi sebagai citra rontgen paru-paru. "
                    "Kemungkinan merupakan foto benda asing, objek non-medis, "
                    "atau citra yang tidak sesuai format pemeriksaan."
                ),
                "confidence": 0.0,
                "threshold": round(CONFIDENCE_THRESHOLD * 100, 2),
                "probabilities": {cls: 0.0 for cls in CLASSES},
                "note": (
                    f"Anomaly score: {anomaly_score:.4f} "
                    f"(threshold: {ANOMALY_THRESHOLD}) — gambar berada di luar "
                    "distribusi citra rontgen paru-paru."
                ),
            })

        # ── Layer 2: Binary classification ────────────────────────
        probas = model.predict_proba(features_scaled)[0]
        max_prob = float(np.max(probas))
        pred_idx = int(np.argmax(probas))
        prob_dict = {cls: round(float(p) * 100, 2) for cls, p in zip(CLASSES, probas)}

        if max_prob < CONFIDENCE_THRESHOLD:
            return jsonify({
                "status": "unknown",
                "prediction": "TIDAK DIKENALI",
                "label_display": "Gambar Tidak Dikenali",
                "description": (
                    "Gambar terdeteksi sebagai rontgen paru-paru namun "
                    "confidence terlalu rendah untuk memberikan klasifikasi yang andal."
                ),
                "confidence": round(max_prob * 100, 2),
                "threshold": round(CONFIDENCE_THRESHOLD * 100, 2),
                "probabilities": prob_dict,
                "note": (
                    f"Confidence ({max_prob*100:.1f}%) berada di bawah "
                    f"threshold ({CONFIDENCE_THRESHOLD*100:.0f}%)."
                ),
            })

        cls_name = CLASS_LABELS[pred_idx]
        return jsonify({
            "status": "success",
            "prediction": cls_name,
            "label_display": "Normal" if cls_name == "NORMAL" else "Pneumonia",
            "description": PREDICTION_DESCRIPTIONS.get(cls_name, ""),
            "confidence": round(max_prob * 100, 2),
            "threshold": round(CONFIDENCE_THRESHOLD * 100, 2),
            "probabilities": prob_dict,
        })

    except Exception as exc:
        return jsonify({"error": f"Gagal memproses gambar: {exc}"}), 500

    finally:
        if os.path.exists(fpath):
            os.remove(fpath)


@app.route("/train", methods=["GET"])
def train_page():
    trained = model_is_trained()
    metrics = get_saved_metrics() if trained else None
    return render_template(
        "train.html",
        trained=trained,
        metrics=metrics,
        status=training_status,
    )


@app.route("/train/mulai", methods=["POST"])
def train_start():
    """Kick off training in a background daemon thread."""
    if training_status.get("running"):
        return jsonify({"error": "Training sedang berjalan."}), 409

    raw = request.form.get("max_per_class", "").strip()
    max_per_class = int(raw) if raw.isdigit() else None

    t = threading.Thread(
        target=train_model,
        kwargs={"max_per_class": max_per_class},
        daemon=True,
    )
    t.start()

    return jsonify({"message": "Training dimulai."})


@app.route("/api/train-status")
def api_train_status():
    return jsonify(training_status)


@app.route("/api/model-info")
def api_model_info():
    return jsonify({
        "trained": model_is_trained(),
        "metrics": get_saved_metrics(),
    })


@app.route("/tentang")
def tentang():
    return render_template("about.html")


# ─── Startup ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
