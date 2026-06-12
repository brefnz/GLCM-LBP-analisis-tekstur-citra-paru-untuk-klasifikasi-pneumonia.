import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
CLASSES = ['NORMAL', 'PNEUMONIA']
CLASS_LABELS = {0: 'NORMAL', 1: 'PNEUMONIA'}

# Model persistence
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')
ANOMALY_PATH = os.path.join(MODEL_DIR, 'anomaly_detector.pkl')  # Isolation Forest

# Upload
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Feature extraction
IMAGE_SIZE = (256, 256)
GLCM_DISTANCES = [1, 2]
GLCM_ANGLES_RAD = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_LEVELS = 64
LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS  # 24

# Classification threshold
# Isolation Forest: if anomaly_score < ANOMALY_THRESHOLD → "Tidak Dikenali"
# (score is negative; more negative = more anomalous)
ANOMALY_THRESHOLD = -0.05        # tune if needed; -0.05 works well in practice
CONFIDENCE_THRESHOLD = 0.55      # secondary guard after anomaly check passes

# Training
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_SAMPLES_PER_CLASS = 1200  # Cap per class to keep training manageable
