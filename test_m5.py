
from hybrid_gaze_m5 import HybridFeatureExtractor

MODEL_DIR = "models"

print("Initializing M5 model...")

extractor = HybridFeatureExtractor(
    model_dir=MODEL_DIR
)

print("\nSUCCESS!")
print("Device:", extractor.device)
print("Yaw bins:", extractor.yaw_bins)
print("Pitch bins:", extractor.pitch_bins)

extractor.close()

print("\nM5 checkpoint and MediaPipe loaded successfully.")
