
import os
import time
import torch
import torch.nn as nn
import torchvision.models as models

# ---------------- Configuration ----------------

MODEL_DIR = os.path.expanduser("~/GazeConnect/models")
IMG_SIZE = 64
WARMUP = 30
ITERATIONS = 200

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("=" * 95)
print("GazeConnect Model Inference Benchmark")
print("=" * 95)
print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Input shape: (1, 3, {IMG_SIZE}, {IMG_SIZE})")
print(f"Warmup iterations: {WARMUP}")
print(f"Timed iterations: {ITERATIONS}")


# ---------------- Checkpoint loading ----------------

def load_checkpoint(filename):
    path = os.path.join(MODEL_DIR, filename)

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


# ---------------- Model architectures ----------------

def make_efficientnet():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )
    return model


def make_mobilenet():
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, 2
    )
    return model


def make_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model



class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.regressor(self.features(x))


def make_basic_cnn():
    return BasicCNN()


class L2CSStyle(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()

        self.yaw_head = nn.Linear(512, 7)
        self.pitch_head = nn.Linear(512, 3)

    def forward(self, x):
        features = self.backbone(x)
        yaw_logits = self.yaw_head(features)
        pitch_logits = self.pitch_head(features)
        return yaw_logits, pitch_logits


# ---------------- Checkpoints to benchmark ----------------

MODEL_BUILDERS = {
    "Basic CNN": (
        "basic_cnn_best.pth",
        make_basic_cnn
    ),
    "ResNet18": (
        "resnet18_best.pth",
        make_resnet18
    ),
    "MobileNetV3-Large": (
        "mobilenetv3_best.pth",
        make_mobilenet
    ),
    "EfficientNet-B0 pilot": (
        "efficientnet_b0_best.pth",
        make_efficientnet
    ),
    "EfficientNet-B0 frozen": (
        "efficientnet_b0_frozen_best.pth",
        make_efficientnet
    ),
    "EfficientNet-B0 fine-tuned": (
        "efficientnet_b0_finetuned_best.pth",
        make_efficientnet
    ),
    "L2CS-style": (
        "l2cs_style_best.pth",
        L2CSStyle
    ),
}


# ---------------- Benchmark utilities ----------------

def synchronize_device():
    if device.type == "mps":
        torch.mps.synchronize()


def benchmark_model(name, filename, builder):
    path = os.path.join(MODEL_DIR, filename)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = load_checkpoint(filename)

    # The fine-tuned EfficientNet checkpoint is wrapped
    # in a dictionary; the others contain direct state dicts.
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = builder()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    # Random input measures model compute time only.
    # It does not measure webcam capture or preprocessing.
    x = torch.randn(
        1, 3, IMG_SIZE, IMG_SIZE,
        device=device
    )

    with torch.inference_mode():
        # Warm up the device and model.
        for _ in range(WARMUP):
            model(x)

        synchronize_device()

        timings_ms = []

        for _ in range(ITERATIONS):
            synchronize_device()
            start = time.perf_counter()

            model(x)

            synchronize_device()
            elapsed_ms = (time.perf_counter() - start) * 1000
            timings_ms.append(elapsed_ms)

    timings_ms.sort()

    mean_ms = sum(timings_ms) / len(timings_ms)
    median_ms = timings_ms[len(timings_ms) // 2]
    fps = 1000.0 / mean_ms

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )
    checkpoint_size_mb = os.path.getsize(path) / (1024 ** 2)

    return {
        "name": name,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "fps": fps,
        "params_m": parameter_count / 1_000_000,
        "size_mb": checkpoint_size_mb,
    }


# ---------------- Run all benchmarks ----------------

results = []

print("\n")
print(
    f"{'Model':28} | {'Mean ms':>9} | {'Median ms':>10} | "
    f"{'FPS':>9} | {'Params (M)':>10} | {'File (MB)':>9}"
)
print("-" * 95)

for name, (filename, builder) in MODEL_BUILDERS.items():
    try:
        result = benchmark_model(name, filename, builder)
        results.append(result)

        print(
            f"{result['name']:28} | "
            f"{result['mean_ms']:9.2f} | "
            f"{result['median_ms']:10.2f} | "
            f"{result['fps']:9.2f} | "
            f"{result['params_m']:10.2f} | "
            f"{result['size_mb']:9.2f}"
        )

    except Exception as error:
        print(
            f"{name:28} | ERROR: "
            f"{type(error).__name__}: {error}"
        )

print("-" * 95)
print(f"Successfully benchmarked: {len(results)}/{len(MODEL_BUILDERS)}")

if results:
    fastest = min(results, key=lambda r: r["mean_ms"])
    smallest = min(results, key=lambda r: r["size_mb"])

    print(f"\nFastest model in this test: {fastest['name']}")
    print(f"  Mean latency: {fastest['mean_ms']:.2f} ms")
    print(f"  Approximate throughput: {fastest['fps']:.2f} FPS")

    print(f"\nSmallest checkpoint: {smallest['name']}")
    print(f"  File size: {smallest['size_mb']:.2f} MB")

print(
    "\nNote: FPS is calculated as 1000 / mean model latency. "
    "It is not end-to-end webcam FPS."
)

