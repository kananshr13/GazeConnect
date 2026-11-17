#!/usr/bin/env python3
"""
GazeConnect M5: binocular L2CS + MediaPipe geometry + PSO/Ridge mapper.

Expected checkpoint:
    <model_dir>/binocular_l2cs_best.pth

The checkpoint must contain:
    model_state_dict, yaw_bins, pitch_bins

The `extract()` method preserves the existing application interface:
    result = extractor.extract(frame_bgr, landmarks_xy, geom_dict)

`geom_dict["f"]` must already contain exactly three geometric values:
    [horizontal_iris_ratio, vertical_offset, eyelid_relative_vertical_ratio]

The returned `geom_dict["f"]` contains nine raw features:
    [h, v_off, v_lid, head_yaw_proxy, head_pitch_proxy,
     yaw_eye_image_left, pitch_eye_image_left,
     yaw_eye_image_right, pitch_eye_image_right]
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    mp = None
    python = None
    vision = None


# Exact contour indices and crop settings used in GazeConnect_Training_Updated.ipynb.
LEFT_EYE = [
    33, 246, 161, 160, 159, 158, 157, 173,
    133, 155, 154, 153, 145, 144, 163, 7
]
RIGHT_EYE = [
    362, 398, 384, 385, 386, 387, 388, 466,
    263, 249, 390, 373, 374, 380, 381, 382
]
IMAGE_SIZE = 64
CROP_PADDING = 0.35
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class BinocularL2CS(nn.Module):
    """Architecture matching Cell 42 of the supplied training notebook."""

    def __init__(self, yaw_count: int = 7, pitch_count: int = 3):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.yaw_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, yaw_count),
        )
        self.pitch_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, pitch_count),
        )

    def forward(self, eyes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # eyes: B x 2 x 3 x 64 x 64; shared ResNet processes both eyes.
        batch_size = eyes.shape[0]
        eyes = eyes.reshape(batch_size * 2, 3, IMAGE_SIZE, IMAGE_SIZE)
        features = self.backbone(eyes)
        yaw_logits = self.yaw_head(features).reshape(batch_size, 2, -1)
        pitch_logits = self.pitch_head(features).reshape(batch_size, 2, -1)
        return yaw_logits, pitch_logits


def _landmarks_to_xy(landmarks: Any, width: int, height: int) -> np.ndarray:
    """Accept either MediaPipe normalized landmark objects or pixel Nx2 arrays."""
    if isinstance(landmarks, np.ndarray):
        points = np.asarray(landmarks, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("landmarks array must have shape (N, 2+) ")
        points = points[:, :2].copy()
        # Existing GazeConnect code passes pixel-space P. If values look normalized,
        # convert them to pixels.
        if np.nanmax(np.abs(points)) <= 2.0:
            points[:, 0] *= width
            points[:, 1] *= height
        return points

    points = []
    for point in landmarks:
        points.append([float(point.x) * width, float(point.y) * height])
    return np.asarray(points, dtype=np.float32)


def _crop_eye_rgb(
    image_rgb: np.ndarray,
    points_xy: np.ndarray,
    indices: Sequence[int],
    padding: float = CROP_PADDING,
) -> Optional[np.ndarray]:
    """Reproduce the notebook crop: 35% padding, 64x64, INTER_AREA."""
    h, w = image_rgb.shape[:2]
    if len(points_xy) <= max(indices):
        return None
    points = points_xy[np.asarray(indices, dtype=np.int64)]
    if not np.isfinite(points).all():
        return None

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    eye_w = float(x_max - x_min)
    eye_h = float(y_max - y_min)
    if eye_w < 3 or eye_h < 3:
        return None

    pad_x = eye_w * padding
    pad_y = eye_h * padding
    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(w, int(x_max + pad_x))
    y2 = min(h, int(y_max + pad_y))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def _hybrid_design(F: np.ndarray) -> np.ndarray:
    """
    F columns:
      0:h, 1:v_off, 2:v_lid, 3:head_yaw, 4:head_pitch,
      5:yaw_image_left, 6:pitch_image_left,
      7:yaw_image_right, 8:pitch_image_right.

    The quadratic mapper uses averaged binocular yaw/pitch while PSO scales
    all four raw appearance features before this design transformation.
    """
    if F.ndim != 2 or F.shape[1] != 9:
        raise ValueError(f"Expected F with shape (N, 9), received {F.shape}")
    h, v, lid = F[:, 0], F[:, 1], F[:, 2]
    head_yaw, head_pitch = F[:, 3], F[:, 4]
    yaw_avg = 0.5 * (F[:, 5] + F[:, 7])
    pitch_avg = 0.5 * (F[:, 6] + F[:, 8])
    return np.column_stack([
        np.ones(len(F)),
        h, v, lid,
        h * h, v * v, h * v,
        yaw_avg, pitch_avg,
        head_yaw, head_pitch,
    ])


class HybridFeatureExtractor:
    """Live MediaPipe eye crops + trained binocular L2CS predictions."""

    def __init__(
        self,
        model_dir: str,
        landmark_model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if mp is None:
            raise ImportError(
                "MediaPipe Tasks is required. Install a compatible mediapipe package."
            )

        self.device = torch.device(
            device or (
                "mps" if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        )
        checkpoint_path = os.path.join(model_dir, "binocular_l2cs_best.pth")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Binocular checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise RuntimeError(
                "Unexpected checkpoint format: expected a dict containing model_state_dict."
            )

        self.yaw_bins = np.asarray(checkpoint.get("yaw_bins", [-15, -10, -5, 0, 5, 10, 15]),
                                   dtype=np.float32)
        self.pitch_bins = np.asarray(checkpoint.get("pitch_bins", [-10, 0, 10]),
                                     dtype=np.float32)
        if len(self.yaw_bins) != 7 or len(self.pitch_bins) != 3:
            raise RuntimeError(
                f"Unexpected bin counts: yaw={self.yaw_bins}, pitch={self.pitch_bins}"
            )

        self.model = BinocularL2CS(len(self.yaw_bins), len(self.pitch_bins))
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device).eval()

        if landmark_model_path is None:
            landmark_model_path = os.path.join(model_dir, "face_landmarker.task")
        if not os.path.isfile(landmark_model_path):
            raise FileNotFoundError(
                f"MediaPipe task model not found: {landmark_model_path}"
            )

        base = python.BaseOptions(model_asset_path=landmark_model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        print(f"[M5] Binocular L2CS loaded on {self.device}")

    def _appearance_vector(self, bgr: np.ndarray, P: Any) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        points = _landmarks_to_xy(P, rgb.shape[1], rgb.shape[0])
        left = _crop_eye_rgb(rgb, points, LEFT_EYE)
        right = _crop_eye_rgb(rgb, points, RIGHT_EYE)
        if left is None or right is None:
            def eye_stats(indices):
                q = points[np.asarray(indices, dtype=np.int64)]
                return {
                    "finite": bool(np.isfinite(q).all()),
                    "min": np.nanmin(q, axis=0).tolist(),
                    "max": np.nanmax(q, axis=0).tolist(),
                    "width": float(np.nanmax(q[:, 0]) - np.nanmin(q[:, 0])),
                    "height": float(np.nanmax(q[:, 1]) - np.nanmin(q[:, 1])),
                }

            print("[M5 DEBUG] frame:", rgb.shape)
            print("[M5 DEBUG] landmarks:", points.shape)
            print("[M5 DEBUG] left eye:", eye_stats(LEFT_EYE))
            print("[M5 DEBUG] right eye:", eye_stats(RIGHT_EYE))
            raise RuntimeError("Eye crop diagnostic failed.")

        # Match notebook's horizontal-centre sorting before concatenation.
        centres = [
            (float(points[LEFT_EYE, 0].mean()), left),
            (float(points[RIGHT_EYE, 0].mean()), right),
        ]
        centres.sort(key=lambda item: item[0])
        pair = np.concatenate([centres[0][1], centres[1][1]], axis=2)

        # Match Cell 42: uint8 / 255, split channels, CHW, ImageNet normalization.
        eyes = []
        for crop in (pair[:, :, :3], pair[:, :, 3:6]):
            arr = crop.astype(np.float32) / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            eyes.append(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        tensor = torch.from_numpy(np.stack(eyes, axis=0)).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            yaw_logits, pitch_logits = self.model(tensor)
            yaw_prob = torch.softmax(yaw_logits, dim=-1)
            pitch_prob = torch.softmax(pitch_logits, dim=-1)
            yaw_bins = torch.as_tensor(self.yaw_bins, device=self.device)
            pitch_bins = torch.as_tensor(self.pitch_bins, device=self.device)
            yaw = (yaw_prob * yaw_bins).sum(dim=-1)[0].cpu().numpy()
            pitch = (pitch_prob * pitch_bins).sum(dim=-1)[0].cpu().numpy()

        # Return image-left then image-right to match crop ordering in training.
        return np.asarray([yaw[0], pitch[0], yaw[1], pitch[1]], dtype=np.float64)

    def extract(self, bgr: np.ndarray, P: Any, geom_dict: Dict[str, Any]) -> Dict[str, Any]:
        if bgr is None or bgr.size == 0:
            raise ValueError("Input frame is empty.")
        if "f" not in geom_dict:
            raise KeyError("geom_dict must contain the existing three geometric features in geom_dict['f'].")

        base_geom = np.asarray(geom_dict["f"], dtype=np.float64).reshape(-1)
        if base_geom.shape != (3,):
            raise ValueError(
                f"Expected exactly 3 geometric features [h, v_off, v_lid], got {base_geom.shape}"
            )

        points = _landmarks_to_xy(P, bgr.shape[1], bgr.shape[0])
        if len(points) <= 263:
            raise ValueError("At least 264 face landmarks are required.")

        eye_mid = 0.5 * (points[33] + points[263])
        inter_ocular = float(np.linalg.norm(points[263] - points[33])) + 1e-6
        nose = points[1]
        head_yaw_proxy = float((nose[0] - eye_mid[0]) / inter_ocular)
        head_pitch_proxy = float((nose[1] - eye_mid[1]) / inter_ocular)

        app = self._appearance_vector(bgr, P)
        geom_dict["f"] = np.concatenate([
            base_geom,
            np.asarray([head_yaw_proxy, head_pitch_proxy], dtype=np.float64),
            app,
        ])
        return geom_dict

    def close(self) -> None:
        if getattr(self, "landmarker", None) is not None:
            self.landmarker.close()


class PSOHybridMapper:
    """Weighted z-score scaling + quadratic Ridge regression."""

    def __init__(self, params: Optional[Dict[str, float]] = None):
        self.params = params or {
            "lam": 3.0, "w_geom": 1.2, "w_cnn": 0.6, "w_head": 0.3
        }
        self.W = self.mu = self.sd = None

    def _scale_features(self, F_norm: np.ndarray, p: Dict[str, float]) -> np.ndarray:
        S = F_norm.copy()
        S[:, 0:3] *= p["w_geom"]
        S[:, 3:5] *= p["w_head"]
        S[:, 5:9] *= p["w_cnn"]
        return np.clip(S, -3.5, 3.5)

    def _fit_with_params(self, F: np.ndarray, Y: np.ndarray, p: Dict[str, float]):
        mu = F.mean(axis=0)
        sd = np.maximum(F.std(axis=0), 0.015)
        S = self._scale_features((F - mu) / sd, p)
        X = _hybrid_design(S)
        penalty = p["lam"] * np.eye(X.shape[1])
        penalty[0, 0] = 0.0
        W = np.linalg.solve(X.T @ X + penalty, X.T @ Y)
        return W, mu, sd

    def fit(self, F: np.ndarray, Y: np.ndarray):
        F = np.asarray(F, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if F.ndim != 2 or F.shape[1] != 9:
            raise ValueError(f"F must have shape (N, 9), got {F.shape}")
        if Y.ndim != 2 or Y.shape[1] != 2 or len(Y) != len(F):
            raise ValueError("Y must have shape (N, 2), aligned with F.")
        if len(F) < 3 or not np.isfinite(F).all() or not np.isfinite(Y).all():
            raise ValueError("Calibration data must contain at least 3 finite samples.")
        self.W, self.mu, self.sd = self._fit_with_params(F, Y, self.params)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("Mapper has not been fitted.")
        F = np.atleast_2d(np.asarray(F, dtype=np.float64))
        if F.shape[1] != 9:
            raise ValueError(f"Expected 9 features, got {F.shape}")
        S = self._scale_features((F - self.mu) / self.sd, self.params)
        return _hybrid_design(S) @ self.W

    def to_json(self) -> Dict[str, Any]:
        if self.W is None:
            raise RuntimeError("Cannot serialize an unfitted mapper.")
        return {
            "params": self.params,
            "W": self.W.tolist(),
            "mu": self.mu.tolist(),
            "sd": self.sd.tolist(),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "PSOHybridMapper":
        mapper = cls(data["params"])
        mapper.W = np.asarray(data["W"], dtype=np.float64)
        mapper.mu = np.asarray(data["mu"], dtype=np.float64)
        mapper.sd = np.asarray(data["sd"], dtype=np.float64)
        if mapper.mu.shape != (9,) or mapper.sd.shape != (9,) or mapper.W.shape != (11, 2):
            raise ValueError("Serialized mapper has incompatible dimensions.")
        return mapper


def fit_best_pso(
    F: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    size: Sequence[float],
    n_particles: int = 24,
    n_iters: int = 30,
    seed: int = 42,
):
    """PSO tunes Ridge lambda and modality weights using leave-one-group-out CV."""
    F = np.asarray(F, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    groups = np.asarray(groups)
    size = np.asarray(size, dtype=np.float64)
    if F.ndim != 2 or F.shape[1] != 9:
        raise ValueError(f"Expected F shape (N, 9), got {F.shape}")
    if Y.shape != (len(F), 2) or len(groups) != len(F):
        raise ValueError("F, Y and groups must have aligned rows; Y must be (N, 2).")
    if size.shape != (2,) or np.any(size <= 0):
        raise ValueError("size must be (screen_width, screen_height), both positive.")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two distinct groups are required for PSO CV.")

    rng = np.random.default_rng(seed)
    lower = np.array([0.0, 0.5, 0.0, 0.0])
    upper = np.array([2.5, 2.5, 1.5, 0.8])

    def evaluate(position):
        params = {
            "lam": float(10.0 ** position[0]),
            "w_geom": float(position[1]),
            "w_cnn": float(position[2]),
            "w_head": float(position[3]),
        }
        errors = []
        for group in unique_groups:
            train_mask = groups != group
            test_mask = groups == group
            if train_mask.sum() < 15 or test_mask.sum() == 0:
                continue
            mapper = PSOHybridMapper(params).fit(F[train_mask], Y[train_mask])
            errors.append(np.linalg.norm(
                (mapper.predict(F[test_mask]) - Y[test_mask]) * size, axis=1
            ))
        if not errors:
            raise RuntimeError("No valid leave-one-group-out folds.")
        e = np.concatenate(errors)
        median = float(np.median(e))
        p80 = float(np.percentile(e, 80))
        return median + 0.3 * p80, median, p80

    positions = rng.uniform(lower, upper, size=(n_particles, 4))
    positions[0] = np.array([0.8, 1.2, 0.5, 0.2])
    velocities = np.zeros_like(positions)
    personal_positions = positions.copy()
    personal_scores = np.full(n_particles, np.inf)
    global_position = positions[0].copy()
    global_score = np.inf
    global_metrics = (float("nan"), float("nan"))

    for iteration in range(n_iters):
        inertia = 0.9 - 0.5 * iteration / max(1, n_iters - 1)
        for i in range(n_particles):
            score, median, p80 = evaluate(positions[i])
            if score < personal_scores[i]:
                personal_scores[i] = score
                personal_positions[i] = positions[i].copy()
            if score < global_score:
                global_score = score
                global_position = positions[i].copy()
                global_metrics = (median, p80)
        r1 = rng.random((n_particles, 4))
        r2 = rng.random((n_particles, 4))
        velocities = (
            inertia * velocities
            + 1.6 * r1 * (personal_positions - positions)
            + 1.6 * r2 * (global_position - positions)
        )
        positions = np.clip(positions + velocities, lower, upper)

    best = {
        "lam": float(10.0 ** global_position[0]),
        "w_geom": float(global_position[1]),
        "w_cnn": float(global_position[2]),
        "w_head": float(global_position[3]),
    }
    print(
        f"[M5 PSO] params={best}; median={global_metrics[0]:.2f}px; "
        f"P80={global_metrics[1]:.2f}px"
    )
    return PSOHybridMapper(best).fit(F, Y), global_metrics[0], global_metrics[1]
