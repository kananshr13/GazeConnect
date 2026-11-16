#!/usr/bin/env python3
"""
GazeConnect Stabilized Hybrid Appearance-Geometric Gaze Estimator + PSO Calibrator
Fixes head-drift z-score amplification and aligns mirrored webcam geometry with CNN predictions.
"""
import os
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models


class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.regressor(self.features(x))


class L2CSStyle(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.yaw_head = nn.Linear(512, 7)
        self.pitch_head = nn.Linear(512, 3)
        self.register_buffer("yaw_bins", torch.linspace(-1.0, 1.0, 7))
        self.register_buffer("pitch_bins", torch.linspace(-1.0, 1.0, 3))

    def forward(self, x):
        feat = self.backbone(x)
        yaw_prob = torch.softmax(self.yaw_head(feat), dim=1)
        pitch_prob = torch.softmax(self.pitch_head(feat), dim=1)
        yaw_exp = torch.sum(yaw_prob * self.yaw_bins, dim=1)
        pitch_exp = torch.sum(pitch_prob * self.pitch_bins, dim=1)
        return torch.stack([yaw_exp, pitch_exp], dim=1)


LEFT_EYE_CONTOUR = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155, 246]
RIGHT_EYE_CONTOUR = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382, 466]


class HybridFeatureExtractor:
    def __init__(self, model_dir, arch="l2cs"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.arch = arch
        self.cnn = None

        ckpt_name = "l2cs_style_best.pth" if arch == "l2cs" else "basic_cnn_best.pth"
        ckpt_path = os.path.join(model_dir, ckpt_name)

        if os.path.isfile(ckpt_path):
            model = L2CSStyle() if arch == "l2cs" else BasicCNN()
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state, strict=False)
            model.eval().to(self.device)
            self.cnn = model
            print(f"[HybridGaze] Loaded appearance backbone: {ckpt_name} on {self.device}")

    def _crop_eye(self, bgr, P, indices, size=64):
        h_img, w_img = bgr.shape[:2]
        pts = P[indices]
        cx, cy = pts.mean(axis=0)
        span = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min())
        half = int(max(16, span * 0.85))

        x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
        x1, y1 = min(w_img, int(cx + half)), min(h_img, int(cy + half))
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return np.zeros((size, size, 3), dtype=np.float32)

        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (rgb - mean) / std

    def _appearance_vector(self, bgr, P):
        if self.cnn is None:
            return np.zeros(4, dtype=np.float64)
        left_crop = self._crop_eye(bgr, P, LEFT_EYE_CONTOUR)
        right_crop = self._crop_eye(bgr, P, RIGHT_EYE_CONTOUR)
        batch = np.stack([left_crop.transpose(2, 0, 1), right_crop.transpose(2, 0, 1)], axis=0)
        tensor = torch.from_numpy(batch).to(self.device)
        with torch.inference_mode():
            out = self.cnn(tensor).cpu().numpy().astype(np.float64)
        return out.flatten()

    def extract(self, bgr, P, geom_dict):
        """
        Extracts 9 clean, bounded eye-in-head features:
          - 3 geometric eye ratios: [h, v_off, v_lid]
          - 2 relative head-pose angles (yaw_head, pitch_head from nose-to-eye geometry)
          - 4 CNN binocular appearance predictions: [yaw_L, pitch_L, yaw_R, pitch_R]
        """
        # Bounded head-orientation proxy using nose tip (1) relative to eye midpoint
        eye_mid = 0.5 * (P[33] + P[263])
        inter_ocular = float(np.linalg.norm(P[263] - P[33])) + 1e-6
        nose = P[1]
        head_yaw_proxy = float((nose[0] - eye_mid[0]) / inter_ocular)
        head_pitch_proxy = float((nose[1] - eye_mid[1]) / inter_ocular)

        pose_vec = np.array([head_yaw_proxy, head_pitch_proxy], dtype=np.float64)
        app_vec = self._appearance_vector(bgr, P)

        geom_dict["f"] = np.concatenate([geom_dict["f"], pose_vec, app_vec])
        return geom_dict


def _hybrid_design(F):
    """
    F columns:
      0:h, 1:v_off, 2:v_lid,
      3:head_yaw, 4:head_pitch,
      5:yL, 6:pL, 7:yR, 8:pR
    """
    h, v, l = F[:, 0], F[:, 1], F[:, 2]
    hy, hp = F[:, 3], F[:, 4]
    y_cnn = 0.5 * (F[:, 5] + F[:, 7])
    p_cnn = 0.5 * (F[:, 6] + F[:, 8])
    return np.stack([
        np.ones_like(h),
        h, v, l,
        h * h, v * v, h * v,
        y_cnn, p_cnn,
        hy, hp
    ], axis=1)


class PSOHybridMapper:
    """
    Stabilized PSO-calibrated Hybrid Mapper.
    Clamps standard deviations so low-variance head features never explode on slight movement.
    """
    def __init__(self, params=None):
        self.params = params or {"lam": 3.0, "w_geom": 1.2, "w_cnn": 0.6, "w_head": 0.3}
        self.W = None
        self.mu = None
        self.sd = None

    def _scale_features(self, F_norm, p):
        S = F_norm.copy()
        S[:, 0:3] *= p["w_geom"]
        S[:, 3:5] *= p["w_head"]
        S[:, 5:9] *= p["w_cnn"]
        # Soft-clip extreme z-scores beyond +/- 3.5 sigma so cursor never flies off-screen
        return np.clip(S, -3.5, 3.5)

    def _fit_with_params(self, F, Y, p):
        mu = F.mean(0)
        # Floor sd at 0.015 so stationary calibration features never blow up at runtime
        sd = np.maximum(F.std(0), 0.015)
        S = self._scale_features((F - mu) / sd, p)
        X = _hybrid_design(S)
        R = p["lam"] * np.eye(X.shape[1])
        R[0, 0] = 0.0
        W = np.linalg.solve(X.T @ X + R, X.T @ Y)
        return W, mu, sd

    def fit(self, F, Y):
        self.W, self.mu, self.sd = self._fit_with_params(F, Y, self.params)
        return self

    def predict(self, F):
        F = np.atleast_2d(F)
        S = self._scale_features((F - self.mu) / self.sd, self.params)
        return _hybrid_design(S) @ self.W

    def to_json(self):
        return {
            "params": self.params,
            "W": self.W.tolist(),
            "mu": self.mu.tolist(),
            "sd": self.sd.tolist(),
        }

    @classmethod
    def from_json(cls, d):
        m = cls(d.get("params"))
        m.W = np.array(d["W"])
        m.mu = np.array(d["mu"])
        m.sd = np.array(d["sd"])
        return m


def fit_best_pso(F, Y, groups, size, n_particles=24, n_iters=30, seed=42):
    rng = np.random.default_rng(seed)
    unique_g = np.unique(groups)

    # Search bounds: [log10(lam), w_geom, w_cnn, w_head]
    lb = np.array([0.0,  0.5, 0.0, 0.0])
    ub = np.array([2.5,  2.5, 1.5, 0.8])

    def evaluate(pos):
        p = {
            "lam": float(10.0 ** pos[0]),
            "w_geom": float(pos[1]),
            "w_cnn": float(pos[2]),
            "w_head": float(pos[3]),
        }
        errs = []
        for g in unique_g:
            tr, te = groups != g, groups == g
            if tr.sum() < 15 or te.sum() < 1:
                continue
            m = PSOHybridMapper(p).fit(F[tr], Y[tr])
            errs.append(np.linalg.norm((m.predict(F[te]) - Y[te]) * size, axis=1))
        e = np.concatenate(errs)
        med = float(np.median(e))
        p80 = float(np.percentile(e, 80))
        return med + 0.3 * p80, med, p80

    X = rng.uniform(lb, ub, size=(n_particles, 4))
    X[0] = np.array([0.8, 1.2, 0.5, 0.2])
    V = np.zeros_like(X)

    pbest_X = X.copy()
    pbest_fit = np.full(n_particles, np.inf)
    gbest_X = X[0].copy()
    gbest_fit = np.inf
    gbest_metrics = (0.0, 0.0)

    for it in range(n_iters):
        w = 0.9 - 0.5 * (it / max(1, n_iters - 1))
        for i in range(n_particles):
            fit, med, p80 = evaluate(X[i])
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_X[i] = X[i].copy()
            if fit < gbest_fit:
                gbest_fit = fit
                gbest_X = X[i].copy()
                gbest_metrics = (med, p80)

        r1, r2 = rng.random((n_particles, 4)), rng.random((n_particles, 4))
        V = w * V + 1.6 * r1 * (pbest_X - X) + 1.6 * r2 * (gbest_X - X)
        X = np.clip(X + V, lb, ub)

    best_params = {
        "lam": float(10.0 ** gbest_X[0]),
        "w_geom": float(gbest_X[1]),
        "w_cnn": float(gbest_X[2]),
        "w_head": float(gbest_X[3]),
    }
    print(f"[PSO Calibration] Optimal params: {best_params} | "
          f"Median Err: {gbest_metrics[0]:.1f} px | P80 Err: {gbest_metrics[1]:.1f} px")
    return PSOHybridMapper(best_params).fit(F, Y), gbest_metrics[0], gbest_metrics[1]