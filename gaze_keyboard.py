#!/usr/bin/env python3
"""
GazeConnect - Hybrid Deep Learning + Geometric Gaze Keyboard (English & Hindi)

Pipeline:
  webcam -> MediaPipe Face Landmarker (478 pts incl. iris)
         -> HybridFeatureExtractor:
              1. Geometric eye ratios (iris position inside each eye)
              2. Head-pose compensation features (cx, cy, scale, roll)
              3. Binocular 64x64 eye crops -> L2CS-Style / BasicCNN appearance features
         -> 9-point calibration + Particle Swarm Optimization (PSO) Ridge Regression
         -> speed-adaptive EMA smoothing, blink pause
         -> dwell selection (progress ring) on an on-screen keyboard

Usage:
    python gaze_keyboard.py                  # Hybrid L2CS + Geometry + PSO (default)
    python gaze_keyboard.py --arch basic     # Hybrid BasicCNN + Geometry + PSO
    python gaze_keyboard.py --arch geom      # Original pure-geometric baseline
    python gaze_keyboard.py --load           # Reuse the last saved calibration
    python gaze_keyboard.py --mouse          # Test UI with mouse (no webcam)

Keys while running:
    R = recalibrate | P = webcam preview | - / = = dwell time
    SPACE = start (on intro / result screens) | ESC = quit
"""
import argparse
import base64
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import font as tkfont

import cv2
import numpy as np

from hybrid_gaze import HybridFeatureExtractor, PSOHybridMapper, fit_best_pso

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
MODEL_CANDIDATES = [
    os.path.join(MODEL_DIR, "face_landmarker.task"),
    os.path.expanduser("~/GazeConnect/models/face_landmarker.task"),
]
CALIB_FILE = os.path.join(HERE, "gaze_calibration.json")

HINDI_VOICE = "Lekha"          # macOS Hindi voice (System Settings > Accessibility > Spoken Content)
DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# ----------------------------------------------------------------------------
# Face landmarks and eye features
# ----------------------------------------------------------------------------
# (image-left corner, image-right corner, upper lid, lower lid, iris centre)
EYE_A = (33, 133, 159, 145, 468)    # subject's right eye, appears on the image left
EYE_B = (362, 263, 386, 374, 473)   # subject's left eye, appears on the image right


class FaceTracker:
    """MediaPipe Tasks API; falls back to legacy solutions."""

    def __init__(self, model_path=None):
        import mediapipe as mp
        self.mp = mp
        self.t0 = time.monotonic()
        self.last_ts = -1
        path = model_path or next((p for p in MODEL_CANDIDATES if os.path.isfile(p)), None)
        if path:
            opts = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=path),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
            )
            self.lm = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
            self.mode = "tasks"
        elif hasattr(mp, "solutions"):
            self.lm = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.mode = "solutions"
        else:
            raise FileNotFoundError(
                "face_landmarker.task not found. Expected at models/face_landmarker.task")

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.mode == "tasks":
            ts = max(int((time.monotonic() - self.t0) * 1000), self.last_ts + 1)
            self.last_ts = ts
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.lm.detect_for_video(img, ts)
            if not res.face_landmarks:
                return None
            pts = res.face_landmarks[0]
        else:
            res = self.lm.process(rgb)
            if not res.multi_face_landmarks:
                return None
            pts = res.multi_face_landmarks[0].landmark
        if len(pts) < 478:
            return None
        return np.array([[p.x * w, p.y * h] for p in pts], dtype=np.float32)

    def close(self):
        try:
            self.lm.close()
        except Exception:
            pass


def eye_features(P, idx):
    a, b, up, lo, iris = idx
    A, B, I = P[a].astype(float), P[b].astype(float), P[iris].astype(float)
    d = B - A
    w = float(np.linalg.norm(d)) + 1e-6
    u = d / w
    n = np.array([-u[1], u[0]])                 # perpendicular (towards the chin)
    mid = (A + B) / 2
    h = float(np.dot(I - A, u) / w)             # 0 .. 1 across the eye
    v_off = float(np.dot(I - mid, n) / w)       # iris offset from the eye axis
    lid_up = float(np.dot(P[up] - mid, n))
    lid_lo = float(np.dot(P[lo] - mid, n))
    opening = (lid_lo - lid_up) / w             # eye-aspect ratio (blink signal)
    v_lid = (float(np.dot(I - mid, n)) - lid_up) / (lid_lo - lid_up + 1e-6)
    return h, v_off, v_lid, opening


def face_features(P):
    """Returns dict(f=[h, v_off, v_lid], ear, center, scale) or None."""
    if P is None or len(P) < 478:
        return None
    ha, va, la, oa = eye_features(P, EYE_A)
    hb, vb, lb, ob = eye_features(P, EYE_B)
    xs = P[:, 0]
    ys = P[:, 1]
    return {
        "f": np.array([(ha + hb) / 2, (va + vb) / 2, (la + lb) / 2]),
        "ear": (oa + ob) / 2,
        "cx": float(np.mean(xs)),
        "cy": float(np.mean(ys)),
        "scale": float(xs.max() - xs.min()),
    }


class BlinkDetector:
    def __init__(self):
        self.hist = deque(maxlen=90)

    def update(self, ear):
        self.hist.append(ear)
        base = np.percentile(self.hist, 80) if len(self.hist) >= 10 else ear
        return ear < 0.62 * base


# ----------------------------------------------------------------------------
# Legacy Geometric Calibration Mapper (kept for --arch geom baseline comparison)
# ----------------------------------------------------------------------------
def _design(F):
    h, v, l = F[:, 0], F[:, 1], F[:, 2]
    return np.stack([np.ones_like(h), h, v, l, h * h, v * v, h * v], axis=1)


class GazeMapper:
    """Eye features -> normalised screen position, ridge-regularised quadratic fit."""

    def __init__(self, lam=1.0):
        self.lam = lam
        self.W = None
        self.mu = None
        self.sd = None

    def fit(self, F, Y):
        self.mu = F.mean(0)
        self.sd = F.std(0) + 1e-6
        X = _design((F - self.mu) / self.sd)
        R = self.lam * np.eye(X.shape[1])
        R[0, 0] = 0.0
        self.W = np.linalg.solve(X.T @ X + R, X.T @ Y)
        return self

    def predict(self, F):
        F = np.atleast_2d(F)
        return _design((F - self.mu) / self.sd) @ self.W

    def to_json(self):
        return {"lam": self.lam, "W": self.W.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def from_json(cls, d):
        m = cls(d["lam"])
        m.W, m.mu, m.sd = np.array(d["W"]), np.array(d["mu"]), np.array(d["sd"])
        return m


def cross_validate(F, Y, groups, lam, size):
    errs = []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if tr.sum() < 10 or te.sum() < 1:
            continue
        m = GazeMapper(lam).fit(F[tr], Y[tr])
        errs.append(np.linalg.norm((m.predict(F[te]) - Y[te]) * size, axis=1))
    e = np.concatenate(errs)
    return float(np.median(e)), float(np.percentile(e, 80))


def fit_best(F, Y, groups, size):
    best = None
    for lam in (0.1, 0.3, 1.0, 3.0, 10.0):
        med, p80 = cross_validate(F, Y, groups, lam, size)
        if best is None or med < best[1]:
            best = (lam, med, p80)
    lam, med, p80 = best
    return GazeMapper(lam).fit(F, Y), med, p80


# ----------------------------------------------------------------------------
# Smoothing
# ----------------------------------------------------------------------------
class Smoother:
    """One-Euro Filter: ultra-steady during fixations, smooth and responsive during eye saccades."""

    def __init__(self, min_cutoff=0.45, beta=0.004, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _alpha(self, dt, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-3))

    def __call__(self, x, y):
        now = time.perf_counter()
        q = np.array([x, y], dtype=float)
        if self.x_prev is None:
            self.x_prev = q
            self.dx_prev = np.zeros(2, dtype=float)
            self.t_prev = now
            return float(q[0]), float(q[1])

        dt = max(now - self.t_prev, 1e-3)
        self.t_prev = now

        # Filtered velocity
        dx = (q - self.x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Speed-adaptive cutoff frequency
        speed = float(np.linalg.norm(dx_hat))
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(dt, cutoff)
        x_hat = a * q + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return float(x_hat[0]), float(x_hat[1])


# ----------------------------------------------------------------------------
# Keyboard layouts
# ----------------------------------------------------------------------------
def M(c):
    """Combining mark shown on a dotted circle, inserts only the mark."""
    return ("\u25cc" + c, c)


BOARD = [("Water", "पानी"), ("Food", "खाना"), ("Pain", "दर्द"), ("Doctor", "डॉक्टर"),
         ("Washroom", "शौचालय"), ("Yes", "हाँ"), ("No", "नहीं"), ("Help", "मदद")]

EN_LARGE = [
    [list("ABCDE"), list("FGHIJ"), list("KLMNO")],
    [list("PQRST"), list("UVWXY"), ["Z", ".", ",", "?", "!"]],
]
EN_DENSE = [[list("QWERTYUIOP"), list("ASDFGHJKL'"), list("ZXCVBNM,.?")]]

HI_LARGE = [
    [["अ", "आ", "इ", "ई", "उ"], ["ऊ", "ए", "ऐ", "ओ", "औ"], ["ऋ", "अं", "अः", "।", "?"]],
    [[M("ा"), M("ि"), M("ी"), M("ु"), M("ू")], [M("े"), M("ै"), M("ो"), M("ौ"), M("ं")],
     [M("ः"), M("्"), M("ँ"), M("ृ"), M("़")]],
    [["क", "ख", "ग", "घ", "ङ"], ["च", "छ", "ज", "झ", "ञ"], ["ट", "ठ", "ड", "ढ", "ण"]],
    [["त", "थ", "द", "ध", "न"], ["प", "फ", "ब", "भ", "म"], ["य", "र", "ल", "व", "श"]],
    [["ष", "स", "ह", "क्ष", "त्र"], ["ज्ञ", "श्र", "ळ", "ड़", "ढ़"], ["ज़", "फ़", "क़", "ख़", "ग़"]],
]
HI_DENSE = [[
    ["अ", "आ", "इ", "ई", "उ", "ऊ", "ए", "ऐ", "ओ", "औ"],
    ["क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ"],
    ["ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न"],
    ["प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श"],
    ["ष", "स", "ह", "क्ष", "त्र", "ज्ञ", "श्र", "।", "?", M("्")],
    [M("ा"), M("ि"), M("ी"), M("ु"), M("ू"), M("े"), M("ै"), M("ो"), M("ौ"), M("ं")],
]]


def key_from_spec(spec):
    if isinstance(spec, tuple):
        if len(spec) == 2 and not spec[0].isascii() and spec[0].startswith("\u25cc"):
            return {"label": spec[0], "act": ("char", spec[1])}
        return {"label": spec[0] + "\n" + spec[1], "act": ("phrase", spec[0], spec[1]),
                "en": spec[0], "hi": spec[1]}
    return {"label": spec, "act": ("char", spec)}


# ----------------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------------
BG, KEY, KEY_HOT, KEY_OK = "#0F172A", "#1E293B", "#334155", "#10B981"
NAV, NAV_ON, TXT, DIM, RING = "#1D3557", "#2563EB", "#F8FAFC", "#94A3B8", "#34D399"
CAL_PTS = [(0.5, 0.5), (0.08, 0.08), (0.5, 0.08), (0.92, 0.08), (0.92, 0.5),
           (0.92, 0.92), (0.5, 0.92), (0.08, 0.92), (0.08, 0.5)]
SETTLE, COLLECT = 0.9, 1.4


class App:
    def __init__(self, args):
        self.args = args
        self.use_hybrid = (args.arch != "geom")
        self.root = tk.Tk()
        self.root.title("GazeConnect")
        self.root.configure(bg=BG)
        if not args.windowed:
            self.root.attributes("-fullscreen", True)
        else:
            self.root.geometry("1280x800")
        self.cv = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        self.root.update()
        self.W, self.H = self.cv.winfo_width(), self.cv.winfo_height()
        mm = max(self.root.winfo_screenmmwidth(), 1)
        self.px_per_cm = self.root.winfo_screenwidth() / (mm / 10.0)

        fams = set(tkfont.families())
        self.hi_family = next((f for f in ("Kohinoor Devanagari", "Devanagari Sangam MN",
                                           "Noto Sans Devanagari", "Mangal", "Nirmala UI")
                               if f in fams), tkfont.nametofont("TkDefaultFont").actual("family"))
        self.en_family = tkfont.nametofont("TkDefaultFont").actual("family")
        self._fonts = {}

        # state
        self.state = "intro"
        self.screen, self.style, self.page = "board", "large", 0
        self.text = ""
        self.keys, self.cur_key, self.flash_key = [], None, None
        self.dwell = args.dwell
        self.dwell_start, self.cool_until = 0.0, 0.0
        self._last_now = None
        self.progress = 0.0
        self.locked_act = None
        self.gaze = None
        self.warn = ""
        self.show_preview = True
        self.preview_img = None
        self.frame_i = 0
        self.last_feats = None
        self.cal_err = None
        self.smoother = Smoother()
        self.blinker = BlinkDetector()
        self.mapper = None
        self.ref = None
        self.cap = None
        self.tracker = None
        self.hybrid_ext = None
        self.cal_i, self.cal_t0 = 0, 0.0
        self.cal_F, self.cal_Y, self.cal_G, self.cal_ref = [], [], [], []
        self.pt_count = 0

        if args.mouse:
            self.state = "run"
        else:
            self.tracker = FaceTracker()
            if self.use_hybrid:
                self.hybrid_ext = HybridFeatureExtractor(MODEL_DIR, arch=args.arch)
            self.cap = cv2.VideoCapture(args.cam)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open the webcam (check camera permission for Terminal).")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if args.load and self.load_calibration():
                self.state = "run"

        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("<space>", self.on_space)
        self.root.bind("<Key-r>", lambda e: self.start_calibration())
        self.root.bind("<Key-p>", lambda e: self.toggle_preview())
        self.root.bind("<Key-minus>", lambda e: self.change_dwell(-0.1))
        self.root.bind("<Key-equal>", lambda e: self.change_dwell(+0.1))
        self.cv.bind("<Configure>", self.on_resize)
        self.build_layout()
        self.root.after(10, self.tick)

    # ---------------------------------------------------------------- helpers
    def font(self, family, px, bold=False):
        k = (family, int(px), bold)
        if k not in self._fonts:
            self._fonts[k] = tkfont.Font(family=family, size=-max(8, int(px)),
                                         weight="bold" if bold else "normal")
        return self._fonts[k]

    def quit(self):
        try:
            if self.cap:
                self.cap.release()
            if self.tracker:
                self.tracker.close()
        finally:
            self.root.destroy()

    def on_resize(self, e):
        if (e.width, e.height) != (self.W, self.H) and e.width > 100:
            self.W, self.H = e.width, e.height
            if self.state == "run":
                self.build_layout()

    def on_space(self, _):
        if self.state in ("intro", "result"):
            if self.state == "intro":
                self.start_calibration()
            else:
                self.state = "run"
                self.smoother.reset()
                self.build_layout()

    def toggle_preview(self):
        self.show_preview = not self.show_preview
        self.cv.delete("prev")

    def change_dwell(self, d):
        self.dwell = round(min(3.0, max(0.5, self.dwell + d)), 1)
        self.draw_status()

    # -------------------------------------------------------- feature wrapper
    def extract_all_features(self, frame, P):
        ft = face_features(P)
        if ft is not None and self.use_hybrid and self.hybrid_ext is not None:
            ft = self.hybrid_ext.extract(frame, P, ft)
        return ft

    # ---------------------------------------------------------- calibration IO
    def save_calibration(self):
        d = {
            "arch": self.args.arch,
            "mapper": self.mapper.to_json(),
            "ref": self.ref,
            "err_px": self.cal_err,
            "size": [self.W, self.H],
        }
        with open(CALIB_FILE, "w") as f:
            json.dump(d, f)

    def load_calibration(self):
        try:
            with open(CALIB_FILE) as f:
                d = json.load(f)
            saved_arch = d.get("arch", "geom")
            if saved_arch != self.args.arch:
                print(f"[GazeConnect] Saved calibration is for '{saved_arch}', current mode is '{self.args.arch}'. Recalibration required.")
                return False
            if self.use_hybrid:
                self.mapper = PSOHybridMapper.from_json(d["mapper"])
            else:
                self.mapper = GazeMapper.from_json(d["mapper"])
            self.ref, self.cal_err = d["ref"], d.get("err_px")
            return True
        except Exception as exc:
            print(f"[GazeConnect] Could not load calibration: {exc}")
            return False

    def start_calibration(self):
        if self.args.mouse:
            return
        self.state = "calib"
        self.cal_i, self.cal_t0, self.pt_count = 0, time.perf_counter(), 0
        self.cal_F, self.cal_Y, self.cal_G, self.cal_ref = [], [], [], []
        self.cv.delete("all")

    def finish_calibration(self):
        F, Y, G = np.array(self.cal_F), np.array(self.cal_Y), np.array(self.cal_G)
        screen_size = np.array([self.W, self.H], float)
        if self.use_hybrid:
            self.mapper, med, p80 = fit_best_pso(F, Y, G, screen_size)
        else:
            self.mapper, med, p80 = fit_best(F, Y, G, screen_size)
        r = np.array(self.cal_ref)
        self.ref = {"cx": float(r[:, 0].mean()), "cy": float(r[:, 1].mean()), "scale": float(r[:, 2].mean())}
        self.cal_err = {"median_px": med, "p80_px": p80}
        self.save_calibration()
        self.state = "result"

    # ------------------------------------------------------------- gaze input
    def read_gaze(self, now):
        """returns (x, y, blink) in canvas px; x is None if no usable gaze."""
        if self.args.mouse:
            x = self.root.winfo_pointerx() - self.cv.winfo_rootx()
            y = self.root.winfo_pointery() - self.cv.winfo_rooty()
            return float(x), float(y), False
        ok, frame = self.cap.read()
        if not ok:
            return None, None, False
        P = self.tracker.detect(frame)
        ft = self.extract_all_features(frame, P)
        self.frame_i += 1
        if self.show_preview and self.frame_i % 3 == 0:
            self.update_preview(frame, P)
        self.last_feats = ft
        if ft is None:
            return None, None, False
        blink = self.blinker.update(ft["ear"])
        self.warn = ""
        if self.ref and self.state == "run":
            drift = math.hypot(ft["cx"] - self.ref["cx"], ft["cy"] - self.ref["cy"]) / frame.shape[1]
            # Hybrid model includes head-pose terms, so allow slightly wider drift tolerance
            max_drift = 0.12 if self.use_hybrid else 0.07
            max_scale = 0.22 if self.use_hybrid else 0.15
            if drift > max_drift or abs(ft["scale"] / self.ref["scale"] - 1) > max_scale:
                self.warn = "Large head shift - press R to recalibrate"
        if self.mapper is None:
            return None, None, blink
        g = self.mapper.predict(ft["f"])[0] * np.array([self.W, self.H])
        return float(np.clip(g[0], 0, self.W)), float(np.clip(g[1], 0, self.H)), blink

    def update_preview(self, frame, P):
        small = cv2.resize(cv2.flip(frame, 1), (112, 84))
        if P is not None:
            sx, sy = 112 / frame.shape[1], 84 / frame.shape[0]
            for i in (468, 473):
                cv2.circle(small, (int(112 - P[i][0] * sx), int(P[i][1] * sy)), 2, (0, 0, 255), -1)
        ok, buf = cv2.imencode(".png", small)
        if ok:
            self.preview_img = tk.PhotoImage(data=base64.b64encode(buf.tobytes()))
            self.cv.delete("prev")
            self.cv.create_image(self.W * 0.89, max(6, int(0.012 * min(self.W, self.H))) + 42,
                                 image=self.preview_img, tags="prev")

    # ------------------------------------------------------------------- tick
    def tick(self):
        t0 = time.perf_counter()
        try:
            self.step(t0)
        except tk.TclError:
            return
        delay = max(1, 12 - int((time.perf_counter() - t0) * 1000))
        self.root.after(delay, self.tick)

    def step(self, now):
        if self.state in ("intro", "result"):
            if self.cap:
                self.cap.read()          # keep the camera buffer fresh
            self.draw_message()
            return
        if self.state == "calib":
            self.step_calibration(now)
            return
        x, y, blink = self.read_gaze(now)
        if x is None:
            self.gaze = None
            if not self.args.mouse:
                self.warn = "Face not found"
            self.handle_gaze(None, None, False, now)
        else:
            if not blink:
                x, y = self.smoother(x, y) if not self.args.mouse else (x, y)
                self.gaze = (x, y)
            self.handle_gaze(*(self.gaze or (x, y)), blink, now)
        self.draw_dynamic()

    def step_calibration(self, now):
        if self.cal_i >= len(CAL_PTS):
            self.finish_calibration()
            return
        px, py = CAL_PTS[self.cal_i]
        ok, frame = self.cap.read()
        ft = None
        if ok:
            P = self.tracker.detect(frame)
            ft = self.extract_all_features(frame, P)
            self.last_feats = ft
        t = now - self.cal_t0
        blink = self.blinker.update(ft["ear"]) if ft else True
        if t > SETTLE and ft and not blink:
            self.cal_F.append(ft["f"])
            self.cal_Y.append([px, py])
            self.cal_G.append(self.cal_i)
            self.cal_ref.append([ft["cx"], ft["cy"], ft["scale"]])
            self.pt_count += 1
        if t > SETTLE + COLLECT:
            if self.pt_count >= 12:
                self.cal_i += 1
            else:                        # not enough good frames: drop and repeat this point
                keep = len(self.cal_G) - self.pt_count
                self.cal_F, self.cal_Y = self.cal_F[:keep], self.cal_Y[:keep]
                self.cal_G, self.cal_ref = self.cal_G[:keep], self.cal_ref[:keep]
            self.pt_count, self.cal_t0 = 0, now
        # draw
        self.cv.delete("all")
        cx, cy = px * self.W, py * self.H
        r = 34 - 22 * min(1.0, t / SETTLE) if t < SETTLE else 12
        self.cv.create_oval(cx - r, cy - r, cx + r, cy + r, outline=RING, width=4)
        self.cv.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill="#F8FAFC", outline="")
        self.cv.create_text(
            self.W / 2, self.H * 0.5 + (60 if abs(py - 0.5) > 0.2 else 120),
            text=f"Look at the dot and keep your head still   ({min(self.cal_i + 1, len(CAL_PTS))}/{len(CAL_PTS)})",
            fill=DIM, font=self.font(self.en_family, 24)
        )
        if not ft:
            self.cv.create_text(self.W / 2, self.H * 0.9, text="Face not found", fill="#F87171",
                                font=self.font(self.en_family, 28, True))

    # --------------------------------------------------------------- screens
    def draw_message(self):
        self.cv.delete("all")
        f = lambda s, b=False: self.font(self.en_family, s, b)
        mode_tag = f"Hybrid ({self.args.arch.upper()} + Geometry + PSO)" if self.use_hybrid else "Geometric Baseline"
        if self.state == "intro":
            lines = [("GazeConnect", 64, True, TXT),
                     (f"Gaze keyboard - English & Hindi  [{mode_tag}]", 26, False, DIM),
                     ("", 10, False, DIM),
                     ("1. Sit about 50-70 cm from the screen with your face well lit.", 24, False, TXT),
                     ("2. Look at each dot until it moves. Keep your head still.", 24, False, TXT),
                     ("3. Select keys by looking at them until the ring fills.", 24, False, TXT),
                     ("", 10, False, DIM),
                     ("Press SPACE to start calibration", 30, True, RING)]
        else:
            e = self.cal_err or {}
            med = e.get("median_px", 0)
            cm = med / self.px_per_cm
            lines = [(f"Calibration complete ({mode_tag})", 46, True, TXT),
                     (f"Typical error (cross-validated): about {med:.0f} px  (~{cm:.1f} cm on this screen)", 26, False, TXT),
                     ("Keys are much larger than this error in the 'Large' layout." if cm > 2.5
                      else "Accuracy is good enough to try the QWERTY / Dense layout.", 22, False, DIM),
                     ("", 10, False, DIM),
                     ("SPACE = start    R = recalibrate", 30, True, RING)]
        y = self.H * 0.22
        for text, size, bold, col in lines:
            self.cv.create_text(self.W / 2, y, text=text, fill=col, font=f(size, bold))
            y += size * 1.9

    def pages(self):
        if self.screen == "board":
            return [[[("Water", "पानी"), ("Food", "खाना"), ("Pain", "दर्द"), ("Doctor", "डॉक्टर")],
                     [("Washroom", "शौचालय"), ("Yes", "हाँ"), ("No", "नहीं"), ("Help", "मदद")]]]
        if self.screen == "en":
            return EN_LARGE if self.style == "large" else EN_DENSE
        return HI_LARGE if self.style == "large" else HI_DENSE

    def build_layout(self):
        if self.state != "run":
            return
        cv, W, H = self.cv, self.W, self.H
        cv.delete("all")
        self.keys, self.cur_key, self.flash_key = [], None, None
        m = max(6, int(0.012 * min(W, H)))
        gap = m
        top_h, text_h, bot_h, st_h = int(0.10 * H), int(0.15 * H), int(0.12 * H), int(0.035 * H)
        pages = self.pages()
        self.page %= len(pages)
        rows = pages[self.page]

        # top navigation row (right 20 % left free for the webcam preview)
        nav = [("Needs", "@board"), ("English", "@en"), ("हिन्दी", "@hi"),
               ("Keys: " + ("Large" if self.style == "large" else "Dense"), "@style")]
        nav_w = int(W * 0.78)
        kw = (nav_w - 2 * m - (len(nav) - 1) * gap) / len(nav)
        for i, (lab, tok) in enumerate(nav):
            on = (tok == "@" + self.screen)
            self.add_key(m + i * (kw + gap), m, m + i * (kw + gap) + kw, m + top_h,
                         {"label": lab, "act": ("tok", tok[1:])}, NAV_ON if on else NAV, hi=(tok == "@hi"))
        # text strip
        ty0, ty1 = m + top_h + gap, m + top_h + gap + text_h
        cv.create_rectangle(m, ty0, W - m, ty1, fill="#020617", outline="#334155", width=2, tags="ui")
        self.text_item = cv.create_text(m + 20, (ty0 + ty1) / 2, anchor="w", text=self.text + "|",
                                        fill=TXT, font=self.font(self.hi_family, text_h * 0.38), tags="ui",
                                        width=W - 2 * m - 40)
        # main grid
        gy0, gy1 = ty1 + gap, H - m - bot_h - gap - st_h
        rh = (gy1 - gy0 - (len(rows) - 1) * gap) / len(rows)
        for ri, row in enumerate(rows):
            n = len(row)
            cw = (W - 2 * m - (n - 1) * gap) / n
            for ci, spec in enumerate(row):
                k = key_from_spec(spec)
                x0 = m + ci * (cw + gap)
                y0 = gy0 + ri * (rh + gap)
                self.add_key(x0, y0, x0 + cw, y0 + rh, k, KEY)
        # bottom action row
        bot = [("Back", "back", 1), ("Space", "space", 2.2), ("Clear", "clear", 1), ("Speak", "speak", 1.3)]
        if len(pages) > 1:
            bot.append((f"Next {self.page + 1}/{len(pages)}", "next", 1.2))
        tot = sum(b[2] for b in bot)
        unit = (W - 2 * m - (len(bot) - 1) * gap) / tot
        x = m
        by0 = H - m - bot_h - st_h
        for lab, tok, wgt in bot:
            self.add_key(x, by0, x + unit * wgt, by0 + bot_h, {"label": lab, "act": ("tok", tok)},
                         NAV if tok != "speak" else "#14532D")
            x += unit * wgt + gap
        self.status_item = cv.create_text(m, H - m - st_h / 2 + 2, anchor="w", fill=DIM,
                                          font=self.font(self.en_family, st_h * 0.6), tags="ui", text="")
        self.draw_status()
        self.prog = cv.create_arc(0, 0, 1, 1, start=90, extent=0, style="arc", width=8,
                                  outline=RING, state="hidden", tags="dyn")
        self.cursor = cv.create_oval(-20, -20, -20, -20, outline="#FACC15", width=4, tags="dyn")
        self.warn_item = cv.create_text(W / 2, ty1 + 4, anchor="n", text="", fill="#F87171",
                                        font=self.font(self.en_family, 26, True), tags="dyn")

    def add_key(self, x0, y0, x1, y1, k, fill, hi=False):
        cv = self.cv
        k["rect"] = (x0, y0, x1, y1)
        k["fill"] = fill
        k["rid"] = cv.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#475569", width=2, tags="ui")
        w, h = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if k["act"][0] == "phrase":
            cv.create_text(cx, cy - h * 0.16, text=k["en"], fill=TXT,
                           font=self.font(self.en_family, min(h * 0.26, w * 0.2), True), tags="ui")
            cv.create_text(cx, cy + h * 0.22, text=k["hi"], fill="#FDE68A",
                           font=self.font(self.hi_family, min(h * 0.24, w * 0.2)), tags="ui")
        else:
            lab = k["label"]
            fam = self.hi_family if (hi or DEVANAGARI.search(lab)) else self.en_family
            px = min(h * 0.45, (w * 0.8) / (0.62 * max(len(lab), 1)))
            cv.create_text(cx, cy, text=lab, fill=TXT,
                           font=self.font(fam, px, len(lab) > 1 and k["act"][0] == "tok"), tags="ui")
        self.keys.append(k)

    def draw_status(self):
        if not hasattr(self, "status_item"):
            return
        err = ""
        if self.cal_err:
            err = f"   |   calib. error ~{self.cal_err['median_px'] / self.px_per_cm:.1f} cm ({self.cal_err['median_px']:.0f} px)"
        mode = f"Hybrid-{self.args.arch.upper()}" if self.use_hybrid else "Geom-Baseline"
        self.cv.itemconfig(
            self.status_item,
            text=f"[{mode}]  Dwell {self.dwell:.1f}s (- / =){err}   |   R recalibrate   P preview   ESC quit"
        )

    # ---------------------------------------------------------------- dynamic
    def key_at(self, x, y, pad=0):
        for k in self.keys:
            x0, y0, x1, y1 = k["rect"]
            if x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad:
                return k
        return None

    def handle_gaze(self, x, y, blink, now):
        """Dwell logic. Blinks / lost face pause the timer instead of resetting it."""
        dt = now - self._last_now if self._last_now is not None else 0.0
        self._last_now = now
        if x is None or blink:
            self.dwell_start += dt
            self.cool_until += dt
            return
        pad = 6 if self.cur_key is None else 0.12 * min(self.cur_key["rect"][2] - self.cur_key["rect"][0],
                                                        self.cur_key["rect"][3] - self.cur_key["rect"][1])
        sticky = self.cur_key is not None and self.key_at(x, y, pad) is self.cur_key
        key = self.cur_key if sticky else self.key_at(x, y, 4)
        if key is not self.cur_key:
            if self.cur_key is not None:
                self.cv.itemconfig(self.cur_key["rid"], fill=self.cur_key["fill"])
            self.cur_key = key
            self.dwell_start = now
            if key is not None:
                self.cv.itemconfig(key["rid"], fill=KEY_HOT)
        self.progress = 0.0
        if self.cur_key is None or self.cur_key["act"] != self.locked_act:
            self.locked_act = None          # gaze left the key: it can be selected again
        if self.cur_key is not None and now >= self.cool_until and self.cur_key["act"] != self.locked_act:
            self.progress = min(1.0, (now - self.dwell_start) / self.dwell)
            if self.progress >= 1.0:
                act = self.cur_key["act"]
                if act[0] == "tok" and act[1] not in ("back", "space"):
                    self.locked_act = act       # navigation keys: look away before re-triggering
                self.activate(self.cur_key)
                self.cool_until = now + 0.7
                self.dwell_start = now + 0.7
                self.progress = 0.0

    def draw_dynamic(self):
        if self.gaze:
            x, y = self.gaze
            self.cv.coords(self.cursor, x - 14, y - 14, x + 14, y + 14)
        if self.cur_key is not None and self.progress > 0:
            x0, y0, x1, y1 = self.cur_key["rect"]
            cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, min(x1 - x0, y1 - y0) * 0.36
            self.cv.coords(self.prog, cx - r, cy - r, cx + r, cy + r)
            self.cv.itemconfig(self.prog, extent=-360 * self.progress, state="normal")
            self.cv.tag_raise(self.prog)
        else:
            self.cv.itemconfig(self.prog, state="hidden")
        self.cv.itemconfig(self.warn_item, text=self.warn)
        self.cv.tag_raise(self.cursor)

    # ---------------------------------------------------------------- actions
    def set_text(self, s):
        self.text = s
        self.cv.itemconfig(self.text_item, text=self.text + "|")

    def activate(self, key):
        kind = key["act"][0]
        self.cv.itemconfig(key["rid"], fill=KEY_OK)
        self.root.after(260, lambda k=key: self.cv.itemconfig(k["rid"], fill=k["fill"]) if k in self.keys else None)
        if kind == "char":
            self.set_text(self.text + key["act"][1])
        elif kind == "phrase":
            en, hi = key["act"][1], key["act"][2]
            self.set_text(f"{en} / {hi}")
            self.speak([(en, None), (hi, HINDI_VOICE)])
        else:
            tok = key["act"][1]
            if tok in ("board", "en", "hi"):
                self.screen, self.page = tok, 0
                self.build_layout()
            elif tok == "style":
                self.style = "dense" if self.style == "large" else "large"
                self.page = 0
                self.build_layout()
            elif tok == "next":
                self.page += 1
                self.build_layout()
            elif tok == "space":
                self.set_text(self.text + " ")
            elif tok == "back":
                self.set_text(self.text[:-1])
            elif tok == "clear":
                self.set_text("")
            elif tok == "speak":
                runs = re.findall(r"[\u0900-\u097F][\u0900-\u097F\s,।?!]*|[^\u0900-\u097F]+", self.text)
                self.speak([(r.strip(), HINDI_VOICE if DEVANAGARI.search(r) else None)
                            for r in runs if r.strip()])

    def speak(self, parts):
        def run():
            for text, voice in parts:
                if sys.platform == "darwin":
                    cmd = ["say"] + (["-v", voice] if voice else []) + [text]
                    try:
                        if subprocess.run(cmd, capture_output=True).returncode != 0 and voice:
                            subprocess.run(["say", text], capture_output=True)
                    except FileNotFoundError:
                        pass
                else:
                    print("[speak]", text)
        threading.Thread(target=run, daemon=True).start()

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="GazeConnect hybrid gaze keyboard")
    ap.add_argument("--arch", default="l2cs", choices=["l2cs", "basic", "geom"],
                    help="gaze model: l2cs (hybrid L2CS+Geom+PSO), basic (hybrid BasicCNN+Geom+PSO), or geom (baseline)")
    ap.add_argument("--mouse", action="store_true", help="use the mouse instead of the webcam (UI test)")
    ap.add_argument("--load", action="store_true", help="reuse saved calibration")
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--dwell", type=float, default=1.3, help="dwell time in seconds")
    ap.add_argument("--windowed", action="store_true", help="do not use fullscreen")
    App(ap.parse_args()).run()


if __name__ == "__main__":
    main()