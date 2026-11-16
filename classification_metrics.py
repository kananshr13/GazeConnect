#!/usr/bin/env python3
"""
Computes full Regression (MAE, RMSE, Median px/cm, Degrees, R^2) and
Discrete Key-Selection Classification Metrics (Accuracy, Macro Precision, Recall, F1-Score)
for GazeConnect's Hybrid-L2CS + PSO model using pure NumPy (no scikit-learn needed).
"""
import os
import math
import time
import tkinter as tk
import cv2
import numpy as np

from gaze_keyboard import CAL_PTS, COLLECT, MODEL_DIR, SETTLE, BlinkDetector, FaceTracker, face_features
from hybrid_gaze import HybridFeatureExtractor, fit_best_pso

HERE = os.path.dirname(os.path.abspath(__file__))


def assign_grid_class(pts_norm, cols, rows):
    """Maps normalized (x, y) coordinates in [0, 1]^2 to discrete key IDs (0 .. cols*rows - 1)."""
    cx = np.clip((pts_norm[:, 0] * cols).astype(int), 0, cols - 1)
    cy = np.clip((pts_norm[:, 1] * rows).astype(int), 0, rows - 1)
    return cy * cols + cx


def compute_multiclass_metrics(y_true, y_pred):
    """Computes Accuracy, Macro Precision, Macro Recall, and Macro F1-Score using pure NumPy."""
    classes = np.unique(y_true)
    acc = float(np.mean(y_true == y_pred))
    precs, recs, f1s = [], [], []
    for c in classes:
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0
        precs.append(p)
        recs.append(r)
        f1s.append(f1)
    return acc, float(np.mean(precs)), float(np.mean(recs)), float(np.mean(f1s))


def compute_r2(y_true, y_pred):
    """Uniform-average multi-output R^2 score."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    return float(np.mean(1.0 - (ss_res / np.maximum(ss_tot, 1e-12))))


def main():
    tracker = FaceTracker()
    ext = HybridFeatureExtractor(MODEL_DIR, arch="l2cs")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    root = tk.Tk()
    root.configure(bg="#0F172A")
    root.attributes("-fullscreen", True)
    cv = tk.Canvas(root, bg="#0F172A", highlightthickness=0)
    cv.pack(fill="both", expand=True)
    root.update()
    W, H = cv.winfo_width(), cv.winfo_height()
    px_per_cm = root.winfo_screenwidth() / (max(root.winfo_screenmmwidth(), 1) / 10.0)

    blinker = BlinkDetector()
    F, Y, G = [], [], []
    started = [False]
    root.bind("<space>", lambda e: started.__setitem__(0, True))

    while not started[0]:
        cap.read()
        cv.delete("all")
        cv.create_text(W / 2, H * 0.45, text="Full Metrics Evaluation (Accuracy, Precision, Recall, F1, R²)",
                       fill="#F8FAFC", font=("Helvetica", 32, "bold"))
        cv.create_text(W / 2, H * 0.55, text="Press SPACE to capture 9-target evaluation set",
                       fill="#34D399", font=("Helvetica", 24, "bold"))
        root.update()
        time.sleep(0.015)

    cal_i, t0, pt_count = 0, time.perf_counter(), 0
    while cal_i < len(CAL_PTS):
        now = time.perf_counter()
        px, py = CAL_PTS[cal_i]
        ok, frame = cap.read()
        if ok:
            P = tracker.detect(frame)
            if P is not None:
                ft = face_features(P)
                if ft is not None and not blinker.update(ft["ear"]) and (now - t0) > SETTLE:
                    ft = ext.extract(frame, P, ft)
                    F.append(ft["f"])
                    Y.append([px, py])
                    G.append(cal_i)
                    pt_count += 1
        t = now - t0
        if t > SETTLE + COLLECT:
            if pt_count >= 12:
                cal_i += 1
            else:
                keep = len(G) - pt_count
                F, Y, G = F[:keep], Y[:keep], G[:keep]
            pt_count, t0 = 0, now

        cv.delete("all")
        cx, cy = px * W, py * H
        r = 34 - 22 * min(1.0, t / SETTLE) if t < SETTLE else 12
        cv.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#34D399", width=4)
        cv.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill="#F8FAFC", outline="")
        cv.create_text(W / 2, H * 0.65, text=f"Target {min(cal_i + 1, 9)} / 9",
                       fill="#94A3B8", font=("Helvetica", 22))
        root.update()

    root.destroy()
    cap.release()
    tracker.close()

    F, Y, G = np.array(F), np.array(Y), np.array(G)
    size = np.array([W, H], dtype=float)
    best_mapper, _, _ = fit_best_pso(F, Y, G, size)

    # Leave-One-Target-Out predictions
    Y_true_all, Y_pred_all = [], []
    for g in np.unique(G):
        tr, te = G != g, G == g
        m = best_mapper.__class__(best_mapper.params).fit(F[tr], Y[tr])
        Y_true_all.append(Y[te])
        Y_pred_all.append(np.clip(m.predict(F[te]), 0.0, 1.0))

    Y_true = np.vstack(Y_true_all)
    Y_pred = np.vstack(Y_pred_all)

    # 1. Regression Metrics
    true_px = Y_true * size
    pred_px = Y_pred * size
    radial_err_px = np.linalg.norm(pred_px - true_px, axis=1)
    med_px = float(np.median(radial_err_px))
    mae_px = float(np.mean(radial_err_px))
    rmse_px = float(np.sqrt(np.mean((pred_px - true_px) ** 2)))
    med_cm = med_px / px_per_cm
    deg_err = math.degrees(math.atan2(med_cm, 50.0))  # at 50 cm viewing distance
    r2 = compute_r2(true_px, pred_px)

    print("\n================ CONTINUOUS GAZE REGRESSION METRICS ================")
    print(f"Median Euclidean Error : {med_px:.2f} px ({med_cm:.2f} cm)")
    print(f"Mean Radial Error (MAE): {mae_px:.2f} px ({mae_px / px_per_cm:.2f} cm)")
    print(f"Root Mean Sq Error RMSE: {rmse_px:.2f} px")
    print(f"Angular Gaze Error     : {deg_err:.2f}° (at 50 cm viewing distance)")
    print(f"Coefficient of Det R²  : {r2:.4f}")

    # 2. Classification Metrics across Keyboard Layouts
    print("\n================ DISCRETE KEY SELECTION METRICS ====================")
    for layout_name, cols, rows in [
        ("Needs Communication Board (4x2 = 8 large keys)", 4, 2),
        ("Large Alphabet / Varnamala Grid (5x3 = 15 keys)", 5, 3),
        ("Dense QWERTY Keyboard (10x3 = 30 keys)", 10, 3),
    ]:
        y_cls_true = assign_grid_class(Y_true, cols, rows)
        y_cls_pred = assign_grid_class(Y_pred, cols, rows)
        acc, prec, rec, f1 = compute_multiclass_metrics(y_cls_true, y_cls_pred)
        print(f"\n[{layout_name}]")
        print(f"  Accuracy        : {acc * 100:.2f}%")
        print(f"  Macro Precision : {prec * 100:.2f}%")
        print(f"  Macro Recall    : {rec * 100:.2f}%")
        print(f"  Macro F1-Score  : {f1 * 100:.2f}%")


if __name__ == "__main__":
    main()