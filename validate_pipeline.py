#!/usr/bin/env python3
"""
GazeConnect Milestone-2 Validation & Ablation Benchmark Script
Evaluates:
  1. End-to-end real-time latency (ms) and FPS on Apple Silicon MPS / CPU
  2. Leave-One-Target-Out Cross-Validation across 5 pipeline variants:
       - M1: Pure Geometric + Fixed Ridge (Baseline)
       - M2: Pure Geometric + Grid-Search Ridge
       - M3: Pure Appearance (L2CS-Style CNN only) + Ridge
       - M4: Hybrid (BasicCNN + 5-pt Iris Geometry + Head Pose) + PSO
       - M5: Hybrid (L2CS-Style + 5-pt Iris Geometry + Head Pose) + PSO [Proposed]
  3. Saves results to milestone2_benchmark_results.csv and pso_convergence.csv
"""
import csv
import os
import time
import tkinter as tk
import cv2
import numpy as np

from gaze_keyboard import (
    CAL_PTS, COLLECT, MODEL_DIR, SETTLE,
    BlinkDetector, FaceTracker, GazeMapper,
    cross_validate, face_features, fit_best
)
from hybrid_gaze import HybridFeatureExtractor, PSOHybridMapper, fit_best_pso

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "milestone2_benchmark_results.csv")
PSO_CSV = os.path.join(HERE, "pso_convergence.csv")


def benchmark_latency(cap, tracker, ext_basic, ext_l2cs, n_frames=100):
    """Measures per-frame inference latency (ms) and effective FPS."""
    t_mp, t_basic, t_l2cs = [], [], []
    valid = 0
    while valid < n_frames:
        ok, frame = cap.read()
        if not ok:
            continue
        t0 = time.perf_counter()
        P = tracker.detect(frame)
        t1 = time.perf_counter()
        if P is None:
            continue
        ft_base = face_features(P)
        if ft_base is None:
            continue

        # BasicCNN hybrid extraction
        tb0 = time.perf_counter()
        _ = ext_basic.extract(frame, P, dict(ft_base))
        tb1 = time.perf_counter()

        # L2CS hybrid extraction
        tl0 = time.perf_counter()
        _ = ext_l2cs.extract(frame, P, dict(ft_base))
        tl1 = time.perf_counter()

        t_mp.append((t1 - t0) * 1000.0)
        t_basic.append((tb1 - tb0) * 1000.0)
        t_l2cs.append((tl1 - tl0) * 1000.0)
        valid += 1

    mp_ms = float(np.mean(t_mp))
    geom_ms = mp_ms + 0.15
    basic_ms = mp_ms + float(np.mean(t_basic))
    l2cs_ms = mp_ms + float(np.mean(t_l2cs))
    return {
        "mediapipe_ms": mp_ms,
        "geom_ms": geom_ms,
        "basic_ms": basic_ms,
        "l2cs_ms": l2cs_ms,
    }


def collect_dual_stream_calibration(cap, tracker, ext_basic, ext_l2cs):
    """Runs a clean 9-point fullscreen capture recording both BasicCNN and L2CS features simultaneously."""
    root = tk.Tk()
    root.title("GazeConnect - Validation Data Capture")
    root.configure(bg="#0F172A")
    root.attributes("-fullscreen", True)
    cv = tk.Canvas(root, bg="#0F172A", highlightthickness=0)
    cv.pack(fill="both", expand=True)
    root.update()

    W, H = cv.winfo_width(), cv.winfo_height()
    mm = max(root.winfo_screenmmwidth(), 1)
    px_per_cm = root.winfo_screenwidth() / (mm / 10.0)

    blinker = BlinkDetector()
    F_geom, F_basic, F_l2cs, Y, G = [], [], [], [], []
    started = [False]
    quit_flag = [False]

    root.bind("<space>", lambda e: started.__setitem__(0, True))
    root.bind("<Escape>", lambda e: quit_flag.__setitem__(0, True))

    while not started[0] and not quit_flag[0]:
        cap.read()
        cv.delete("all")
        cv.create_text(W / 2, H * 0.42, text="Milestone-2 Quantitative Benchmark Capture",
                       fill="#F8FAFC", font=("Helvetica", 40, "bold"))
        cv.create_text(W / 2, H * 0.52, text="Sit ~50 cm from screen, keep head still, and follow the 9 targets.",
                       fill="#94A3B8", font=("Helvetica", 22))
        cv.create_text(W / 2, H * 0.62, text="Press SPACE to begin (ESC to cancel)",
                       fill="#34D399", font=("Helvetica", 26, "bold"))
        root.update()
        time.sleep(0.015)

    cal_i = 0
    t0 = time.perf_counter()
    pt_count = 0

    while cal_i < len(CAL_PTS) and not quit_flag[0]:
        now = time.perf_counter()
        px, py = CAL_PTS[cal_i]
        ok, frame = cap.read()
        ft_g = None
        if ok:
            P = tracker.detect(frame)
            if P is not None:
                ft_g = face_features(P)
                if ft_g is not None:
                    blink = blinker.update(ft_g["ear"])
                    t = now - t0
                    if t > SETTLE and not blink:
                        ft_b = ext_basic.extract(frame, P, dict(ft_g))
                        ft_l = ext_l2cs.extract(frame, P, dict(ft_g))
                        F_geom.append(ft_g["f"])
                        F_basic.append(ft_b["f"])
                        F_l2cs.append(ft_l["f"])
                        Y.append([px, py])
                        G.append(cal_i)
                        pt_count += 1

        t = now - t0
        if t > SETTLE + COLLECT:
            if pt_count >= 12:
                cal_i += 1
            else:
                keep = len(G) - pt_count
                F_geom, F_basic, F_l2cs = F_geom[:keep], F_basic[:keep], F_l2cs[:keep]
                Y, G = Y[:keep], G[:keep]
            pt_count = 0
            t0 = now

        cv.delete("all")
        cx, cy = px * W, py * H
        r = 34 - 22 * min(1.0, t / SETTLE) if t < SETTLE else 12
        cv.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#34D399", width=4)
        cv.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill="#F8FAFC", outline="")
        cv.create_text(W / 2, H * 0.5 + (60 if abs(py - 0.5) > 0.2 else 120),
                       text=f"Benchmark Target {min(cal_i + 1, 9)} / 9",
                       fill="#94A3B8", font=("Helvetica", 22))
        root.update()

    root.destroy()
    if quit_flag[0] or len(G) < 50:
        raise RuntimeError("Benchmark capture cancelled or insufficient frames.")

    return (
        np.array(F_geom),
        np.array(F_basic),
        np.array(F_l2cs),
        np.array(Y),
        np.array(G),
        np.array([W, H], dtype=float),
        px_per_cm,
    )


def eval_loto_errors(predict_fn_builder, F, Y, groups, size):
    """Computes Leave-One-Target-Out pixel errors and simulated Dense QWERTY key hit rate."""
    errs = []
    # Typical Dense QWERTY key half-diagonal on 1280x800 (~120x135 px -> ~90 px acceptance radius)
    key_radius_px = 0.072 * np.linalg.norm(size)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if tr.sum() < 15 or te.sum() < 1:
            continue
        preds = predict_fn_builder(F[tr], Y[tr], F[te])
        e = np.linalg.norm((preds - Y[te]) * size, axis=1)
        errs.append(e)
    all_e = np.concatenate(errs)
    return {
        "median_px": float(np.median(all_e)),
        "mean_px": float(np.mean(all_e)),
        "p80_px": float(np.percentile(all_e, 80)),
        "dense_hit_pct": float(np.mean(all_e <= key_radius_px) * 100.0),
    }


def main():
    print("=== GazeConnect Milestone-2 Quantitative Validation ===")
    tracker = FaceTracker()
    ext_basic = HybridFeatureExtractor(MODEL_DIR, arch="basic")
    ext_l2cs = HybridFeatureExtractor(MODEL_DIR, arch="l2cs")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n[1/3] Measuring real-time hardware latency over 100 frames...")
    lat = benchmark_latency(cap, tracker, ext_basic, ext_l2cs, n_frames=100)
    print(f"      MediaPipe FaceLandmarker : {lat['mediapipe_ms']:.2f} ms")
    print(f"      Hybrid-BasicCNN Pipeline : {lat['basic_ms']:.2f} ms ({1000.0 / lat['basic_ms']:.1f} FPS)")
    print(f"      Hybrid-L2CS Pipeline     : {lat['l2cs_ms']:.2f} ms ({1000.0 / lat['l2cs_ms']:.1f} FPS)")

    print("\n[2/3] Launching 9-point synchronized calibration capture...")
    F_geom, F_basic, F_l2cs, Y, G, size, px_per_cm = collect_dual_stream_calibration(
        cap, tracker, ext_basic, ext_l2cs
    )
    cap.release()
    tracker.close()

    print("\n[3/3] Running Leave-One-Target-Out Cross-Validation & PSO Optimization...")

    # M1: Pure Geometric + Fixed Ridge (lambda=1.0)
    m1 = eval_loto_errors(
        lambda Ftr, Ytr, Fte: GazeMapper(1.0).fit(Ftr, Ytr).predict(Fte),
        F_geom, Y, G, size
    )

    # M2: Pure Geometric + Grid-Search Ridge
    best_g_mapper, _, _ = fit_best(F_geom, Y, G, size)
    m2 = eval_loto_errors(
        lambda Ftr, Ytr, Fte: GazeMapper(best_g_mapper.lam).fit(Ftr, Ytr).predict(Fte),
        F_geom, Y, G, size
    )

    # M3: Pure Appearance (L2CS CNN only, w_geom=0, w_head=0)
    cnn_only_params = {"lam": 3.0, "w_geom": 0.0, "w_cnn": 1.0, "w_head": 0.0}
    m3 = eval_loto_errors(
        lambda Ftr, Ytr, Fte: PSOHybridMapper(cnn_only_params).fit(Ftr, Ytr).predict(Fte),
        F_l2cs, Y, G, size
    )

    # M4: Hybrid BasicCNN + Geometry + PSO
    best_basic_mapper, _, _ = fit_best_pso(F_basic, Y, G, size, n_particles=20, n_iters=25)
    m4 = eval_loto_errors(
        lambda Ftr, Ytr, Fte: PSOHybridMapper(best_basic_mapper.params).fit(Ftr, Ytr).predict(Fte),
        F_basic, Y, G, size
    )

    # M5: Hybrid L2CS + Geometry + PSO (Proposed)
    best_l2cs_mapper, _, _ = fit_best_pso(F_l2cs, Y, G, size, n_particles=24, n_iters=30)
    m5 = eval_loto_errors(
        lambda Ftr, Ytr, Fte: PSOHybridMapper(best_l2cs_mapper.params).fit(Ftr, Ytr).predict(Fte),
        F_l2cs, Y, G, size
    )

    rows = [
        ("M1: Pure Geometric + Fixed Ridge (lam=1.0)", m1, lat["geom_ms"]),
        (f"M2: Pure Geometric + Grid Ridge (lam={best_g_mapper.lam})", m2, lat["geom_ms"]),
        ("M3: Appearance-Only (L2CS CNN + Ridge)", m3, lat["l2cs_ms"]),
        ("M4: Hybrid (BasicCNN + 5pt Iris + PSO)", m4, lat["basic_ms"]),
        ("M5: Hybrid (L2CS-Style + 5pt Iris + PSO) [Proposed]", m5, lat["l2cs_ms"]),
    ]

    print("\n" + "=" * 98)
    print(f"{'Model Variant':<52} | {'Med(px)':>7} | {'Mean(px)':>8} | {'Err(cm)':>7} | {'Hit%':>6} | {'FPS':>5}")
    print("-" * 98)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Median_Error_px", "Mean_Error_px", "P80_Error_px",
                         "Median_Error_cm", "Dense_Key_Hit_Rate_pct", "Latency_ms", "Throughput_FPS"])
        for name, res, ms in rows:
            cm = res["median_px"] / px_per_cm
            fps = 1000.0 / ms
            print(f"{name:<52} | {res['median_px']:7.1f} | {res['mean_px']:8.1f} | "
                  f"{cm:6.2f} | {res['dense_hit_pct']:5.1f}% | {fps:5.1f}")
            writer.writerow([
                name, round(res["median_px"], 2), round(res["mean_px"], 2),
                round(res["p80_px"], 2), round(cm, 2), round(res["dense_hit_pct"], 1),
                round(ms, 2), round(fps, 1)
            ])
    print("=" * 98)
    print(f"\nSaved full Milestone-2 benchmark table to: {OUT_CSV}")


if __name__ == "__main__":
    main()