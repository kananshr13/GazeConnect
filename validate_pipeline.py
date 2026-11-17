#!/usr/bin/env python3
"""
GazeConnect Milestone-2 Nested Validation & Ablation Benchmark

Evaluates five screen-coordinate mapping variants using OUTER
Leave-One-Target-Out (LOTO) validation. Hyperparameters are selected using
only the outer training targets, preventing the outer test target from
influencing model selection.

Metrics:
  - Median and mean Euclidean screen-position error (pixels)
  - P80 Euclidean screen-position error (pixels)
  - MSE of 2D Euclidean position error (pixels squared)
  - RMSE of 2D Euclidean position error (pixels)
  - Median error in centimetres
  - Simulated dense-key hit rate
  - Separately measured feature-pipeline latency and throughput

IMPORTANT:
This is a screen-coordinate calibration evaluation, not an angular yaw/pitch
evaluation. It does not calculate angular MSE/RMSE.
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
    face_features, fit_best,
)
from hybrid_gaze import HybridFeatureExtractor, PSOHybridMapper, fit_best_pso


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "milestone2_benchmark_results_updated.csv")

# Match the original benchmark's simulated dense-key acceptance radius.
DENSE_KEY_RADIUS_FRACTION = 0.072


def benchmark_latency(cap, tracker, ext_basic, ext_l2cs, n_frames=100):
    """Measure feature-pipeline latency on valid face frames."""
    t_mp, t_basic, t_l2cs = [], [], []
    valid = 0
    attempts = 0
    max_attempts = n_frames * 10

    while valid < n_frames and attempts < max_attempts:
        attempts += 1
        ok, frame = cap.read()
        if not ok:
            continue

        t0 = time.perf_counter()
        landmarks = tracker.detect(frame)
        t1 = time.perf_counter()

        if landmarks is None:
            continue

        geom = face_features(landmarks)
        if geom is None:
            continue

        tb0 = time.perf_counter()
        ext_basic.extract(frame, landmarks, dict(geom))
        tb1 = time.perf_counter()

        tl0 = time.perf_counter()
        ext_l2cs.extract(frame, landmarks, dict(geom))
        tl1 = time.perf_counter()

        t_mp.append((t1 - t0) * 1000.0)
        t_basic.append((tb1 - tb0) * 1000.0)
        t_l2cs.append((tl1 - tl0) * 1000.0)
        valid += 1

    if valid < n_frames:
        raise RuntimeError(
            f"Could collect only {valid}/{n_frames} valid face frames "
            "for latency measurement. Check camera and lighting."
        )

    mp_ms = float(np.mean(t_mp))
    # Keep the same small mapper overhead allowance used by the earlier script.
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
    """
    Collect synchronized geometric, BasicCNN and L2CS features at the
    nine calibration targets. Y stores normalized screen coordinates.
    """
    root = tk.Tk()
    root.title("GazeConnect - Validation Data Capture")
    root.configure(bg="#0F172A")
    root.attributes("-fullscreen", True)

    canvas = tk.Canvas(root, bg="#0F172A", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    root.update()

    width, height = canvas.winfo_width(), canvas.winfo_height()
    mm = max(root.winfo_screenmmwidth(), 1)
    px_per_cm = root.winfo_screenwidth() / (mm / 10.0)

    blinker = BlinkDetector()
    F_geom, F_basic, F_l2cs, Y, groups = [], [], [], [], []
    started = [False]
    quit_flag = [False]

    root.bind("<space>", lambda _e: started.__setitem__(0, True))
    root.bind("<Escape>", lambda _e: quit_flag.__setitem__(0, True))

    while not started[0] and not quit_flag[0]:
        cap.read()
        canvas.delete("all")
        canvas.create_text(
            width / 2, height * 0.42,
            text="Milestone-2 Quantitative Benchmark Capture",
            fill="#F8FAFC", font=("Helvetica", 40, "bold")
        )
        canvas.create_text(
            width / 2, height * 0.52,
            text="Sit about 50 cm from the screen and follow the 9 targets.",
            fill="#94A3B8", font=("Helvetica", 22)
        )
        canvas.create_text(
            width / 2, height * 0.62,
            text="Press SPACE to begin (ESC to cancel)",
            fill="#34D399", font=("Helvetica", 26, "bold")
        )
        root.update()
        time.sleep(0.015)

    cal_i = 0
    target_start = time.perf_counter()
    point_count = 0

    while cal_i < len(CAL_PTS) and not quit_flag[0]:
        now = time.perf_counter()
        px, py = CAL_PTS[cal_i]
        ok, frame = cap.read()
        geom = None

        if ok:
            landmarks = tracker.detect(frame)
            if landmarks is not None:
                geom = face_features(landmarks)

            if geom is not None:
                blink = blinker.update(geom["ear"])
                elapsed = now - target_start

                if elapsed > SETTLE and not blink:
                    basic = ext_basic.extract(frame, landmarks, dict(geom))
                    l2cs = ext_l2cs.extract(frame, landmarks, dict(geom))

                    F_geom.append(geom["f"])
                    F_basic.append(basic["f"])
                    F_l2cs.append(l2cs["f"])

                    # The mapper expects normalized target coordinates.
                    Y.append([px, py])
                    groups.append(cal_i)
                    point_count += 1

        elapsed = now - target_start

        if elapsed > SETTLE + COLLECT:
            if point_count >= 12:
                cal_i += 1
            else:
                # Drop only the frames captured for the incomplete target.
                keep = len(groups) - point_count
                F_geom = F_geom[:keep]
                F_basic = F_basic[:keep]
                F_l2cs = F_l2cs[:keep]
                Y = Y[:keep]
                groups = groups[:keep]

            point_count = 0
            target_start = now

        canvas.delete("all")
        cx, cy = px * width, py * height
        radius = 34 - 22 * min(1.0, elapsed / SETTLE) if elapsed < SETTLE else 12
        canvas.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius,
            outline="#34D399", width=4
        )
        canvas.create_oval(
            cx - 6, cy - 6, cx + 6, cy + 6,
            fill="#F8FAFC", outline=""
        )
        canvas.create_text(
            width / 2, height * 0.5 + (60 if abs(py - 0.5) > 0.2 else 120),
            text=f"Benchmark Target {min(cal_i + 1, 9)} / 9",
            fill="#94A3B8", font=("Helvetica", 22)
        )
        root.update()

    root.destroy()

    if quit_flag[0] or len(groups) < 50:
        raise RuntimeError("Benchmark capture cancelled or insufficient frames.")

    return (
        np.asarray(F_geom, dtype=float),
        np.asarray(F_basic, dtype=float),
        np.asarray(F_l2cs, dtype=float),
        np.asarray(Y, dtype=float),
        np.asarray(groups, dtype=int),
        np.asarray([width, height], dtype=float),
        float(px_per_cm),
    )


def calculate_position_metrics(predicted, actual, screen_size):
    """
    Compute metrics from normalized [x,y] coordinates.

    Each sample's error is the Euclidean distance in screen pixels:
        e_i = sqrt((dx_i * screen_width)^2 +
                   (dy_i * screen_height)^2)

    MSE = mean(e_i ** 2), unit pixels^2.
    RMSE = sqrt(MSE), unit pixels.
    """
    delta_px = (np.asarray(predicted) - np.asarray(actual)) * screen_size
    errors_px = np.linalg.norm(delta_px, axis=1)
    squared_errors_px2 = np.sum(delta_px ** 2, axis=1)

    mse_px2 = float(np.mean(squared_errors_px2))
    rmse_px = float(np.sqrt(mse_px2))

    return {
        "median_px": float(np.median(errors_px)),
        "mean_px": float(np.mean(errors_px)),
        "p80_px": float(np.percentile(errors_px, 80)),
        "mse_px2": mse_px2,
        "rmse_px": rmse_px,
        "errors_px": errors_px,
    }


def eval_outer_loto(model_builder, features, targets, groups, screen_size):
    """
    Outer LOTO evaluation.

    For each target group:
      1. Exclude that target entirely from training/model selection.
      2. Build/optimize a model using only the remaining targets.
      3. Predict the held-out target.
    """
    all_predictions = []
    all_targets = []

    for held_out_group in np.unique(groups):
        train_mask = groups != held_out_group
        test_mask = groups == held_out_group

        if train_mask.sum() < 15 or test_mask.sum() < 1:
            continue

        model = model_builder(
            features[train_mask],
            targets[train_mask],
            groups[train_mask],
            screen_size,
        )
        predictions = model.predict(features[test_mask])

        all_predictions.append(predictions)
        all_targets.append(targets[test_mask])

    if not all_predictions:
        raise RuntimeError("No valid outer LOTO folds were evaluated.")

    predictions = np.concatenate(all_predictions, axis=0)
    actual = np.concatenate(all_targets, axis=0)

    result = calculate_position_metrics(predictions, actual, screen_size)

    dense_radius_px = DENSE_KEY_RADIUS_FRACTION * np.linalg.norm(screen_size)
    result["dense_hit_pct"] = float(
        np.mean(result["errors_px"] <= dense_radius_px) * 100.0
    )
    result.pop("errors_px")

    return result


def build_fixed_geometric(F_train, Y_train, G_train, screen_size):
    """M1: Pure geometric mapper with fixed lambda=1."""
    return GazeMapper(1.0).fit(F_train, Y_train)


def build_tuned_geometric(F_train, Y_train, G_train, screen_size):
    """
    M2: Select lambda using only the outer training targets.
    fit_best performs inner LOTO on the supplied training subset.
    """
    best_mapper, _, _ = fit_best(F_train, Y_train, G_train, screen_size)
    return GazeMapper(best_mapper.lam).fit(F_train, Y_train)


def build_appearance_only(F_train, Y_train, G_train, screen_size):
    """M3: L2CS appearance features only; geometric/head weights disabled."""
    params = {
        "lam": 3.0,
        "w_geom": 0.0,
        "w_cnn": 1.0,
        "w_head": 0.0,
    }
    return PSOHybridMapper(params).fit(F_train, Y_train)


def build_basic_hybrid(F_train, Y_train, G_train, screen_size):
    """M4: PSO tuning occurs only inside the outer training fold."""
    mapper, _, _ = fit_best_pso(
        F_train, Y_train, G_train, screen_size,
        n_particles=20, n_iters=25
    )
    return mapper


def build_l2cs_hybrid(F_train, Y_train, G_train, screen_size):
    """M5: Proposed Hybrid-L2CS; PSO sees only outer training targets."""
    mapper, _, _ = fit_best_pso(
        F_train, Y_train, G_train, screen_size,
        n_particles=24, n_iters=30
    )
    return mapper


def fit_final_deployment_mapper(features, targets, groups, screen_size):
    """
    Fit a final M5 mapper on all captured calibration data.
    This mapper is for deployment after evaluation; its training fit is NOT
    used to calculate the outer LOTO performance metrics.
    """
    mapper, med, p80 = fit_best_pso(
        features, targets, groups, screen_size,
        n_particles=24, n_iters=30
    )
    return mapper, med, p80


def main():
    print("=== GazeConnect Milestone-2 Nested Quantitative Validation ===")
    print("Metrics are screen-coordinate errors, not angular yaw/pitch errors.")

    tracker = FaceTracker()
    ext_basic = HybridFeatureExtractor(MODEL_DIR, arch="basic")
    ext_l2cs = HybridFeatureExtractor(MODEL_DIR, arch="l2cs")

    if ext_basic.cnn is None:
        raise RuntimeError(
            "BasicCNN checkpoint was not loaded. Check models/basic_cnn_best.pth."
        )
    if ext_l2cs.cnn is None:
        raise RuntimeError(
            "L2CS checkpoint was not loaded. Check models/l2cs_style_best.pth."
        )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        tracker.close()
        raise RuntimeError("Could not open webcam. Check camera permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        print("\n[1/3] Measuring feature-pipeline latency...")
        latency = benchmark_latency(
            cap, tracker, ext_basic, ext_l2cs, n_frames=100
        )
        print(f"MediaPipe: {latency['mediapipe_ms']:.2f} ms")
        print(
            f"Hybrid BasicCNN: {latency['basic_ms']:.2f} ms "
            f"({1000.0 / latency['basic_ms']:.1f} FPS)"
        )
        print(
            f"Hybrid L2CS: {latency['l2cs_ms']:.2f} ms "
            f"({1000.0 / latency['l2cs_ms']:.1f} FPS)"
        )

        print("\n[2/3] Starting synchronized 9-point calibration capture...")
        F_geom, F_basic, F_l2cs, Y, G, screen_size, px_per_cm = (
            collect_dual_stream_calibration(
                cap, tracker, ext_basic, ext_l2cs
            )
        )
    finally:
        cap.release()
        tracker.close()

    print(f"\nCaptured {len(G)} valid frames across {len(np.unique(G))} targets.")
    print("\n[3/3] Running nested outer LOTO evaluation...")
    print("Each held-out target is excluded from hyperparameter selection.")

    model_specs = [
        (
            "M1: Pure Geometric + Fixed Ridge",
            F_geom, build_fixed_geometric, latency["geom_ms"]
        ),
        (
            "M2: Pure Geometric + Tuned Ridge",
            F_geom, build_tuned_geometric, latency["geom_ms"]
        ),
        (
            "M3: Appearance-Only L2CS + Ridge",
            F_l2cs, build_appearance_only, latency["l2cs_ms"]
        ),
        (
            "M4: Hybrid BasicCNN + Geometry + PSO",
            F_basic, build_basic_hybrid, latency["basic_ms"]
        ),
        (
            "M5: Hybrid L2CS + Geometry + PSO [Proposed]",
            F_l2cs, build_l2cs_hybrid, latency["l2cs_ms"]
        ),
    ]

    results = []
    for name, features, builder, latency_ms in model_specs:
        print(f"\nEvaluating {name} ...")
        metrics = eval_outer_loto(
            builder, features, Y, G, screen_size
        )
        metrics["median_cm"] = metrics["median_px"] / px_per_cm
        metrics["latency_ms"] = float(latency_ms)
        metrics["fps"] = 1000.0 / latency_ms
        results.append((name, metrics))

    headers = [
        "Model",
        "Median_Error_px",
        "Mean_Error_px",
        "P80_Error_px",
        "MSE_2D_Position_px2",
        "RMSE_2D_Position_px",
        "Median_Error_cm",
        "Dense_Key_Hit_Rate_pct",
        "Latency_ms",
        "Throughput_FPS",
    ]

    print("\n" + "=" * 150)
    print(
        f"{'Model':<48} {'Median':>9} {'Mean':>9} {'P80':>9} "
        f"{'MSE(px2)':>12} {'RMSE(px)':>10} {'Hit%':>8} {'FPS':>8}"
    )
    print("-" * 150)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)

        for name, m in results:
            print(
                f"{name:<48} "
                f"{m['median_px']:9.2f} "
                f"{m['mean_px']:9.2f} "
                f"{m['p80_px']:9.2f} "
                f"{m['mse_px2']:12.2f} "
                f"{m['rmse_px']:10.2f} "
                f"{m['dense_hit_pct']:7.2f}% "
                f"{m['fps']:8.1f}"
            )
            writer.writerow([
                name,
                round(m["median_px"], 4),
                round(m["mean_px"], 4),
                round(m["p80_px"], 4),
                round(m["mse_px2"], 4),
                round(m["rmse_px"], 4),
                round(m["median_cm"], 4),
                round(m["dense_hit_pct"], 4),
                round(m["latency_ms"], 4),
                round(m["fps"], 4),
            ])

    print("=" * 150)
    print(f"\nSaved nested-validation results to:\n{OUT_CSV}")

    print("\nFitting final M5 deployment mapper on all captured calibration data...")
    final_mapper, train_median, train_p80 = fit_final_deployment_mapper(
        F_l2cs, Y, G, screen_size
    )
    print("Final deployment PSO parameters:")
    print(final_mapper.params)
    print(
        "Full-data calibration fit diagnostics (NOT held-out test metrics): "
        f"median={train_median:.2f}px, P80={train_p80:.2f}px"
    )
    print(
        "\nUse the outer nested-LOTO MSE/RMSE in your report. "
        "Do not report the final full-data fit diagnostics as test performance."
    )


if __name__ == "__main__":
    main()
