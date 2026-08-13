import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ultralytics import YOLO

plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})


def plot_confusion_matrix(cm_full, names, out_dir, top_n=20):
    nc = len(names)
    cm = cm_full[:nc, :nc]
    support = cm.sum(axis=0)
    top_idx = np.argsort(support)[::-1][:top_n]
    top_idx = np.sort(top_idx)
    cm_top = cm[np.ix_(top_idx, top_idx)]
    labels = [names[i] for i in top_idx]

    cm_norm = cm_top / (cm_top.sum(axis=0, keepdims=True) + 1e-9)

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle("Confusion Matrix - ATR Aircraft Detection (Top 20 classes)", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)

    for i, (mat, title, fmt, cmap) in enumerate([
        (cm_top, "(a) Raw Counts", "d", "Blues"),
        (cm_norm * 100, "(b) Normalized (%)", ".1f", "Blues"),
    ]):
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_title(title, fontsize=13)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        thresh = mat.max() / 2
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                val = mat[r, c]
                if val > 0:
                    if fmt == "d":
                        txt = f"{int(round(val))}"
                    else:
                        txt = f"{val:{fmt}}%"
                    ax.text(c, r, txt, ha="center", va="center", fontsize=6,
                             color="white" if val > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = out_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

    pd.DataFrame(cm, index=names, columns=names).to_csv(out_dir / "confusion_matrix_full_114.csv")
    print(f"wrote {out_dir / 'confusion_matrix_full_114.csv'}")


def write_metrics_table(box, names, out_dir):
    p, r, f1 = box.p, box.r, box.f1
    try:
        support = box.nt_per_class
    except AttributeError:
        support = None

    lines = []
    lines.append("EVALUATION METRICS (Object Detection)")
    lines.append(f"mAP50-95: {box.map:.4f}")
    lines.append(f"mAP50: {box.map50:.4f}")
    lines.append(f"Mean Precision: {box.mp:.4f}")
    lines.append(f"Mean Recall: {box.mr:.4f}")
    lines.append("")
    header = f"{'Class':<20}{'precision':>10}{'recall':>10}{'f1-score':>10}"
    if support is not None:
        header += f"{'support':>10}"
    lines.append(header)
    for i, name in enumerate(names):
        row = f"{name:<20}{p[i]:>10.2f}{r[i]:>10.2f}{f1[i]:>10.2f}"
        if support is not None:
            row += f"{int(support[i]):>10}"
        lines.append(row)
    lines.append("")
    lines.append(f"{'macro avg':<20}{p.mean():>10.2f}{r.mean():>10.2f}{f1.mean():>10.2f}")

    out_path = out_dir / "metrics_table.txt"
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


def plot_pr_curve(box, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        px, py = box.curves_results[0][0], box.curves_results[0][1]
        ax.plot(px, py.mean(axis=0) if py.ndim > 1 else py, linewidth=2, color="#1f77b4")
    except Exception:
        ax.text(0.5, 0.5, "PR curve data unavailable from this ultralytics version\nsee runs/detect/val/PR_curve.png",
                ha="center", va="center")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (mAP50={box.map50:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    out_path = out_dir / "pr_curve.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path} (Ultralytics own PR_curve.png in runs/detect/val/ is the more reliable version)")


def plot_training_curves(results_csv, out_dir):
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Curves", fontsize=15, fontweight="bold")

    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    for c in loss_cols:
        axes[0, 0].plot(df["epoch"], df[c], label=c)
    axes[0, 0].set_title("Losses")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    map_cols = [c for c in df.columns if "mAP" in c]
    for c in map_cols:
        axes[0, 1].plot(df["epoch"], df[c], label=c)
    axes[0, 1].set_title("mAP")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    pr_cols = [c for c in df.columns if "precision" in c.lower() or "recall" in c.lower()]
    for c in pr_cols:
        axes[1, 0].plot(df["epoch"], df[c], label=c)
    axes[1, 0].set_title("Precision / Recall")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    lr_cols = [c for c in df.columns if "lr" in c.lower()]
    for c in lr_cols:
        axes[1, 1].plot(df["epoch"], df[c], label=c)
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    out_path = out_dir / "training_curves.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def actual_vs_predicted_grid(yolo, val_images_dir, out_dir, n=6, imgsz=640):
    import cv2
    imgs = sorted(list(Path(val_images_dir).glob("*.jpg")))[:n]
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle("Actual (Ground Truth) vs Predicted", fontsize=15, fontweight="bold")

    for i, img_path in enumerate(imgs):
        label_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (imgsz, imgsz))
        gt_img = img.copy()
        if label_path.exists():
            h, w = imgsz, imgsz
            for line in label_path.read_text().splitlines():
                cls, cx, cy, bw, bh = map(float, line.split())
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        result = yolo.predict(str(img_path), imgsz=imgsz, verbose=False)[0]
        pred_img = cv2.resize(result.plot(), (imgsz, imgsz))

        axes[0, i].imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
        axes[0, i].axis("off")
        axes[0, i].set_title("Actual", fontsize=9)
        axes[1, i].imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
        axes[1, i].axis("off")
        axes[1, i].set_title("Predicted", fontsize=9)

    out_path = out_dir / "actual_vs_predicted.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="outputs/models/best.pt")
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--val-images", default="data/yolo/images/validation")
    ap.add_argument("--train-results", default=None, help="path to runs/detect/train-N/results.csv")
    ap.add_argument("--out", default="outputs/report")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(args.weights)
    metrics = yolo.val(data=args.data, device=args.device, plots=True)
    names = [metrics.names[i] for i in range(len(metrics.names))]

    plot_confusion_matrix(metrics.confusion_matrix.matrix, names, out_dir)
    write_metrics_table(metrics.box, names, out_dir)
    plot_pr_curve(metrics.box, out_dir)
    actual_vs_predicted_grid(yolo, args.val_images, out_dir)

    if args.train_results and Path(args.train_results).exists():
        plot_training_curves(args.train_results, out_dir)
    else:
        print("Pass --train-results runs/detect/train-N/results.csv to also generate training_curves.png")


if __name__ == "__main__":
    main()
