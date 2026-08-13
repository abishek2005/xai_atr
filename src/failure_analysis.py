import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


def find_worst_classes(box, names, top_n, min_support=5):
    p, r, f1 = box.p, box.r, box.f1
    try:
        support = box.nt_per_class
    except AttributeError:
        support = np.full(len(names), min_support + 1)

    rows = []
    for i, name in enumerate(names):
        rows.append({"class": name, "precision": p[i], "recall": r[i], "f1": f1[i],
                     "support": int(support[i]) if support is not None else None})
    df = pd.DataFrame(rows)
    df = df[df["support"] >= min_support] if support is not None else df
    df = df.sort_values("f1").head(top_n)
    return df


def confusion_breakdown(cm_full, names, cls_name, top_k=5):
    idx = names.index(cls_name)
    col = cm_full[:len(names), idx]
    order = np.argsort(col)[::-1]
    order = [o for o in order if o != idx and col[o] > 0][:top_k]
    return [(names[o], int(col[o])) for o in order]


def find_sample_images(val_images_dir, labels_dir, class_idx, n=3):
    found = []
    for lbl in Path(labels_dir).glob("*.txt"):
        lines = lbl.read_text().splitlines()
        if any(line.split()[0] == str(class_idx) for line in lines if line.strip()):
            img_path = Path(val_images_dir) / (lbl.stem + ".jpg")
            if img_path.exists():
                found.append(img_path)
        if len(found) >= n:
            break
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="outputs/models/best.pt")
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--val-images", default="data/yolo/images/validation")
    ap.add_argument("--out", default="outputs/report")
    ap.add_argument("--device", default="0")
    ap.add_argument("--top-n", type=int, default=6)
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = Path(str(args.val_images).replace("images", "labels"))

    yolo = YOLO(args.weights)
    metrics = yolo.val(data=args.data, device=args.device, plots=False)
    names = [metrics.names[i] for i in range(len(metrics.names))]
    cm_full = metrics.confusion_matrix.matrix

    worst = find_worst_classes(metrics.box, names, args.top_n, args.min_support)
    worst.to_csv(out_dir / "worst_classes.csv", index=False)
    print(f"Worst {args.top_n} classes (min support {args.min_support}):")
    print(worst.to_string(index=False))
    print(f"wrote {out_dir / 'worst_classes.csv'}")

    n_samples = 3
    fig, axes = plt.subplots(len(worst), n_samples + 1,
                              figsize=(4 * (n_samples + 1), 4 * len(worst)))
    if len(worst) == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Failure Analysis: Worst-Performing Classes", fontsize=16, fontweight="bold")

    for row_i, (_, rec) in enumerate(worst.iterrows()):
        cls_name = rec["class"]
        cls_idx = names.index(cls_name)

        samples = find_sample_images(args.val_images, labels_dir, cls_idx, n=n_samples)
        for col_i in range(n_samples):
            ax = axes[row_i, col_i]
            if col_i < len(samples):
                result = yolo.predict(str(samples[col_i]), imgsz=args.imgsz, device=args.device, verbose=False)[0]
                img = cv2.resize(result.plot(), (args.imgsz, args.imgsz))
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            ax.set_title(f"{cls_name} sample {col_i+1}" if col_i < len(samples) else "no sample found",
                         fontsize=8)

        confused_with = confusion_breakdown(cm_full, names, cls_name)
        ax_bar = axes[row_i, n_samples]
        if confused_with:
            labels, counts = zip(*confused_with)
            ax_bar.barh(labels, counts, color="#d62728")
            ax_bar.invert_yaxis()
            ax_bar.set_title(f"{cls_name}: confused with", fontsize=8)
        else:
            ax_bar.text(0.5, 0.5, "no major confusions\n(mostly missed, not misclassified)",
                        ha="center", va="center", fontsize=8)
            ax_bar.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out_dir / "failure_analysis.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
