import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", type=str, default=None)

    args = ap.parse_args()

    if args.resume:
        print(f"Resuming training from: {args.resume}")

        model = YOLO(args.resume)

        model.train(
            resume=args.resume,
            device=args.device,
            workers=args.workers
        )

    else:
        model = YOLO(args.model)

        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=20,
            project="runs/detect"
        )

        metrics = model.val()
        print("mAP50-95:", metrics.box.map)
        print("mAP50:", metrics.box.map50)


if __name__ == "__main__":
    main()