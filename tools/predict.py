import cv2
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Process video with YOLO")
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLO weights (e.g. yolo11n.pt)')
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, required=True, help='Path to save output video')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[640, 640], help='Inference image size (height width)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--codec', type=str, default='mp4v', help='FourCC video codec (e.g. mp4v, XVID, avc1)')
    args = parser.parse_args()

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print(f"[ERROR] Could not open input video: {args.input}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"[INFO] Processing video: {args.input}")
    print(f"[INFO] Saving output to: {args.output}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=args.imgsz, conf=args.conf)
        annotated = results[0].plot()
        out.write(annotated)

    cap.release()
    out.release()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
