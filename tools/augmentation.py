from multiprocessing import Pool, cpu_count
from functools import partial
import os
import cv2
import argparse
import albumentations as A
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11 image augmentation with Albumentations")
    parser.add_argument("--input_images", required=True, help="Path to input images folder")
    parser.add_argument("--input_labels", required=True, help="Path to input YOLO labels folder")
    parser.add_argument("--output_images", required=True, help="Path to output images folder")
    parser.add_argument("--output_labels", required=True, help="Path to output labels folder")
    parser.add_argument('--resize_max_side', type=int, default=0, help='Resize so that the largest side equals this size (with proportional scaling)')
    parser.add_argument("--count", type=int, default=5, help="Number of augmented copies per image")
    parser.add_argument('--isonoise', type=float, default=0.0)
    parser.add_argument('--motionblur', type=int, default=0)
    parser.add_argument('--brightness', type=float, default=0.0, help='Brightness jitter')
    parser.add_argument('--contrast', type=float, default=0.0, help='Contrast jitter')
    parser.add_argument('--saturation', type=float, default=0.0, help='Saturation jitter')
    parser.add_argument('--hue', type=float, default=0.0, help='Hue jitter')
    parser.add_argument('--rotate', type=int, default=0)
    parser.add_argument('--optical', type=float, default=0.0)
    parser.add_argument('--shift', type=float, default=0.0, help='Max shift (0–1) for ShiftScaleRotate')
    parser.add_argument('--scale', type=float, default=0.0, help='Max scale (0–1) for ShiftScaleRotate')
    parser.add_argument('--hflip', action='store_true', help='Enable horizontal flip')
    parser.add_argument('--vflip', action='store_true', help='Enable vertical flip')
    return parser.parse_args()

def get_transform(args):
    transforms = []

    if args.isonoise > 0:
        transforms.append(A.ISONoise(intensity=(0.0, args.isonoise), p=1.0))

    if args.motionblur > 0:
        transforms.append(A.MotionBlur(blur_limit=(3, int(args.motionblur)), p=1.0))

    if any([args.brightness, args.contrast, args.saturation, args.hue]):
        transforms.append(A.ColorJitter(
            brightness=args.brightness,
            contrast=args.contrast,
            saturation=args.saturation,
            hue=args.hue,
            p=1.0
        ))

    if args.optical > 0:
        transforms.append(A.OpticalDistortion(distort_limit=args.optical, p=1.0))

    if any([args.shift, args.scale, args.rotate]):
        transforms.append(A.Affine(
            translate_percent={"x": args.shift, "y": args.shift},
            scale=(1.0 - args.scale, 1.0 + args.scale),
            rotate=(-args.rotate, args.rotate),
            p=1.0
        ))

    if args.hflip:
        transforms.append(A.HorizontalFlip(p=0.5))

    if args.vflip:
        transforms.append(A.VerticalFlip(p=0.5))
        
    if args.resize_max_side > 0:
        transforms.append(A.LongestMaxSize(max_size=args.resize_max_side))

    return A.Compose(
        [t for t in transforms if t is not None],
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
    )


def load_labels(label_path):
    bboxes = []
    class_labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x, y, w, h = parts
                bboxes.append([float(x), float(y), float(w), float(h)])
                class_labels.append(int(cls))
    return bboxes, class_labels

def save_labels(label_path, bboxes, class_labels):
    with open(label_path, "w") as f:
        for bbox, cls in zip(bboxes, class_labels):
            x, y, w, h = bbox
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def process_single_image(filename, args, transform):
    name, _ = os.path.splitext(filename)
    image_path = os.path.join(args.input_images, filename)
    label_path = os.path.join(args.input_labels, f"{name}.txt")

    if not os.path.exists(label_path):
        print(f"[WARN] Label file not found for image: {filename}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    bboxes, class_labels = load_labels(label_path)

    for i in range(args.count):
        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        except Exception as e:
            print(f"[ERROR] Augmentation failed for {filename}: {e}")
            continue

        new_image = transformed['image']
        new_bboxes = transformed['bboxes']
        new_labels = transformed['class_labels']

        if not new_bboxes:
            print(f"[INFO] Skipped augmentation (empty bbox) for {filename} (copy {i})")
            continue

        new_image_name = f"{name}_aug{i}.jpg"
        new_label_name = f"{name}_aug{i}.txt"

        cv2.imwrite(os.path.join(args.output_images, new_image_name), new_image)
        save_labels(os.path.join(args.output_labels, new_label_name), new_bboxes, new_labels)

        print(f"[OK] Saved: {new_image_name}")

def main():
    args = parse_args()
    transform = get_transform(args)

    os.makedirs(args.output_images, exist_ok=True)
    os.makedirs(args.output_labels, exist_ok=True)

    filenames = [f for f in os.listdir(args.input_images) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    print(f"🔁 Starting augmentation on {len(filenames)} images using {cpu_count()-1} threads...")

    with Pool(processes=cpu_count()-1) as pool:
        fn = partial(process_single_image, args=args, transform=transform)
        pool.map(fn, filenames)

    print("✅ Augmentation completed.")

if __name__ == "__main__":
    main()
