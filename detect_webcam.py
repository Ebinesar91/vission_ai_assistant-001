#!/usr/bin/env python3
"""
YOLOv9 Webcam Detection Script
A lightweight script for real-time object detection using webcam feed.
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import cv2

# Add YOLOv9 directory to Python path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Current directory
YOLO_ROOT = ROOT / 'yolov9-main'  # YOLOv9 directory
if str(YOLO_ROOT) not in sys.path:
    sys.path.append(str(YOLO_ROOT))

try:
    from models.common import DetectMultiBackend
    from utils.dataloaders import LoadStreams
    from utils.general import (LOGGER, Profile, check_img_size, check_imshow, colorstr, cv2,
                               non_max_suppression, scale_boxes)
    from utils.plots import Annotator, colors
    from utils.torch_utils import select_device, smart_inference_mode
    USE_ULTRA = False
except ImportError:
    # Fallback to ultralytics if YOLOv9 modules not available
    from ultralytics import YOLO
    USE_ULTRA = True


@smart_inference_mode()
def run_webcam_detection(
        weights='yolov9n.pt',  # model path
        source=0,  # webcam source (0 for default camera)
        data='yolov9-main/data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    """
    Run YOLOv9 object detection on webcam feed.
    """
    # Check if weights file exists
    if not os.path.exists(weights):
        print(f"Error: Model weights file '{weights}' not found!")
        print(f"Please download yolov9n.pt from:")
        print(f"https://github.com/ultralytics/yolov9/releases/download/v0.1.0/yolov9n.pt")
        print(f"and place it in the current directory.")
        return

    # Check if data file exists
    if not os.path.exists(data):
        print(f"Warning: Dataset file '{data}' not found. Using default COCO classes.")
        data = None

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Check if webcam is available
    view_img = check_imshow(warn=True)
    if not view_img:
        print("Error: Cannot display webcam feed. Please check your camera connection.")
        return

    # Load webcam stream
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = len(dataset)

    # Warmup model
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    
    # Initialize variables
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    print(f"Starting webcam detection...")
    print(f"Press 'q' to quit")
    print(f"Model: {weights}")
    print(f"Device: {device}")
    print(f"Classes: {len(names)}")

    # Main detection loop
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s = f'{i}: '

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Draw bounding boxes and labels
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Display results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                
                # Check for 'q' key press to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nExiting...")
                    break

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Cleanup
    cv2.destroyAllWindows()
    print(f"Detection completed. Processed {seen} frames.")


def main():
    """Main function to run webcam detection."""
    parser = argparse.ArgumentParser(description='YOLOv9 Webcam Detection')
    parser.add_argument('--weights', type=str, default='yolov9n.pt', help='model path')
    parser.add_argument('--source', type=int, default=0, help='webcam source (0 for default camera)')
    parser.add_argument('--data', type=str, default='yolov9-main/data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    
    print(f"YOLOv9 Webcam Detection")
    print(f"Arguments: {vars(opt)}")
    
    run_webcam_detection(**vars(opt))


if __name__ == "__main__":
    main()
