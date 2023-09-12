import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

# YOLO-specific imports
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# OpenFace-specific imports
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

import face_recognition
from sort import Sort

def detect(save_img=False):
    # Rest of the existing code...

    # Load YOLO model
    yolo_weights = 'D:\Gardena\Skripsi\Program\counting-and-tracking\yolo.pt'  # Replace with the path to your YOLO weights
    yolo_cfg = 'path/to/your/yolo/cfg'  # Replace with the path to your YOLO configuration file
    yolo_names = 'D:\Gardena\Skripsi\Program\counting-and-tracking\coco.names'  # Replace with the path to your YOLO class names file

    yolo_model = attempt_load(yolo_weights, map_location=device)
    yolo_names = load_classifier(yolo_names)

    # Process detections using YOLO
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference using YOLO
        t1 = time_synchronized()
        yolo_pred = yolo_model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS using YOLO results
        yolo_pred = non_max_suppression(yolo_pred, opt.conf_thres, opt.iou_thres)

        # Process YOLO detections and draw bounding boxes
        for pred in yolo_pred:
            if pred is not None and len(pred):
                # Rescale YOLO boxes from img_size to im0 size
                pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()

                # Extract class labels and confidences
                yolo_classes = pred[:, -1].cpu().numpy().astype(int)
                yolo_confidences = pred[:, 4].cpu().numpy()

                # Draw YOLO bounding boxes
                for i, box in enumerate(pred):
                    x1, y1, x2, y2 = [int(i) for i in box[:4]]
                    confidence = yolo_confidences[i]
                    class_index = yolo_classes[i]
                    class_name = yolo_names[class_index]

                    # Draw the bounding box and label on the image
                    label = f'{class_name} {confidence:.2f}'
                    cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Assuming im0 adalah gambar hasil deteksi dari langkah sebelumnya

        # 1. Ekstraksi wajah menggunakan OpenFace
        face_locations = face_recognition.face_locations(im0)
        face_encodings = face_recognition.face_encodings(im0, face_locations)

        # 2. Load gambar referensi untuk perbandingan
        reference_image_path = 'D:\Gardena\Skripsi\Program\counting-and-tracking\coba.mp4'  # Ganti dengan path ke gambar referensi
        reference_image = face_recognition.load_image_file(reference_image_path)
        reference_face_encoding = face_recognition.face_encodings(reference_image)[0]

        # 3. Perbandingan wajah dengan gambar referensi menggunakan euclidean distance atau metode lain
        for face_location, face_encoding in zip(face_locations, face_encodings):
            # Perhitungan jarak antara wajah yang dikenali dengan wajah referensi
            distance = np.linalg.norm(face_encoding - reference_face_encoding)
            # Misalnya, jika jaraknya di bawah batas tertentu, anggap itu sebagai wajah yang sama
            if distance < 0.6:
                cv2.rectangle(im0, (face_location[3], face_location[0]), (face_location[1], face_location[2]),
                              (255, 0, 0), 2)
                cv2.putText(im0, 'Match', (face_location[3], face_location[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 2)
            else:
                cv2.rectangle(im0, (face_location[3], face_location[0]), (face_location[1], face_location[2]),
                              (0, 0, 255), 2)
                cv2.putText(im0, 'Not Match', (face_location[3], face_location[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)

        # 4. Menampilkan gambar hasil deteksi dan perbandingan
        cv2.imshow('Detection and Comparison', im0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    parser.add_argument('--track', action='store_true', help='run tracking')
    parser.add_argument('--show-track', action='store_true', help='show tracked path')
    parser.add_argument('--show-fps', action='store_true', help='show fps')
    parser.add_argument('--thickness', type=int, default=2, help='bounding box and font size thickness')
    parser.add_argument('--seed', type=int, default=1, help='random seed to control bbox colors')
    parser.add_argument('--nobbox', action='store_true', help='don`t show bounding box')
    parser.add_argument('--nolabel', action='store_true', help='don`t show label')
    parser.add_argument('--unique-track-color', action='store_true', help='show each track in unique color')

    opt = parser.parse_args()
    print(opt)
    np.random.seed(opt.seed)

    sort_tracker = Sort(max_age=5,
                        min_hits=2,
                        iou_threshold=0.2)

    # check_requirements(exclude=('pycocotools', 'thop'))
    device = select_device(opt.device)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolo.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
