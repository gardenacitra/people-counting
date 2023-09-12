import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device, time_synchronized, TracedModel
import face_recognition
import numpy as np
import pickle

from sort import Sort
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load FaceNet model
MyFaceNet = load_model('facenet_keras.h5')

ort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)

# Function for face recognition using FaceNet model
def recognize_face(face, database):
    # Preprocess the face image
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    face = np.expand_dims(face, axis=0)

    # Get face encoding using FaceNet model
    signature = MyFaceNet.predict(face)

    # Compare the face encoding with the ones in the database
    min_dist = 100
    identity = 'Unknown'
    for key, value in database.items():
        dist = np.linalg.norm(value - signature)
        if dist < min_dist:
            min_dist = dist
            identity = key

    return identity

# 1. Definisikan variabel device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Definisikan variabel im0 (contoh)
im0 = cv2.imread('path_to_image.jpg')  # Ganti 'path_to_image.jpg' dengan path gambar Anda

# 3. Definisikan variabel names (contoh)
names = ['person', 'car', 'bicycle']

# 4. Inisialisasi model YOLOv7
model = attempt_load('yolov7.pt', map_location=device)

# 5. Definisikan variabel view_img (contoh)
view_img = True

def detect_and_recognize_faces(save_img=False):

    # Load face recognition database
    myfile = open("data.pkl", "rb")
    database = pickle.load(myfile)
    myfile.close()

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        detected_faces = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Iterate through each detected object
                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                    if names[int(detclass)] == 'person':  # Check if the detected object is a person
                        face = im0[int(y1):int(y2), int(x1):int(x2)]
                        detected_faces.append(face)

        # Recognize faces
        for face in detected_faces:
            identity = recognize_face(face, database)
            # Draw bounding box and label for the face
            cv2.putText(im0, identity, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Print time (inference + NMS)
        print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        # Stream results
        ######################################################
        if dataset.mode != 'image' and opt.show_fps:
            currentTime = time.time()

            fps = 1 / (currentTime - startTime)
            startTime = currentTime
            cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        #######################################################
        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                print(f" The image with the result is saved in: {save_path}")
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

detect_and_recognize_faces(save_img=True)  # Menyimpan gambar dengan hasil deteksi
detect_and_recognize_faces(view_img=True)  # Menampilkan gambar dengan hasil deteksi

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # ... (argumen lain tetap sama)

    opt = parser.parse_args()
    print(opt)
    np.random.seed(opt.seed)

    sort_tracker = Sort(max_age=5,
                       min_hits=2,
                       iou_threshold=0.2)

    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolo.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect_and_recognize_faces(opt)