from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# for webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
# for video
cap = cv2.VideoCapture("../videos/cars.mp4")

model = YOLO('../YOLO_weights/yolov8l.pt')

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

mask = cv2.imread("mask.png")

# Tracker instance
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# limit for the line from which we count the cars
limits = [410, 297, 683, 297] 
idCarPassed = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    # putting limit line on the img
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,0,255), 5)
    for r in results:
        boxes = r.boxes
        # looping over the detections
        for box in boxes:
            # getting the boxes of each detection
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)

            # getting the confidence
            conf = math.ceil((box.conf[0]*100))/100

            # putting the class name on the screen
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if (currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike") and conf > 0.3:
                # puts rectagle on the screen
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255), 1)
                # puts confidence and class on the screen
                cv2.putText(img, f'{currentClass} {conf}', (x1+10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1,y1,x2,y2, Id = result
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx,cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1]-20 < cy < limits[3]+20 and Id not in idCarPassed:
            idCarPassed.append(Id)
            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,255,0), 5)
        
    # puts the count on the image
    cv2.putText(img, f'Count: {len(idCarPassed)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)