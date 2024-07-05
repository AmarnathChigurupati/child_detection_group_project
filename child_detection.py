import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import math
import time

# Load YOLOv3
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize Tkinter
root = tk.Tk()
root.withdraw()  # Hide the main window

# Helper function to display pop-up
def show_popup(title, message):
    messagebox.showinfo(title, message)

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to detect objects in frame
def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [(class_ids[i], boxes[i]) for i in indices.flatten()]

# Video capture from webcam or video file
cap = cv2.VideoCapture(0)

child_position = None
start_time = time.time()
log_file = open("detection_log.txt", "w")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_objects(frame)

    for (class_id, box) in detections:
        x, y, w, h = box
        label = str(classes[class_id])
        if label == "person":
            if child_position is None:
                child_position = (x + w // 2, y + h // 2)
            else:
                new_position = (x + w // 2, y + h // 2)
                distance_moved = calculate_distance(child_position, new_position)
                if distance_moved > 500:
                    show_popup("Child Alert", "The child has moved more than 500 meters!")
                    log_file.write(f"Child moved: {distance_moved} meters\n")
                    child_position = new_position
        elif label in ["knife", "scissors", "sharp object"]:  # Add appropriate sharp objects here
            if child_position:
                sharp_object_position = (x + w // 2, y + h // 2)
                distance_to_child = calculate_distance(child_position, sharp_object_position)
                if distance_to_child < 250:
                    show_popup("Sharp Object Alert", "A sharp object is detected within 250 meters of the child!")
                    log_file.write(f"Sharp object detected at {distance_to_child} meters from child\n")

    # Show frame with detections
    for (class_id, box) in detections:
        x, y, w, h = box
        label = str(classes[class_id])
        color = (0, 255, 0) if label == "person" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()
