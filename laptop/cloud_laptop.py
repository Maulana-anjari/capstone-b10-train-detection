import cv2
import torch
from ultralytics import YOLO
from vidgear.gears import NetGear
import socket
import struct
import threading
import time
from queue import Queue

# Initialize YOLO model (NCNN)
model = YOLO("bestv2.pt")

# Video Stream Setup
options = {"flag": 0, "copy": True, "track": False}
client = NetGear(address="192.168.82.35", port="5454", protocol="tcp", pattern=1, receive_mode=True, logging=True, **options)

# Message Socket Setup
message_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
message_host_ip = '192.168.82.220'
message_port = 9999
message_socket.connect((message_host_ip, message_port))

# Queue for frames
frame_queue = Queue(maxsize=5)

# Function to send confidence scores
def send_confidence_scores(detections):
    for box in detections:
        confidence = box.conf[0].item()
        message = f"{confidence:.2f}"
        message_length = struct.pack("Q", len(message))
        try:
            message_socket.sendall(message_length + message.encode())
        except Exception as e:
            pass  # Handle exceptions silently to avoid blocking

# Detection thread function with FPS display
def detection_thread():
    prev_time = 0  # Initialize previous time for FPS calculation

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Perform YOLO detection
            results = model(frame)
            detections = results[0].boxes

            # Send confidence scores in a separate thread
            threading.Thread(target=send_confidence_scores, args=(detections,), daemon=True).start()

            # Draw bounding boxes and labels
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                label = box.cls[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
            prev_time = current_time

            # Display FPS on the frame
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the processed frame
            cv2.imshow("Output Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

# Start the detection thread
threading.Thread(target=detection_thread, daemon=True).start()

# Main loop to receive frames
try:
    while True:
        frame = client.recv()
        if frame is None:
            break

        # Add frame to the queue for processing
        if not frame_queue.full():
            frame_queue.put(frame)

finally:
    client.close()
    message_socket.close()
    cv2.destroyAllWindows()
