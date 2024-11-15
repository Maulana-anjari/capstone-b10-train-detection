import cv2
import torch
from ultralytics import YOLO
from vidgear.gears import NetGear
import socket
import struct
import threading
import time
from queue import Queue
from collections import deque
import signal
import sys

# Initialize YOLO model
model = YOLO("bestv2.pt")

# Video Stream Setup
options = {"flag": 0, "copy": True, "track": False}
client = NetGear(address="192.168.82.35", port="5454", protocol="tcp", pattern=1, receive_mode=True, logging=True, **options)

# Message Socket Setup
message_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
message_host_ip = '192.168.82.220'
message_port = 9999
message_socket.connect((message_host_ip, message_port))
print("Connected to message server at", message_host_ip, "on port", message_port)

# Queue for frames
frame_queue = Queue(maxsize=5)

# Flag to indicate termination
terminate_program = False

# Function to send confidence scores
def send_confidence_scores(detections):
    for box in detections:
        confidence = float(box.conf[0].item()) * 100  # Confidence as a percentage
        message = "{:.2f}".format(confidence).rstrip('0').rstrip('.')
        message_length = struct.pack("Q", len(message))
        try:
            message_socket.sendall(message_length + message.encode())
            print("Sent confidence score:", message)  # Debugging line to confirm message sending
        except Exception as e:
            print("Failed to send message:", e)  # Print error if sending fails

# Initialize a deque to store the time taken for a certain number of frames
fps_window = deque(maxlen=10)  # Moving average over the last 10 frames

# Detection thread function with FPS display
def detection_thread():
    try:
        while not terminate_program:
            if not frame_queue.empty():
                frame = frame_queue.get()

                # Flip the frame horizontally (mirroring)
                frame = cv2.flip(frame, 1)  # Flip horizontally
                # Flip the frame vertically
                frame = cv2.flip(frame, 0)  # Flip vertically

                # Perform YOLO detection
                results = model(frame)
                detections = results[0].boxes

                # Send confidence scores for each detection
                send_confidence_scores(detections)

                # Draw bounding boxes and labels with confidence percentage
                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0].item()) * 100  # Confidence as a percentage
                    label = box.cls[0]

                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Display the confidence score as a percentage in blue
                    cv2.putText(
                        frame, f"{confidence:.2f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2  # Blue color
                    )

                # Calculate and display FPS using a moving average
                current_time = time.time()
                fps_window.append(current_time)
                if len(fps_window) > 1:
                    fps = len(fps_window) / (fps_window[-1] - fps_window[0])
                else:
                    fps = 0

                # Display FPS on the frame
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Display the processed frame
                cv2.imshow("Output Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Exiting program...")
                    break
    except KeyboardInterrupt:
        print("Detection thread interrupted.")

# Signal handler to catch Ctrl+C and exit gracefully
def signal_handler(sig, frame):
    global terminate_program
    print("Ctrl+C pressed. Terminating program...")
    terminate_program = True
    client.close()
    message_socket.close()
    cv2.destroyAllWindows()

# Set up signal handler for Ctrl+C (SIGINT)
signal.signal(signal.SIGINT, signal_handler)

# Start the detection thread
detection_thread = threading.Thread(target=detection_thread, daemon=True)
detection_thread.start()

# Main loop to receive frames
try:
    while not terminate_program:
        frame = client.recv()
        if frame is None:
            break

        # Add frame to the queue for processing
        if not frame_queue.full():
            frame_queue.put(frame)

except KeyboardInterrupt:
    print("Main loop interrupted by Ctrl+C.")
finally:
    client.close()
    message_socket.close()
    cv2.destroyAllWindows()
