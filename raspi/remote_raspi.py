# Import required libraries
from vidgear.gears import NetGear
from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import time
import threading
import RPi.GPIO as GPIO
from gpiozero import Servo
from time import sleep

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(22, GPIO.OUT)  # Red LED 1
GPIO.setup(17, GPIO.OUT)  # Green LED 1
GPIO.setup(24, GPIO.OUT)  # Buzzer 1
GPIO.setup(6, GPIO.OUT)   # Red LED 2
GPIO.setup(26, GPIO.OUT)  # Green LED 2
GPIO.setup(25, GPIO.OUT)  # Buzzer 2

# Setup for components
servo1 = Servo(12)  # Servo 1
red_led1 = GPIO.PWM(22, 100)
green_led1 = GPIO.PWM(17, 100)
buzzer1 = GPIO.PWM(24, 1000)
red_led2 = GPIO.PWM(6, 100)
green_led2 = GPIO.PWM(26, 100)
buzzer2 = GPIO.PWM(25, 1000)

# Initialize components
green_led1.start(100)
green_led2.start(100)

# Initialize Picamera2
picam2 = Picamera2()
picam2.start()

# Load the YOLO model
model = YOLO("/home/kepstun/project/bestv3_ncnn_model")  # Adjust path as needed

# Define NetGear options for server
options = {
    "flag": 0,
    "copy": True,
    "track": False,
    "jpeg_compression": True,
    "jpeg_compression_quality": 90,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": True,
}

def create_server():
    """Create a NetGear server."""
    return NetGear(
        address="192.168.82.35",
        port="5454",
        protocol="tcp",
        pattern=1,
        logging=False,
        **options
    )

server = None
streaming_active = False

# Global flag for alarm status
alarm_active = False

def activate_alarm(confidence_score):
    """Activate the alarm based on detection confidence."""
    global alarm_active
    if confidence_score > 0.5:  # Threshold to trigger alarm
        print(f"Confidence Score: {confidence_score:.2f} - Alarm Activated")
        alarm_active = True

        # Turn off green LEDs
        green_led1.ChangeDutyCycle(0)
        green_led2.ChangeDutyCycle(0)

        # Move servo to "close" position and activate alarms
        servo1.value = -1
        for _ in range(6):  # Blink and sound alarm for 3 seconds
            red_led1.start(100)
            buzzer1.start(50)
            sleep(0.25)
            red_led1.stop()
            buzzer1.stop()
            sleep(0.25)
            red_led2.start(100)
            buzzer2.start(50)
            sleep(0.25)
            red_led2.stop()
            buzzer2.stop()
            sleep(0.25)

        # Reset components
        servo1.value = 1
        sleep(0.5)
        servo1.detach()
        green_led1.ChangeDutyCycle(100)
        green_led2.ChangeDutyCycle(100)

    else:
        alarm_active = False

def detection_thread():
    """Object detection and streaming function."""
    global server, streaming_active
    prev_time = time.time()  # Initialize previous time for FPS calculation

    while True:
        try:
            # Capture a frame
            frame = picam2.capture_array()
            
            # Rotate the frame by 180 degrees (flip it upside down)
            frame_bgr = cv2.rotate(cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR), cv2.ROTATE_180)

            # Perform object detection
            results = model(frame_bgr)
            detections = results[0].boxes

            # Draw bounding boxes and calculate confidence
            highest_confidence = 0.0
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = box.cls[0]
                confidence = box.conf[0]
                highest_confidence = max(highest_confidence, confidence)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame_bgr, f"{label} {confidence:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

            # Activate alarm based on highest confidence score
            activate_alarm(highest_confidence)

            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            cv2.putText(frame_bgr, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Send the frame if streaming is active
            if streaming_active and server:
                try:
                    server.send(frame_bgr)
                except Exception:
                    streaming_active = False

        except KeyboardInterrupt:
            break

def stream_thread():
    """Handle streaming control."""
    global server, streaming_active
    while True:
        key = input("Press 'r' to start/stop streaming: ").strip().lower()
        if key == 'r':
            if streaming_active:
                streaming_active = False
                if server:
                    server.close()
                    server = None
            else:
                server = create_server()
                streaming_active = True

# Start detection and streaming threads
detection_thread = threading.Thread(target=detection_thread)
stream_thread = threading.Thread(target=stream_thread)
detection_thread.start()
stream_thread.start()

# Wait for threads to complete
detection_thread.join()
stream_thread.join()

# Cleanup
picam2.stop()
if server:
    server.close()
GPIO.cleanup()
