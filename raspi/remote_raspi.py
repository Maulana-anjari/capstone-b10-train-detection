# Import required libraries
import RPi.GPIO as GPIO
from gpiozero import Servo
from time import sleep, time
from vidgear.gears import NetGear
from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import threading
from collections import deque

# GPIO Pin Definitions
RED_LED_PIN = 16    # Output LED Merah (Kuning)
GREEN_LED_PIN = 22  # Output LED Hijau
BUZZER_PIN1 = 17    # Output Buzzer 1
BUZZER_PIN2 = 24    # Output Buzzer 2
SERVO_PIN = 12      # Pin for Servo motor

# Define activation settings
activation_time = 5          # Time in seconds the alarm will stay active
confidence_threshold = 0.5   # Confidence threshold for triggering the alarm

# Setup GPIO mode and pins
GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN1, GPIO.OUT)
GPIO.setup(BUZZER_PIN2, GPIO.OUT)

# Setup Servo using gpiozero
servo = Servo(SERVO_PIN)

# Setup LED and Buzzers using PWM
green_led = GPIO.PWM(GREEN_LED_PIN, 100)  # Using PWM for green LED (optional)
buzzer1 = GPIO.PWM(BUZZER_PIN1, 1000)     # Using PWM for Buzzer 1
buzzer2 = GPIO.PWM(BUZZER_PIN2, 1000)     # Using PWM for Buzzer 2

# Initialize Picamera2
picam2 = Picamera2()
picam2.start()

# Load the YOLO model
model = YOLO("/home/kepstun/project/bestv3_ncnn_model")  # Ensure this path is correct

# Define tweak flags for NetGear server
options = {
    "flag": 0,
    "copy": True,
    "track": False,
    "jpeg_compression": True,
    "jpeg_compression_quality": 90,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": True,
}

# Function to reset the alarm state
def reset_alarm_state():
    GPIO.output(RED_LED_PIN, GPIO.LOW)
    green_led.start(100)  # Green LED ON by default
    buzzer1.stop()
    buzzer2.stop()
    servo.value = 1  # Reset servo to initial position
    sleep(0.1)
    servo.detach()  # Detach to stop sending PWM signal

    # Set buzzer pins to INPUT to fully disable them
    GPIO.setup(BUZZER_PIN1, GPIO.IN)
    GPIO.setup(BUZZER_PIN2, GPIO.IN)

# Function to activate the alarm
def activate_alarm():
    print("Activating Alarm...")

    # Set buzzer pins back to OUTPUT mode before starting the alarm
    GPIO.setup(BUZZER_PIN1, GPIO.OUT)
    GPIO.setup(BUZZER_PIN2, GPIO.OUT)
    servo.value = -1  # Move servo to "closed" position

    start_time = time()  # Record the current time

    while time() - start_time < activation_time:
        # Turn off green LED
        green_led.ChangeDutyCycle(0)

        # Activate red LED, buzzers, and move servo
        GPIO.output(RED_LED_PIN, GPIO.HIGH)
        buzzer1.start(50)  # Start buzzer 1 at 50% duty cycle
        buzzer2.start(50)  # Start buzzer 2 at 50% duty cycle
        sleep(0.2)

        # Deactivate red LED, buzzers
        buzzer1.stop()
        buzzer2.stop()
        sleep(0.2)

        GPIO.output(RED_LED_PIN, GPIO.LOW)
        buzzer1.start(50)
        buzzer2.start(50)
        sleep(0.2)

        buzzer1.stop()
        buzzer2.stop()
        sleep(0.2)

    # Reset to initial state after alarm cycle
    reset_alarm_state()

# Initialize NetGear Server but do not start streaming immediately
def create_server():
    return NetGear(
        address="192.168.82.35",  # Change to your laptop's IP
        port="5454",
        protocol="tcp",
        pattern=1,
        logging=False,
        **options
    )

server = None

# Variables for controlling streaming
streaming_active = False
prev_time = 0

# Variables for FPS moving average calculation
fps_window = deque(maxlen=10)  # Store the last 10 FPS values for smoothing

# Function to handle object detection and sending frames
def detection_thread():
    global server, streaming_active, prev_time

    while True:
        try:
            # Capture a frame from the camera
            frame = picam2.capture_array()

            # Flip the frame vertically
            frame = cv2.flip(frame, 0)

            # Convert from RGBA (or RGB) to BGR if needed
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            # Start time for FPS calculation
            start_time = time()

            # Perform object detection
            results = model(frame_bgr)
            detections = results[0].boxes

            # Draw bounding boxes and check confidence score
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0].item())
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame_bgr, f"{confidence * 100:.2f}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

                # Trigger the alarm if confidence exceeds threshold
                if confidence > confidence_threshold:
                    activate_alarm()

            # Calculate FPS and update moving average
            end_time = time()
            fps = 1 / (end_time - start_time)
            fps_window.append(fps)
            avg_fps = sum(fps_window) / len(fps_window)

            # Display averaged FPS on the frame
            cv2.putText(
                frame_bgr, f"FPS: {avg_fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

            # Send the frame if streaming is active
            if streaming_active and server:
                try:
                    server.send(frame_bgr)
                except Exception:
                    streaming_active = False

        except KeyboardInterrupt:
            break

# Function to handle streaming control
def stream_thread():
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

# Start the threads
detection_thread = threading.Thread(target=detection_thread)
stream_thread = threading.Thread(target=stream_thread)
detection_thread.start()
stream_thread.start()

# Wait for both threads to complete
detection_thread.join()
stream_thread.join()

# Cleanup
picam2.stop()
if server:
    server.close()
GPIO.cleanup()
