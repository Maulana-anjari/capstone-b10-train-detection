import cv2
from picamera2 import Picamera2
from vidgear.gears import NetGear
import socket
import struct
import threading
import RPi.GPIO as GPIO
from gpiozero import Servo
from time import sleep

# Video Stream Setup
options = {
    "flag": 0,
    "copy": True,
    "track": False,
    "jpeg_compression": True,
    "jpeg_compression_quality": 90,
    "jpeg_compression_fastdct": True,
    "jpeg_compression_fastupsample": True,
}
server = NetGear(address="192.168.82.35", port="5454", protocol="tcp", pattern=1, logging=True, **options)

# GPIO Setup for System 1
GPIO.setmode(GPIO.BCM)
GPIO.setup(16, GPIO.OUT)  # Pin for yellow LED (System 1)
GPIO.setup(17, GPIO.OUT)  # Pin for buzzer (System 1)
GPIO.setup(22, GPIO.OUT)  # Pin for green LED (System 1)

# Setup for System 1 Components
servo1 = Servo(12)  # GPIO 12 for Servo (System 1)
yellow_led1 = GPIO.PWM(16, 100)  # PWM for yellow LED (System 1)
green_led1 = GPIO.PWM(22, 100)  # PWM for green LED (System 1)
buzzer1 = GPIO.PWM(17, 1000)  # PWM for buzzer (System 1)

# GPIO Setup for System 2
GPIO.setup(23, GPIO.OUT)  # Pin for red LED (System 2)
GPIO.setup(24, GPIO.OUT)  # Pin for buzzer (System 2)
GPIO.setup(25, GPIO.OUT)  # Pin for green LED (System 2)

# Setup for System 2 Components
servo2 = Servo(13)  # GPIO 13 for Servo (System 2)
red_led2 = GPIO.PWM(23, 100)  # PWM for red LED (System 2)
green_led2 = GPIO.PWM(25, 100)  # PWM for green LED (System 2)
buzzer2 = GPIO.PWM(24, 1000)  # PWM for buzzer (System 2)

# Initial LED status
green_led1.start(100)
green_led2.start(100)

# Message Socket Setup
message_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
message_socket.bind(("0.0.0.0", 9999))
message_socket.listen(5)
print("Listening for messages on port 9999...")

# Flags and global variables
alarm_active = False
latest_confidence_score = 0.0  # To store the latest confidence score

def receive_confidence_scores():
    global alarm_active, latest_confidence_score
    """Function to receive confidence scores and set alarm if score is above threshold."""
    while True:
        client_socket, addr = message_socket.accept()
        print("Connected to:", addr)
        with client_socket:
            while True:
                # Receive message length
                message_length = client_socket.recv(struct.calcsize("Q"))
                if not message_length:
                    break
                message_size = struct.unpack("Q", message_length)[0]

                # Receive and decode the confidence score message
                message = client_socket.recv(message_size).decode()
                try:
                    confidence_score = float(message)
                    latest_confidence_score = confidence_score  # Update the latest confidence score
                    if confidence_score > 0.5:
                        alarm_active = True  # Set alarm if confidence is above 0.5
                    else:
                        alarm_active = False
                except ValueError:
                    print("Received non-numeric message:", message)

def stream_video_frames():
    """Function to capture and send video frames."""
    picam2 = Picamera2()
    picam2.start()
    try:
        while True:
            frame = picam2.capture_array()
            if frame is None:
                break

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            server.send(frame_bgr)

    finally:
        picam2.stop()
        server.close()

def activate_alarm():
    """Function to activate both alarm systems (LED, buzzer, and servo) when alarm is triggered."""
    global alarm_active, latest_confidence_score
    while True:
        if alarm_active:
            print(f"Confidence Score: {latest_confidence_score:.2f} - TERDETEKSI KERETA, ALARM AKTIF")

            # Alarm active: System 1 (yellow LED, buzzer, servo) and System 2 (red LED, buzzer, servo) activate
            for _ in range(6):  # Blink for 5 seconds (10 cycles of 0.5-second intervals)
                green_led1.ChangeDutyCycle(0)  # Turn off green LED for System 1
                green_led2.ChangeDutyCycle(0)  # Turn off green LED for System 2
                
                # Move servos to "close" position
                servo1.value = -1
                servo2.value = -1
                sleep(0.5)
                
                # Activate System 1 alarm components
                yellow_led1.start(100)  # Turn on yellow LED
                buzzer1.start(50)  # Start buzzer for System 1
                sleep(0.25)
                
                yellow_led1.stop()  # Turn off yellow LED for System 1
                buzzer1.stop()  # Stop buzzer for System 1
                sleep(0.25)
                
                # Activate System 2 alarm components
                red_led2.start(100)  # Turn on red LED
                buzzer2.start(50)  # Start buzzer for System 2
                sleep(0.25)
                
                red_led2.stop()  # Turn off red LED for System 2
                buzzer2.stop()  # Stop buzzer for System 2
                sleep(0.25)

            # Reset servos to initial position and green LEDs back on
            servo1.value = 1
            servo2.value = 1
            sleep(0.5)
            servo1.detach()  # Turn off PWM signal to servo for System 1
            servo2.detach()  # Turn off PWM signal to servo for System 2
            green_led1.ChangeDutyCycle(100)  # Turn green LED back on for System 1
            green_led2.ChangeDutyCycle(100)  # Turn green LED back on for System 2

            alarm_active = False  # Reset alarm flag after activation

        sleep(0.1)  # Small delay to reduce CPU usage

# Start the confidence score-receiving thread
confidence_thread = threading.Thread(target=receive_confidence_scores, daemon=True)
confidence_thread.start()

# Start the video frame streaming thread
video_thread = threading.Thread(target=stream_video_frames, daemon=True)
video_thread.start()

# Start the alarm activation loop
activate_alarm()  # Runs continuously
