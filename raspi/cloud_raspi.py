import RPi.GPIO as GPIO
from gpiozero import Servo
from time import sleep, time
import cv2
from picamera2 import Picamera2
from vidgear.gears import NetGear
import socket
import struct
import threading

# Configuration Variables
CONFIDENCE_THRESHOLD = 80  # Set the threshold for triggering the alarm
activation_time = 4        # Time in seconds the alarm will stay active
cooldown_period = 1        # Cooldown period in seconds
last_activation_time = 0   # Store the time of the last alarm activation

# GPIO Pin Definitions
RED_LED_PIN = 16    # Output LED Merah (Kuning)
GREEN_LED_PIN = 22  # Output LED Hijau
BUZZER_PIN1 = 17    # Output Buzzer 1
BUZZER_PIN2 = 24    # Output Buzzer 2
SERVO_PIN = 12      # Pin for Servo motor

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

# Message Socket Setup
message_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
message_socket.bind(("0.0.0.0", 9999))
message_socket.listen(5)
print("Listening for messages on port 9999...")

# Flag to prevent multiple activations of the alarm
alarm_triggered = False

# Function to reset the alarm state
def reset_alarm_state():
    global alarm_triggered
    GPIO.output(RED_LED_PIN, GPIO.LOW)
    green_led.start(100)
    buzzer1.stop()
    buzzer2.stop()
    servo.value = 1
    sleep(0.1)
    servo.detach()
    GPIO.setup(BUZZER_PIN1, GPIO.IN)
    GPIO.setup(BUZZER_PIN2, GPIO.IN)
    alarm_triggered = False

# Function to activate the alarm
def activate_alarm():
    global alarm_triggered, last_activation_time
    if alarm_triggered:
        return

    alarm_triggered = True
    print("Activating Alarm...")
    GPIO.setup(BUZZER_PIN1, GPIO.OUT)
    GPIO.setup(BUZZER_PIN2, GPIO.OUT)
    servo.value = -1

    start_time = time()
    while time() - start_time < activation_time:
        green_led.ChangeDutyCycle(0)
        GPIO.output(RED_LED_PIN, GPIO.HIGH)
        buzzer1.start(50)
        buzzer2.start(50)
        sleep(0.2)
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

    reset_alarm_state()
    last_activation_time = time()

# Function to receive confidence scores and activate the alarm if a threshold is met
def receive_confidence_scores():
    global alarm_triggered, last_activation_time
    while True:
        client_socket, addr = message_socket.accept()
        print("Connected to:", addr)
        with client_socket:
            while True:
                message_length = client_socket.recv(struct.calcsize("Q"))
                if not message_length:
                    break
                message_size = struct.unpack("Q", message_length)[0]
                message = client_socket.recv(message_size).decode()

                try:
                    confidence_score = float(message)
                    print("Confidence Score:", confidence_score)
                    current_time = time()
                    if confidence_score > CONFIDENCE_THRESHOLD and not alarm_triggered and (current_time - last_activation_time > cooldown_period):
                        print("Confidence score exceeded threshold, activating alarm!")
                        activate_alarm()
                except ValueError:
                    print("Received non-numeric message:", message)

# Start the confidence score-receiving thread
threading.Thread(target=receive_confidence_scores, daemon=True).start()

# Initialize Picamera2
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
    message_socket.close()
    GPIO.cleanup()

