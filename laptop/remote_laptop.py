# Import required libraries
from vidgear.gears import NetGear
import cv2

# Define tweak flags for NetGear client
options = {"flag": 0, "copy": True, "track": False}

# Define Netgear Client to receive frames from the Raspberry Pi
client = NetGear(
    address="192.168.82.35",
    port="5454",
    protocol="tcp",
    pattern=1,
    receive_mode=True,
    logging=False,
    **options
)

# Loop to receive frames and display them
while True:
    # Receive frames from network
    frame = client.recv()

    # Check if received frame is None (end of stream)
    if frame is None:
        break

    # Display the received frame
    cv2.imshow("Output Frame", frame)

    # Check for 'q' key to break the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Close the display window and safely close the client
cv2.destroyAllWindows()
client.close()