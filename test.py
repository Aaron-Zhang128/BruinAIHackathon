import cv2
import boto3
import time

# Initialize Rekognition client
rekognition = boto3.client('rekognition', region_name='us-west-2')

# Open webcam (0 = default camera, 1 = external USB cam, etc.)
cap = cv2.VideoCapture(0)

frame_number = 0

while True:
    items = []
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera")
        break

    frame_number += 1

    # Encode frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    byte_frame = buffer.tobytes()

    # Call Rekognition for label detection
    response = rekognition.detect_labels(
        Image={'Bytes': byte_frame},
        MaxLabels=15,
        MinConfidence=50
    )

    # Print results for this frame
    print(f"\nFrame {frame_number}:")
    for label in response['Labels']:
        print(f" - {label['Name']} ({label['Confidence']:.2f}%)")
        items.append(label['Name'])
    print(items)
    # Show the live camera feed in a window
    cv2.imshow("Webcam", frame)

    # Wait 1 second between frames to control API usage
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(1)  # ~1 fps (adjust/remove if you want faster)
    items = []
cap.release()
cv2.destroyAllWindows()
