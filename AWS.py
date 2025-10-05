import cv2
import boto3

# Initialize Rekognition client
rekognition = boto3.client('rekognition', region_name='us-west-2')

# Open webcam once, globally
cap = cv2.VideoCapture(0)

def promptAWS():
    """
    Capture a single frame from webcam and detect labels with AWS Rekognition.
    Returns a list of labels detected in this frame.
    """
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        return []

    # Encode frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    byte_frame = buffer.tobytes()

    # Call Rekognition
    response = rekognition.detect_labels(
        Image={'Bytes': byte_frame},
        MaxLabels=20,
        MinConfidence=30
    )

    # Collect labels
    items = [label['Name'] for label in response['Labels']]
    # Optional: print items for debugging
    # print(items)

    # Show webcam feed (optional, can comment out if not needed)
    # cv2.imshow("Webcam", frame)
    # cv2.waitKey(1)
    

    return items

# Cleanup function (call at the end of your program)
def release_camera():
    cap.release()
    cv2.destroyAllWindows()
