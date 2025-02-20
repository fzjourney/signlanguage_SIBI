import cv2
import numpy as np
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector  # ✅ Import HandDetector

np.set_printoptions(suppress=True)

# Load model and labels
model = load_model("Model/keras_model.h5", compile=False)
class_names = [line.strip() for line in open("Model/labels.txt", "r").readlines()]

# Initialize webcam
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Initialize HandDetector
detector = HandDetector(detectionCon=0.8, maxHands=1)  # 0.8 confidence threshold

while True:
    ret, frame = camera.read()
    if not ret:
        print("❌ Error: Failed to capture image.")
        continue

    # Detect hands
    hands, img = detector.findHands(frame, draw=True)

    if hands:
        hand = hands[0]  # Get the first detected hand
        x, y, w, h = hand["bbox"]  # Get bounding box

        # Ensure bounding box is within frame limits
        height, width, _ = frame.shape
        x, y = max(0, x), max(0, y)
        w, h = min(width - x, w), min(height - y, h)

        # Crop hand region
        hand_crop = frame[y:y+h, x:x+w]
        
        # Check if the crop is valid
        if hand_crop is None or hand_crop.size == 0:
            print("⚠️ Warning: Invalid crop, skipping frame.")
            continue

        # Resize to model input size (224x224)
        try:
            hand_resized = cv2.resize(hand_crop, (224, 224))
        except cv2.error as e:
            print("⚠️ OpenCV Resize Error:", e)
            continue

        # Preprocess image
        img_array = np.asarray(hand_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        img_array = (img_array / 127.5) - 1  # Normalize

        # Make prediction
        prediction = model.predict(img_array)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index] * 100

        # Display prediction
        text = f"{class_name} ({confidence_score:.2f}%)"
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw bounding box
    else:
        print("⚠️ No hands detected.")

    cv2.imshow("Sign Language Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🔴 Exiting...")
        break

camera.release()
cv2.destroyAllWindows()
