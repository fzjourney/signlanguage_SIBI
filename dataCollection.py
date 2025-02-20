import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Set webcam properties (optional, improve brightness)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

# Initialize hand detector with better settings
detector = HandDetector(maxHands=1, detectionCon=0.5, minTrackCon=0.5)

# Parameters
offset = 20  # Padding for hand cropping
imgSize = 300  # Final image size
letters = list("QRSTUVWXYZ")  # Letter sequence
folder_index = 0  # Start from 'A'
counter = 0  # Image counter

while folder_index < len(letters):
    folder = f"Data/{letters[folder_index]}"
    os.makedirs(folder, exist_ok=True)
    
    success, img = cap.read()
    if not success:
        print("❌ Error: Failed to read frame from webcam.")
        continue
    
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y1:y2, x1:x2]
        
        if imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
            print("⚠️ Warning: Cropped image is empty, skipping.")
            continue
        
        aspectRatio = h / w
        if aspectRatio > 1:
            scaleFactor = imgSize / h
            newWidth = math.ceil(scaleFactor * w)
            imgResize = cv2.resize(imgCrop, (newWidth, imgSize))
            wGap = math.ceil((imgSize - newWidth) / 2)
            imgWhite[:, wGap:wGap + newWidth] = imgResize
        else:
            scaleFactor = imgSize / w
            newHeight = math.ceil(scaleFactor * h)
            imgResize = cv2.resize(imgCrop, (imgSize, newHeight))
            hGap = math.ceil((imgSize - newHeight) / 2)
            imgWhite[hGap:hGap + newHeight, :] = imgResize
        
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    
    cv2.imshow("Webcam Feed", img)
    
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        imgPath = f"{folder}/Image_{time.time()}.jpg"
        cv2.imwrite(imgPath, imgWhite)
        print(f"✅ Saved {imgPath} ({counter} images in {letters[folder_index]})")
        
        if counter >= 5:
            counter = 0  # Reset counter for next letter
            folder_index += 1  # Move to next letter
            if folder_index >= len(letters):
                print("✅ Already Satisfied until Z")
                break
    elif key == ord("q"):
        print("🔴 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
