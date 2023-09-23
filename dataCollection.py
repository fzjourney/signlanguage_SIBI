import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1) #Deteksi n banyak tangan

offset = 20 # jarak agar saat di crop tidak terlalu kecil
imgSize = 300 # besar img (pixel)

folder = "Data/Y" # directory penyimpanan data set
counter = 0 # banyak image yang di save

while True:
    success, img = cap.read() # variable img untuk menyimpan hasil dari video capture
    hands, img = detector.findHands(img) # menggunakan fungsi findHands() dan menyimpannya ke dalam variable img
    if hands: # jika ada tangan
        hand = hands[0]
        x, y, w, h = hand['bbox'] # koordinat box

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255 # buat canvas putih sebesar imgSize x imgSize dan
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset] # crop image tetapi tambah offset jadi tidak terlalu mepet

        imgCropShape = imgCrop.shape # simpan dimensi imgCrop ke imgCropShape

        aspectRatio = h / w # mencari aspect ratio

        if aspectRatio > 1: # jika h > w --> ini untuk resize si canvas putih tadi
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"): # jika s dipencet maka akan save image dan counter++
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)