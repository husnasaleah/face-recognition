import cv2
import cvzone
import os
from deepface import DeepFace


desired_width = 1690
desired_height = 1190
distance_from_top = 505

# โหลด Haarcascades สำหรับการตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  # ใส่ 0 หรือตำแหน่งที่ถูกต้องของกล้องถ้ามีมากกว่า 1

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

actual_width = cap.get(3)
actual_height = cap.get(4)

aspect_ratio = actual_width / actual_height
new_width = int(desired_height * aspect_ratio)

cap.set(3, new_width)
cap.set(4, desired_height)

cv2.namedWindow("Face Recognition")

imgBg = cv2.imread("Resources/bg.png")

folderModePath = 'Resources/Modes'
modePathList = sorted(os.listdir(folderModePath), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else float('inf'))
imgModeList = []

for path in modePathList:
    img_path = os.path.join(folderModePath, path)
    
    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
        
        if img is not None:
            imgModeList.append(img)





while True:
    success, img = cap.read()

    img_resized = cv2.resize(img, (desired_width, desired_height))
    imgFlipped = cv2.flip(img_resized, 1)

    # Create a copy of the background image
    imgBgCopy = imgBg.copy()

    roi_x1 = int((imgBgCopy.shape[1] - desired_width) / 2)
    roi_x2 = roi_x1 + desired_width
    roi_y1 = distance_from_top
    roi_y2 = roi_y1 + desired_height

    imgBgCopy[roi_y1:roi_y2, roi_x1:roi_x2] = imgFlipped

    if imgModeList:
        mode_x1 = int((imgBgCopy.shape[1] - imgModeList[0].shape[1]) / 2)
        mode_x2 = mode_x1 + imgModeList[0].shape[1]
        mode_y1 = imgBgCopy.shape[0] - imgModeList[0].shape[0]
        mode_y2 = imgBgCopy.shape[0]

        imgBgCopy[mode_y1:mode_y2, mode_x1:mode_x2] = imgModeList[0]

    gray_scale = cv2.cvtColor(imgBgCopy, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_scale)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            offset = 20
            new_width = w + 2 * offset
            new_height = h + 2 * offset

            x1 = max(0, x - offset)
            y1 = max(0, y - offset)
            x2 = min(imgBgCopy.shape[1], x + w + offset)
            y2 = min(imgBgCopy.shape[0], y + h + offset)

            x1 = max(0, min(x1, imgBgCopy.shape[1] - 1))
            y1 = max(0, min(y1, imgBgCopy.shape[0] - 1))
            x2 = max(0, min(x2, imgBgCopy.shape[1] - 1))
            y2 = max(0, min(y2, imgBgCopy.shape[0] - 1))

            bbox = (x1, y1, x2 - x1, y2 - y1)
            imgBgCopy = cvzone.cornerRect(imgBgCopy, bbox, rt=0, t=10, l=60)
    cv2.imshow("Face Recognition", imgBgCopy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


