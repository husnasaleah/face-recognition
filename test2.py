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

# print(len(imgModeList))
# print(modePathList)




while True:
    success, img = cap.read()
    
   
    # Check if the image has valid size
    # if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
    #     print("Error: Invalid image size.")
    #     continue

    img_resized = cv2.resize(img, (desired_width, desired_height))
    imgFlipped = cv2.flip(img_resized, 1)

    roi_x1 = int((imgBg.shape[1] - desired_width) / 2)
    roi_x2 = roi_x1 + desired_width
    roi_y1 = distance_from_top
    roi_y2 = roi_y1 + desired_height

    imgBg[roi_y1:roi_y2, roi_x1:roi_x2] = imgFlipped

    if imgModeList:
        mode_x1 = int((imgBg.shape[1] - imgModeList[0].shape[1]) / 2)
        mode_x2 = mode_x1 + imgModeList[0].shape[1]
        mode_y1 = imgBg.shape[0] - imgModeList[0].shape[0]
        mode_y2 = imgBg.shape[0]

        imgBg[mode_y1:mode_y2, mode_x1:mode_x2] = imgModeList[0]

    

     # ทำการแปลงภาพให้เป็นขาวดำ (grayscale) และใช้ Haar Cascade Classifier ตรวจหาใบหน้า
    gray_scale = cv2.cvtColor(imgBg, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_scale)
    # print('faces:',faces)
    # หากมีใบหน้า
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            offset = 20  # กำหนด offset ขนาดเดียวกับทั้งสองด้านของ bbox
            new_width = w + 2 * offset
            new_height = h + 2 * offset

            # คำนวณตำแหน่งและขนาดใหม่ของ bbox
            x1 = max(0, x - offset)
            y1 = max(0, y - offset)
            x2 = min(imgBg.shape[1], x + w + offset)
            y2 = min(imgBg.shape[0], y + h + offset)
            # print(x1,y1)
            # ปรับค่า bbox ให้เป็นรูปแบบ (x, y, w, h)
            bbox = (x1, y1, new_width, new_height)
            imgBg = cvzone.cornerRect(imgBg, bbox, rt=0,t=10,l=60)
            # face_region = cv2.resize(face_region, (x+w, y+h))
            # Add a rectangle and put text inside it on the image
    # imgBg, bbox = cvzone.putTextRect(
    #             imgBg, "CVZone", (800, 1000),  # Image and starting position of the rectangle
    #             scale=3, thickness=3,  # Font scale and thickness
    #             colorT=(255, 255, 255), colorR=(255, 0, 255),  # Text color and Rectangle color
    #             font=cv2.FONT_HERSHEY_PLAIN,  # Font type
    #             offset=10,  # Offset of text inside the rectangle
    #             border=5, colorB=(0, 255, 0)  # Border thickness and color
    #         )
    cv2.imshow("Face Recognition", imgBg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
