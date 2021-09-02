import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 720, 540
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "handimages"
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)
    #print(lmList)

    if len(lmList)!=0:
        fingers = []
        #thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #other fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        #print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[totalFingers].shape
        img[0:h, 0:w] = overlayList[totalFingers]

        cv2.rectangle(img, (20, 350), (120, 450), (0, 255, 0), -1)
        cv2.putText(img, str(totalFingers), (45, 425), cv2.FONT_HERSHEY_PLAIN,
                    5, (255, 0, 0), 10)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS:"+str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1,
                (0, 0, 0), 1)




    cv2.imshow("Finger Counter Program", img)
    k = cv2.waitKey(1)
    if cv2.waitKey(1) and k == 27:
        break

cap.release()
cv2.destroyAllWindows()