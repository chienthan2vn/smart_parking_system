from ultralytics import YOLO
import cv2
# Load a model
model = YOLO('./best.pt')

cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    results = model(img)
    boxs = results[0].boxes.data
    for i in range(len(results[0].boxes.data)):
        xmin, ymin, xmax, ymax, conf, label = int(boxs[i][0]), int(boxs[i][1]), int(boxs[i][2]), int(boxs[i][3]), int(boxs[i][4]), int(boxs[i][5])
        cv2.putText(img, str(label), (xmin - 10, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (232, 71, 71),2)
        cv2.rectangle(img, (xmin, ymin),(xmax, ymax),(232, 71, 71),2)
    
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# img = cv2.imread('./test.jpg')
# results = model(img)
# boxs = results[0].boxes.data
# for i in range(len(results[0].boxes.data)):
#     xmin, ymin, xmax, ymax, conf, label = int(boxs[i][0]), int(boxs[i][1]), int(boxs[i][2]), int(boxs[i][3]), int(boxs[i][4]), int(boxs[i][5])
#     cv2.putText(img, str(label), (xmin - 10, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (232, 71, 71),2)
#     cv2.rectangle(img, (xmin, ymin),(xmax, ymax),(232, 71, 71),2)
# cv2.imshow('hehe', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()