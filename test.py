from roboflow import Roboflow
import cv2

rf = Roboflow(api_key="qho6x0o1Q6IzU6EItcWC")
project = rf.workspace().project("parking-space-detection-kbvvq")
model = project.version(2).model



cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    boxs = model.predict(img, confidence=40, overlap=30).json()['predictions']
    for i in range(len(boxs)):
        vt = boxs[i]
        x, y, w, h, conf, cl = vt['x'], vt['y'], vt['width'], vt['height'], vt['confidence'], vt['class']
        xmin, ymin, xmax, ymax = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
        cv2.putText(img, cl, (xmin - 10, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (232, 71, 71),2)
        cv2.rectangle(img, (xmin, ymin),(xmax, ymax),(232, 71, 71),2)

    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# img = cv2.imread('./test.jpg')
# boxs = model.predict(img, confidence=40, overlap=30).json()['predictions']
# for i in range(len(boxs)):
#     vt = boxs[i]
#     x, y, w, h = vt['x'], vt['y'], vt['width'], vt['height']
#     xmin, ymin, xmax, ymax = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
#     print(xmin, ymin, xmax, ymax)
#     cv2.rectangle(img, (xmin, ymin),(xmax, ymax),(232, 71, 71),2)
# cv2.imshow('hehe', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()