from fer import FER
import cv2
from fer.utils import draw_annotations
import time


# Resources to learn how to write this code came from FER documentation and examples, as well as geeksforgeeks.org

detector = FER()
capture = cv2.VideoCapture(0)

prevFrameTime = 0
newFrameTime = 0
while True:
    ret, frame = capture.read()

    frame = cv2.flip(frame, 1)

    emotions = detector.detect_emotions(frame)
    frame = draw_annotations(frame, emotions)

    newFrameTime = time.time()
    fps = 1 / (newFrameTime - prevFrameTime)
    prevFrameTime = newFrameTime

    fps = str(int(fps)) # make easier to read as an int

    cv2.putText(frame, "FPS: " + fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA) 

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()

