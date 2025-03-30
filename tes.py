import cv2 as cv

capture = cv.VideoCapture("2025-03-20 22-21-29.mkv")

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('a'):
        break

capture.release()
cv.destroyAllWindows()