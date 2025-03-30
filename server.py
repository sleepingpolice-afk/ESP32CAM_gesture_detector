import cv2
import numpy as np
import urllib.request
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

url = "http://192.168.110.48/cam-hi.jpg"  # Replace with MJPEG stream if available

while True:
    try:
        # Fetch image manually
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)

        if frame is None:
            print("Failed to decode image")
            continue

        # Pose detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow('ESP32-CAM Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        continue

cv2.destroyAllWindows()