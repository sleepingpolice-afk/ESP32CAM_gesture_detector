import cv2
import numpy as np
import urllib.request
import mediapipe as mp
import matplotlib.pyplot as plt
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

url = "http://192.168.110.48/cam-hi.jpg" 

# ty ai
def calculate_angle(a, b, c):
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(math.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle

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
            landmarks = results.pose_landmarks.landmark

            # Example: Classify pose based on wrist and shoulder positions
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate knee angle
            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

            # Classify pose
            if knee_angle < 90:  # Knees bent significantly
                feet_label = "Squat Pose"
            elif knee_angle > 170:  # Knees straightened
                feet_label = "Standing Pose"
            else:
                feet_label = "Neutral Pose"

            if (left_wrist[1] < left_shoulder[1]) and (right_wrist[1] < right_shoulder[1]):
                pose_label = "Praying Pose"
            elif (left_wrist[1] < left_shoulder[1]) and (right_wrist[1] > right_shoulder[1]):
                pose_label = "Left Hand Raised"
            elif (left_wrist[1] > left_shoulder[1]) and (right_wrist[1] < right_shoulder[1]):
                pose_label = "Right Hand Raised"
            elif (left_wrist[1] > left_shoulder[1]) and (right_wrist[1] > right_shoulder[1]):
                pose_label = "Hands Down"
            elif (left_wrist[1] < left_shoulder[1]) and (right_wrist[1] < right_shoulder[1]):
                pose_label = "Hands Together"
            

            # Display pose label on the frame
            cv2.putText(frame, pose_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, feet_label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('ESP32-CAM Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        continue

cv2.destroyAllWindows()
