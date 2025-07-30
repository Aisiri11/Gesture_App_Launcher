import cv2
import mediapipe as mp
import time
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Webcam (with Windows-specific flag)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Failed to open webcam.")
    exit()

print("✅ Webcam opened successfully.")

# Drawing points
draw_points = []
drawing = False

while True:
    success, frame = cap.read()
    if not success:
        print("❌ Failed to read frame.")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    h, w, _ = frame.shape
    cx, cy = 0, 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Index fingertip
            index_tip = hand_landmarks.landmark[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)

            # Draw green dot at fingertip
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            # If index finger is above middle finger → drawing mode
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
                drawing = True
                draw_points.append((cx, cy))
            else:
                drawing = False

    # Draw white trail
    for i in range(1, len(draw_points)):
        cv2.line(frame, draw_points[i - 1], draw_points[i], (255, 255, 255), 2)

    cv2.imshow("Air Draw", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
