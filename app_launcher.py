import cv2
import pytesseract
import mediapipe as mp
import os
import time

# Setup for Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize webcam
cap = cv2.VideoCapture(0)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Define app buttons
apps = {
    "Notepad": ((50, 50), (200, 120)),
    "Calculator": ((250, 50), (400, 120))
}

while True:
    success, frame = cap.read()
    if not success:
        print("❌ Failed to access webcam.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Draw app buttons
    for app_name, ((x1, y1), (x2, y2)) in apps.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, app_name, (x1 + 10, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)

            # Draw a circle at index fingertip
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            # Check for intersection with app buttons
            for app_name, ((x1, y1), (x2, y2)) in apps.items():
                if x1 < cx < x2 and y1 < cy < y2:
                    cv2.putText(frame, f"Launching {app_name}", (10, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if app_name == "Notepad":
                        os.system("start notepad")
                    elif app_name == "Calculator":
                        os.system("start calc")
                    time.sleep(1.5)
                    break

    # ✅ Show the frame
    cv2.imshow("Webcam", frame)

    # Exit on Esc key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
