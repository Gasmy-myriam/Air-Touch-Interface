import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# =========================
# CONFIGURATION
# =========================
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
SMOOTHING = 5
CLICK_THRESHOLD = 30  # distance for pinch click

# =========================
# INIT
# =========================
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

clicking = False

# =========================
# MAIN LOOP
# =========================
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:

            # Get landmarks
            landmarks = handLms.landmark

            # Index finger tip (8)
            x1 = int(landmarks[8].x * w)
            y1 = int(landmarks[8].y * h)

            # Thumb tip (4)
            x2 = int(landmarks[4].x * w)
            y2 = int(landmarks[4].y * h)

            # =========================
            # CURSOR MOVEMENT
            # =========================
            screen_x = np.interp(x1, (0, w), (0, SCREEN_WIDTH))
            screen_y = np.interp(y1, (0, h), (0, SCREEN_HEIGHT))

            curr_x = prev_x + (screen_x - prev_x) / SMOOTHING
            curr_y = prev_y + (screen_y - prev_y) / SMOOTHING

            pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

            # =========================
            # CLICK DETECTION (PINCH)
            # =========================
            distance = np.hypot(x2 - x1, y2 - y1)

            if distance < CLICK_THRESHOLD:
                if not clicking:
                    pyautogui.click()
                    clicking = True
                    time.sleep(0.2)
            else:
                clicking = False

            # =========================
            # DRAW VISUALS
            # =========================
            cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)

            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # =========================
    # DISPLAY
    # =========================
    cv2.putText(img, "Air Touch Interface", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Air Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()