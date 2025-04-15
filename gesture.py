import cv2
import pyautogui
import time
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Configurations
SCROLL_SPEED = 300
SCROLL_DELAY = 1  # seconds
CAM_WIDTH, CAM_HEIGHT = 640, 480

def detect_gesture(landmarks, handedness):
    fingers = []
    tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    
    # Check fingers (except thumb)
    for tip in tips:
        if landmarks.landmark[tip].y < landmarks.landmark[tip - 2].y:
            fingers.append(1)
    
    # Check if the thumb is extended
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    if abs(thumb_tip.x - thumb_mcp.x) > 0.1:  # Threshold to determine if thumb is extended
        fingers.append(1)
    
    if sum(fingers) == 5:
        return "scroll_up"
    elif sum(fingers) == 0:
        return "scroll_down"
    return "none"

# Initialize Camera
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)
last_scroll = p_time = time.time()

print("Gesture Scroll Control Active\nOpen palm: Scroll Up\nFist: Scroll Down\nPress 'q' to exit")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.flip(img, 1)  # Flip horizontally to make movement intuitive
    results = hands.process(img)
    
    gesture, handedness = "none", "Unknown"

    if results.multi_hand_landmarks:
        for hand, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            handedness = handedness_info.classification[0].label
            gesture = detect_gesture(hand, handedness)
            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            
            # Scroll if enough time has passed since the last action
            if (time.time() - last_scroll) > SCROLL_DELAY:
                if gesture == "scroll_up":
                    pyautogui.scroll(SCROLL_SPEED)
                elif gesture == "scroll_down":
                    pyautogui.scroll(-SCROLL_SPEED)
                last_scroll = time.time()
    
    # Compute FPS
    fps = int(1 / (time.time() - p_time)) if (time.time() - p_time) > 0 else 0
    p_time = time.time()
    
    # Display FPS & gesture info
    cv2.putText(img, f"FPS: {fps} | Hand: {handedness} | Gesture: {gesture}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.imshow("Gesture Control", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
