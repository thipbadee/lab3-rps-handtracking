# hand_image_demo.py
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

img_bgr = cv2.imread('rock-scissor-paper.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

results = hands.process(img_rgb)

if not results.multi_hand_landmarks:
    print("No hand detected.")
else:
    image_height, image_width, _ = img_rgb.shape
    print('-----------------')
    for hand_landmarks in results.multi_hand_landmarks:
        tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        print(f'Index finger tip coordinates: ({tip.x * image_width}, {tip.y * image_height})')
        mp_draw.draw_landmarks(
            img_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )
    print('-----------------')

plt.imshow(img_rgb)
plt.axis('off')
plt.show()
