# lab3-app.py
import os
import random
from typing import Optional

from collections import deque

import cv2
import numpy as np
from flask import Flask, render_template, request, Response, redirect, url_for, flash
import mediapipe as mp

app = Flask(__name__)
app.secret_key = "lab3-secret"  # ใช้สำหรับ flash message
os.makedirs("static", exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ตัวตรวจจับสำหรับ "ภาพนิ่ง"
hands_image = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
)
# ตัวตรวจจับสำหรับ "วิดีโอ/สตรีม"
hands_video = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# เก็บผล gesture ล่าสุดของผู้เล่น เพื่อใช้กับปุ่ม Battle
last_player_gesture: Optional[str] = None

VALID_GESTURES = {
    'ROCK', 'PAPER', 'SCISSORS', 'LOVE'
}
# Computer can throw any gesture the player can.
GAME_CHOICES = ['ROCK', 'PAPER', 'SCISSORS', 'LOVE']
# Rules: key beats every gesture in its value set.
WIN_RULES = {
    'ROCK': {'SCISSORS'},
    'PAPER': {'ROCK', 'LOVE'},
    'SCISSORS': {'PAPER'},
    'LOVE': {'ROCK', 'SCISSORS'}
}

gesture_window = deque(maxlen=5)

def majority_vote(labels):
    if not labels:
        return None
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    # คืนค่าที่เจอบ่อยสุด
    return max(counts, key=counts.get)


# ----------------------
# TODO #1: ตัวจำแนกท่ามือ R / P / S
# ใช้ y ของ tip และ pip แต่ละนิ้ว (ค่าพิกัดภาพพิกเซล; แกน y ยิ่งเลื่อนลงยิ่งมีค่ามาก)
def classifier(Index_tip_y, Index_pip_y,
               Middle_tip_y, Middle_pip_y,
               Ring_tip_y, Ring_pip_y,
               Pinky_tip_y, Pinky_pip_y) -> str:
    # margin ป้องกันการสั่นของค่าพิกัด (พิกเซล)
    margin = 10
    index_up = Index_tip_y < Index_pip_y - margin
    middle_up = Middle_tip_y < Middle_pip_y - margin
    ring_up = Ring_tip_y < Ring_pip_y - margin
    pinky_up = Pinky_tip_y < Pinky_pip_y - margin

    # PAPER: ทุกนิ้ว (Index..Pinky) เหยียด
    if index_up and middle_up and ring_up and pinky_up:
        return "PAPER"
    # ROCK: ทุกนิ้ว (Index..Pinky) งอ
    if (not index_up) and (not middle_up) and (not ring_up) and (not pinky_up):
        return "ROCK"
    # SCISSORS: เหยียด 2 นิ้ว (Index, Middle) และงอ (Ring, Pinky)
    if index_up and middle_up and (not ring_up) and (not pinky_up):
        return "SCISSORS"

    return "UNKNOWN"


def classify_from_landmarks(landmarks, image_h, image_w) -> str:
    """ดึงค่าที่ต้องใช้จาก MediaPipe แล้วเรียก classifier
       พร้อมตรวจ gesture 🤟 (LOVE) แบบ optional เพิ่มเติม"""
    # ช่วยลัด
    L = mp_hands.HandLandmark

    def py(lm):  # y (pixel)
        return landmarks.landmark[lm].y * image_h

    # ป้อนเฉพาะนิ้วชี้-ก้อยตามสเปคของ TODO#1
    gesture = classifier(
        py(L.INDEX_FINGER_TIP),  py(L.INDEX_FINGER_PIP),
        py(L.MIDDLE_FINGER_TIP), py(L.MIDDLE_FINGER_PIP),
        py(L.RING_FINGER_TIP),   py(L.RING_FINGER_PIP),
        py(L.PINKY_TIP),         py(L.PINKY_PIP),
    )

    # (Optional) ตรวจ LOVE 🤟 = Thumb+Index+Pinky up, Middle/Ring down
    # ต้องดูนิ้วหัวแม่มือเพิ่ม
    margin = 10
    thumb_up = py(L.THUMB_TIP) < py(L.THUMB_IP) - margin
    index_up = py(L.INDEX_FINGER_TIP)  < py(L.INDEX_FINGER_PIP)  - margin
    middle_up = py(L.MIDDLE_FINGER_TIP) < py(L.MIDDLE_FINGER_PIP) - margin
    ring_up = py(L.RING_FINGER_TIP)   < py(L.RING_FINGER_PIP)   - margin
    pinky_up = py(L.PINKY_TIP)        < py(L.PINKY_PIP)         - margin

    if thumb_up and index_up and pinky_up and (not middle_up) and (not ring_up):
        gesture = "LOVE"

    return gesture


def draw_landmarks_and_label(frame_bgr, results, gesture_text: Optional[str] = None):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
    if gesture_text:
        cv2.putText(frame_bgr, gesture_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
    return frame_bgr


@app.route("/")
def index():
    # แสดงหน้าเว็บหลัก
    return render_template("index.html", last_player_gesture=last_player_gesture)

def detect_image():
    global last_player_gesture
    if "img" not in request.files or request.files["img"].filename == "":
        flash("Please choose an image.")
        return redirect(url_for("index"))

    file = request.files["img"]
    data = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands_image.process(img_rgb)

    gesture = "No hand"
    if results.multi_hand_landmarks:
        h, w, _ = img_bgr.shape
        gesture = classify_from_landmarks(results.multi_hand_landmarks[0], h, w)

    # วาด Landmark + Label และบันทึกภาพ
    out_bgr = img_bgr.copy()
    draw_landmarks_and_label(out_bgr, results, gesture)
    out_path = os.path.join("static", "result.jpg")
    cv2.imwrite(out_path, out_bgr)

    last_player_gesture = gesture if gesture in VALID_GESTURES else None
    return render_template(
        "index.html",
        annotated_url=url_for("static", filename="result.jpg"),
        predicted=gesture,
        last_player_gesture=last_player_gesture
    )


def gen_frames():
    # สตรีมเว็บแคมพร้อม landmark + ทำนายท่าแบบเรียลไทม์
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # กระจก
        h, w, _ = frame.shape

        # MediaPipe ต้องการ RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_video.process(frame_rgb)

        # วาด landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

            # คำนวณ gesture จากมือแรก
            gesture = classify_from_landmarks(results.multi_hand_landmarks[0], h, w)

            # เก็บเข้าหน้าต่าง 5 เฟรม เพื่อให้ผลนิ่งขึ้น
            if gesture in VALID_GESTURES:
                gesture_window.append(gesture)
            else:
                gesture_window.append("UNKNOWN")
        else:
            gesture_window.append("No hand")
            gesture = "No hand"

        # สรุปผลที่ “นิ่ง” จากกรอบโหวต
        stable_gesture = majority_vote(list(gesture_window)) or "No hand"

        # อัปเดตค่าล่าสุดให้ route /battle ใช้ได้ทันที
        global last_player_gesture
        last_player_gesture = stable_gesture if stable_gesture in VALID_GESTURES else None

        # แปะตัวหนังสือบนวิดีโอ
        cv2.putText(frame, f"{stable_gesture}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        # เข้ารหัสส่งเป็น MJPEG
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        jpg = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")



@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot", methods=["POST"])
def snapshot_and_play():
    """กดปุ่ม Capture จากเว็บแคม 1 เฟรม -> จัดคลาส -> เล่นเกมทันที"""
    global last_player_gesture

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        flash("Cannot open webcam.")
        return redirect(url_for("index"))

    # อุ่นกล้อง 5 เฟรม
    for _ in range(5):
        cap.read()

    ok, frame = cap.read()
    cap.release()
    if not ok:
        flash("Failed to capture frame.")
        return redirect(url_for("index"))

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    results = hands_image.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    gesture = "No hand"
    if results.multi_hand_landmarks:
        gesture = classify_from_landmarks(results.multi_hand_landmarks[0], h, w)

    draw_landmarks_and_label(frame, results, gesture)
    cv2.imwrite(os.path.join("static", "snapshot.jpg"), frame)

    last_player_gesture = gesture if gesture in VALID_GESTURES else None
    return redirect(url_for("battle"))


# ----------------------
# TODO #2: ตรรกะเกม R/P/S (คอมสุ่มท่า เทียบผล ชนะ/แพ้/เสมอ)
@app.route("/battle", methods=["GET", "POST"])
def battle():
    # ผู้เล่นอาจส่งมาจากฟอร์ม หรือใช้ last_player_gesture
    if request.method == "POST":
        player_choice = request.form.get("player_choice")
    else:
        player_choice = last_player_gesture

    if player_choice not in VALID_GESTURES:
        # เผื่อกรณีตรวจได้ LOVE/UNKNOWN/No hand
        flash("No valid player gesture (need ROCK/PAPER/SCISSORS/LOVE). Try again.")
        return redirect(url_for("index"))

    computer_choice = random.choice(GAME_CHOICES)

    if player_choice == computer_choice:
        result = "Tie"
    elif computer_choice in WIN_RULES[player_choice]:
        result = "You Win"
    elif player_choice in WIN_RULES[computer_choice]:
        result = "You Lose"
    else:
        result = "Tie"

    return render_template("game.html",
                           computer_choice=computer_choice,
                           player_choice=player_choice,
                           result=result)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)

