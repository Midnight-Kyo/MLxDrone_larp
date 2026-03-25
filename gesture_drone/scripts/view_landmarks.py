r"""
Visualize MediaPipe's 21 hand landmarks in real-time.

This shows ONLY the hand detection model -- no gesture classification.
Each finger is color-coded, fingertips get larger dots, and every
landmark is numbered 0-20.

Run from Windows PowerShell:
    python "\\wsl$\Ubuntu-22.04\home\kyo\Projects\MLxDrone_larp\gesture_drone\scripts\view_landmarks.py" 2

Controls:
    Q  - Quit
"""

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

SCRIPT_DIR = Path(__file__).resolve().parent
HAND_LANDMARK_PATH = SCRIPT_DIR.parent / "models" / "hand_landmarker.task"

# Skeleton connections between the 21 landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # ring finger
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm cross-connections
]

LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

FINGER_COLORS = {
    "wrist":  (255, 255, 255),
    "thumb":  (0, 255, 255),
    "index":  (0, 255, 0),
    "middle": (255, 255, 0),
    "ring":   (255, 0, 255),
    "pinky":  (0, 165, 255),
    "palm":   (200, 200, 200),
}


def get_finger(idx):
    if idx == 0:
        return "wrist"
    elif idx <= 4:
        return "thumb"
    elif idx <= 8:
        return "index"
    elif idx <= 12:
        return "middle"
    elif idx <= 16:
        return "ring"
    else:
        return "pinky"


def connection_color(i, j):
    if i == 0 and j in (5, 9, 13, 17):
        return FINGER_COLORS["palm"]
    if min(i, j) >= 5 and max(i, j) <= 17 and abs(i - j) == 4:
        return FINGER_COLORS["palm"]
    return FINGER_COLORS[get_finger(max(i, j))]


def create_hand_detector():
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_LANDMARK_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def find_camera():
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            print(f"Using camera index {cam_index} (MSMF)")
            return cap
        cap.release()
    cap = cv2.VideoCapture(cam_index)
    if cap.isOpened():
        print(f"Using camera index {cam_index}")
        return cap
    print(f"ERROR: Camera index {cam_index} could not be opened.")
    return None


def draw_landmarks(frame, all_hand_landmarks):
    h, w = frame.shape[:2]

    for hand_landmarks in all_hand_landmarks:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        for i, j in HAND_CONNECTIONS:
            color = connection_color(i, j)
            cv2.line(frame, points[i], points[j], color, 3)

        for idx, (x, y) in enumerate(points):
            finger = get_finger(idx)
            color = FINGER_COLORS[finger]
            is_tip = idx in (4, 8, 12, 16, 20)
            radius = 8 if is_tip else 4

            cv2.circle(frame, (x, y), radius + 2, (0, 0, 0), -1)
            cv2.circle(frame, (x, y), radius, color, -1)

            cv2.putText(frame, str(idx), (x + 10, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


def draw_hud(frame, num_hands, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "MediaPipe Hand Landmarks", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    status = f"{num_hands} hand{'s' if num_hands != 1 else ''} detected" if num_hands > 0 else "No hands detected"
    color = (0, 220, 0) if num_hands > 0 else (100, 100, 100)
    cv2.putText(frame, status, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(frame, "[Q] quit", (w - 130, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

    # Legend at bottom
    legend_y = h - 15
    items = [("Thumb", "thumb"), ("Index", "index"), ("Middle", "middle"),
             ("Ring", "ring"), ("Pinky", "pinky")]
    x_pos = 20
    for name, key in items:
        color = FINGER_COLORS[key]
        cv2.circle(frame, (x_pos, legend_y - 5), 6, color, -1)
        cv2.putText(frame, name, (x_pos + 12, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        x_pos += 100


def main():
    print("Loading MediaPipe HandLandmarker...")
    detector = create_hand_detector()
    print("Ready.")

    cap = find_camera()
    if cap is None:
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0

            if result.hand_landmarks:
                draw_landmarks(frame, result.hand_landmarks)

            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_time = now

            draw_hud(frame, num_hands, fps)
            cv2.imshow("Hand Landmarks", frame)

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q")):
                break

    except KeyboardInterrupt:
        pass
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
