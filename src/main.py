from pathlib import Path
import shutil
import ssl
import urllib.request
import urllib.error

import cv2
import mediapipe as mp

USE_LEGACY_SOLUTIONS = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

def finger_up(landmarks, tip, pip):
    return landmarks[tip].y < landmarks[pip].y

def ensure_task_model(model_path: Path):
    if model_path.exists():
        return

    try:
        import certifi

        ssl_context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ssl_context = ssl.create_default_context()

    try:
        with urllib.request.urlopen(HAND_MODEL_URL, context=ssl_context, timeout=30) as response:
            with open(model_path, "wb") as model_file:
                shutil.copyfileobj(response, model_file)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not download MediaPipe model due to SSL/network issue. "
            f"Download this file manually and place it at {model_path}: {HAND_MODEL_URL}"
        ) from exc

def draw_task_landmarks(frame, hand_landmarks, connections):
    h, w = frame.shape[:2]
    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    for conn in connections:
        start = hand_landmarks[conn.start]
        end = hand_landmarks[conn.end]
        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

if USE_LEGACY_SOLUTIONS:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
    )
else:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    ensure_task_model(HAND_MODEL_PATH)

    base_options = mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(options)
    hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0).")

try:
    frame_index = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if USE_LEGACY_SOLUTIONS:
            results = hands.process(rgb)
            detected_hands = []
            if results.multi_hand_landmarks:
                detected_hands = [hl.landmark for hl in results.multi_hand_landmarks]
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            frame_index += 1
            timestamp_ms = int(frame_index * (1000 / 30))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            detected_hands = results.hand_landmarks if results.hand_landmarks else []
            for hand_landmarks in detected_hands:
                draw_task_landmarks(frame, hand_landmarks, hand_connections)

        for lm in detected_hands:
            index_up = finger_up(lm, 8, 6)
            middle_up = finger_up(lm, 12, 10)
            ring_up = finger_up(lm, 16, 14)
            pinky_up = finger_up(lm, 20, 18)

            # gojo
            if index_up and middle_up and not ring_up and not pinky_up:
                cv2.putText(
                    frame,
                    "Domain Expansion: Unlimited Void",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 255),
                    3,
                )
                break

        cv2.imshow("Domain Expansion Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    if USE_LEGACY_SOLUTIONS:
        hands.close()
    else:
        hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()