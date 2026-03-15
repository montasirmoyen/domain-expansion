from pathlib import Path
from collections import Counter, deque
import math

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

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_POSITION = (50, 50)
PREDICTION_HISTORY = deque(maxlen=8)

def distance_3d(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

def angle_3d(a, b, c):
    bax, bay, baz = a.x - b.x, a.y - b.y, a.z - b.z
    bcx, bcy, bcz = c.x - b.x, c.y - b.y, c.z - b.z

    dot = bax * bcx + bay * bcy + baz * bcz
    norm_ba = math.sqrt(bax * bax + bay * bay + baz * baz)
    norm_bc = math.sqrt(bcx * bcx + bcy * bcy + bcz * bcz)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine = max(-1.0, min(1.0, dot / (norm_ba * norm_bc)))
    return math.degrees(math.acos(cosine))

def is_finger_extended(landmarks, mcp, pip, tip, threshold_deg=155.0):
    return angle_3d(landmarks[mcp], landmarks[pip], landmarks[tip]) >= threshold_deg

def is_finger_curled(landmarks, mcp, pip, tip, threshold_deg=130.0):
    return angle_3d(landmarks[mcp], landmarks[pip], landmarks[tip]) <= threshold_deg

def finger_angle(landmarks, mcp, pip, tip):
    return angle_3d(landmarks[mcp], landmarks[pip], landmarks[tip])

def palm_scale(landmarks):
    return max(distance_3d(landmarks[0], landmarks[9]), 1e-6)

def hand_center(landmarks):
    scale = palm_scale(landmarks)
    center_x = sum(landmarks[i].x for i in (0, 5, 9, 13, 17)) / 5
    center_y = sum(landmarks[i].y for i in (0, 5, 9, 13, 17)) / 5
    return center_x, center_y, scale

def fingertip_cluster_center(landmarks, tip_indices):
    center_x = sum(landmarks[i].x for i in tip_indices) / len(tip_indices)
    center_y = sum(landmarks[i].y for i in tip_indices) / len(tip_indices)
    return center_x, center_y

def normalized_point_distance(point_a, point_b, scale):
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) / scale

def is_sukuna_candidate_hand(landmarks):
    scale = palm_scale(landmarks)
    index_angle = finger_angle(landmarks, 5, 6, 8)
    middle_angle = finger_angle(landmarks, 9, 10, 12)
    ring_angle = finger_angle(landmarks, 13, 14, 16)
    pinky_angle = finger_angle(landmarks, 17, 18, 20)

    middle_ring_tips_close = distance_3d(landmarks[12], landmarks[16]) / scale <= 1.2

    return (
        middle_angle >= 145.0
        and ring_angle >= 145.0
        and index_angle <= 150.0
        and pinky_angle <= 150.0
        and middle_ring_tips_close
    )

def is_sukuna_pair(hand_a, hand_b):
    scale_a = palm_scale(hand_a)
    scale_b = palm_scale(hand_b)
    scale = (scale_a + scale_b) / 2

    # calc finger distances
    middle_dist = distance_3d(hand_a[12], hand_b[12]) / scale
    ring_dist   = distance_3d(hand_a[16], hand_b[16]) / scale
    pinky_dist  = distance_3d(hand_a[20], hand_b[20]) / scale

    # should tweak these nums later
    middles_touch = middle_dist <= 1.1
    rings_touch   = ring_dist   <= 1.1
    pinkies_touch = pinky_dist  <= 1.1

    # check the triangle shape
    mid_ring_left  = distance_3d(hand_a[12], hand_a[16]) / scale
    mid_ring_right = distance_3d(hand_b[12], hand_b[16]) / scale
    triangle_shape = (
        0.25 <= mid_ring_left <= 1.8 and
        0.25 <= mid_ring_right <= 1.8
    )

    # make sure wrists are resonably positioned
    wrist_dist = distance_3d(hand_a[0], hand_b[0]) / scale
    wrists_reasonable = 0.4 <= wrist_dist <= 4.2

    print(middle_dist, ring_dist, pinky_dist)

    touch_count = sum([middles_touch, rings_touch, pinkies_touch])
    return (
        touch_count >= 2 and
        triangle_shape and
        wrists_reasonable
    )

def is_mahito_pair(hand_a, hand_b):
    pass

def is_sukuna_hand(landmarks):
    index_curled = is_finger_curled(landmarks, 5, 6, 8)
    middle_extended = is_finger_extended(landmarks, 9, 10, 12)
    ring_extended = is_finger_extended(landmarks, 13, 14, 16)
    pinky_curled = is_finger_curled(landmarks, 17, 18, 20)

    scale = palm_scale(landmarks)
    middle_ring_tips_close = distance_3d(landmarks[12], landmarks[16]) / scale <= 0.55

    return (
        index_curled
        and middle_extended
        and ring_extended
        and pinky_curled
        and middle_ring_tips_close
    )

def is_gojo_hand(landmarks):
    index_extended = is_finger_extended(landmarks, 5, 6, 8)
    middle_extended = is_finger_extended(landmarks, 9, 10, 12)
    ring_curled = is_finger_curled(landmarks, 13, 14, 16)
    pinky_curled = is_finger_curled(landmarks, 17, 18, 20)

    scale = palm_scale(landmarks)
    index_middle_tips_close = distance_3d(landmarks[8], landmarks[12]) / scale <= 0.65

    return (
        index_extended
        and middle_extended
        and ring_curled
        and pinky_curled
        and index_middle_tips_close
    )

def smooth_domain_prediction(domain, hand_count):
    if hand_count < 2 and domain == "Malevolent Shrine":
        domain = None

    if hand_count >= 2 and domain == "Unlimited Void":
        domain = None

    PREDICTION_HISTORY.append(domain if domain else "")
    votes = Counter(label for label in PREDICTION_HISTORY if label)
    if not votes:
        return None

    top_label, top_count = votes.most_common(1)[0]
    if hand_count < 2 and top_label == "Malevolent Shrine":
        return None

    if hand_count >= 2 and top_label == "Unlimited Void":
        return None

    if top_count >= 4 and top_count >= (len(PREDICTION_HISTORY) // 2 + 1):
        return top_label
    return None

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

def detect_domain_expansion(hands_landmarks):
    if not hands_landmarks:
        return None

    if len(hands_landmarks) >= 2:
        for first_index in range(len(hands_landmarks) - 1):
            for second_index in range(first_index + 1, len(hands_landmarks)):
                if is_sukuna_pair(hands_landmarks[first_index], hands_landmarks[second_index]):
                    return "Malevolent Shrine"

        sukuna_matches = sum(1 for lm in hands_landmarks if is_sukuna_hand(lm))
        if sukuna_matches > 0:
            return "Malevolent Shrine"

        return None

    gojo_matches = sum(1 for lm in hands_landmarks if is_gojo_hand(lm))

    if gojo_matches > 0:
        return "Unlimited Void"
    return None

if USE_LEGACY_SOLUTIONS:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
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
        num_hands=2,
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
            timestamp_ms = int(frame_index * 33)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            detected_hands = results.hand_landmarks if results.hand_landmarks else []
            for hand_landmarks in detected_hands:
                draw_task_landmarks(frame, hand_landmarks, hand_connections)

        frame_domain = detect_domain_expansion(detected_hands)
        stable_domain = smooth_domain_prediction(frame_domain, len(detected_hands))

        if stable_domain:
            cv2.putText(
                frame,
                f"Domain Expansion: {stable_domain}",
                TEXT_POSITION,
                FONT,
                1,
                (255, 0, 255),
                3,
            )

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