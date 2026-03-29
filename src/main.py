from pathlib import Path
from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import Callable

import shutil
import ssl

import urllib.request
import urllib.error

import cv2
import mediapipe as mp

import numpy as np

import random

# Constants/config

USE_LEGACY_SOLUTIONS = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_POSITION = (50, 50)
PREDICTION_HISTORY = deque(maxlen=8)

# Gesture Rule

@dataclass(frozen=True)
class GestureRule:
    name: str
    min_hands: int
    max_hands: int
    priority: int
    matcher: Callable[[list], bool]
    color: tuple = (255, 255, 255)

# Geometry Functions

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

def is_fist(hand):
    return all([
        is_finger_curled(hand, 5, 6, 8, 135),
        is_finger_curled(hand, 9, 10, 12, 135),
        is_finger_curled(hand, 13, 14, 16, 135),
        is_finger_curled(hand, 17, 18, 20, 135),
    ])

def is_open_hand_thumb_in(hand):
    # 4 fingers extended ignore thumb
    fingers_extended = sum([
        is_finger_extended(hand, 5, 6, 8),
        is_finger_extended(hand, 9, 10, 12),
        is_finger_extended(hand, 13, 14, 16),
        is_finger_extended(hand, 17, 18, 20),
    ])

    if fingers_extended < 3: # allow some wiggle room
        return False

    scale = palm_scale(hand)

    # thumb tucked to palm
    thumb_to_palm = distance_3d(hand[4], hand[9]) / scale
    thumb_tucked = thumb_to_palm < 0.55

    return thumb_tucked

# Pair Checks

def is_yuta_pair(hand_a, hand_b):
    scale = (palm_scale(hand_a) + palm_scale(hand_b)) / 2

    a_is_fist = is_fist(hand_a)
    b_is_fist = is_fist(hand_b)

    a_is_open = is_open_hand_thumb_in(hand_a)
    b_is_open = is_open_hand_thumb_in(hand_b)

    # order dont matter
    if not (
        (a_is_fist and b_is_open) or
        (b_is_fist and a_is_open)
    ):
        return False

    # hands should be near each other
    wrist_dist = distance_3d(hand_a[0], hand_b[0]) / scale
    if wrist_dist > 1.8:
        return False

    return True

def is_hakari_pair(hand_a, hand_b):
    # determines which hand is above
    if hand_a[0].y < hand_b[0].y:
        upper, lower = hand_a, hand_b
    else:
        upper, lower = hand_b, hand_a

    scale = (palm_scale(upper) + palm_scale(lower)) / 2

    # upper hand
    tip_dist = distance_3d(upper[4], upper[8]) / scale
    is_circle = tip_dist < 0.35
    middle_straight = is_finger_extended(upper, 9, 10, 12)
    ring_straight   = is_finger_extended(upper, 13, 14, 16)
    pinky_straight  = is_finger_extended(upper, 17, 18, 20)
    
    upper_ok = is_circle and middle_straight and ring_straight and pinky_straight

    # lower hand (flat one)
    lower_fingers_straight = all([
        is_finger_extended(lower, 5, 6, 8),
        is_finger_extended(lower, 9, 10, 12),
        is_finger_extended(lower, 13, 14, 16),
        is_finger_extended(lower, 17, 18, 20)
    ])
    
    # upper hand must be above and close to lower hand
    vertical_gap = (lower[9].y - upper[0].y) # dist from lower middle mcp to upper wrist
    hands_close = 0.1 < vertical_gap < 1.0

    return upper_ok and lower_fingers_straight and hands_close

def is_yuji_pair(hand_a, hand_b):
    scale = (palm_scale(hand_a) + palm_scale(hand_b)) / 2

    index_a = is_finger_extended(hand_a, 5, 6, 8)
    index_b = is_finger_extended(hand_b, 5, 6, 8)

    if not (index_a and index_b):
        return False

    # index fingers are up
    def is_vertical(hand):
        tip = hand[8]
        mcp = hand[5]
        dx = abs(tip.x - mcp.x)
        dy = abs(tip.y - mcp.y)
        return dy > dx * 1.8  # can adjus this for leniency

    if not (is_vertical(hand_a) and is_vertical(hand_b)):
        return False

    # index fingertips dist
    index_dist = distance_3d(hand_a[8], hand_b[8]) / scale
    if index_dist > 0.4:
        return False

    # hand dist
    wrist_dist = distance_3d(hand_a[0], hand_b[0]) / scale
    if not (0.3 <= wrist_dist <= 1.5):
        return False

    # fingers pointing in same direction
    vec_a = hand_a[8].y - hand_a[5].y
    vec_b = hand_b[8].y - hand_b[5].y

    if vec_a * vec_b < 0:
        return False

    return True

def is_mahito_pair(hand_a, hand_b):
    scale_a = palm_scale(hand_a)
    scale_b = palm_scale(hand_b)
    scale = (scale_a + scale_b) / 2

    # pinkies should touch
    pinky_dist = distance_3d(hand_a[20], hand_b[20]) / scale
    pinkies_touch = pinky_dist <= 0.9

    # thumbs overlap or are very close
    thumb_dist = distance_3d(hand_a[4], hand_b[4]) / scale
    thumbs_close = thumb_dist <= 1.0

    # palms shouldnt touch, forming egg shape
    palm_center_a = hand_center(hand_a)
    palm_center_b = hand_center(hand_b)

    palm_gap = normalized_point_distance(
        (palm_center_a[0], palm_center_a[1]),
        (palm_center_b[0], palm_center_b[1]),
        scale,
    )

    palms_separated = 0.6 <= palm_gap <= 2.2

    # wrists still near each other
    wrist_dist = distance_3d(hand_a[0], hand_b[0]) / scale
    wrists_reasonable = 0.4 <= wrist_dist <= 3.5

    return (
        pinkies_touch
        and thumbs_close
        and palms_separated
        and wrists_reasonable
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
    middles_touch = middle_dist <= .75
    rings_touch   = ring_dist   <= .75
    pinkies_touch = pinky_dist  <= .75

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

    touch_count = sum([middles_touch, rings_touch, pinkies_touch])
    return (
        touch_count >= 2 and
        triangle_shape and
        wrists_reasonable
    )

# Match Checks

def match_authentic_mutual_love(hands_landmarks):
    if len(hands_landmarks) != 2:
        return False
    return is_yuta_pair(hands_landmarks[0], hands_landmarks[1])

def match_idle_death_gamble(hands_landmarks):
    if len(hands_landmarks) != 2:
        return False
    return is_hakari_pair(hands_landmarks[0], hands_landmarks[1])

def match_yuji_itadori(hands_landmarks):
    if len(hands_landmarks) < 2:
        return False

    for i in range(len(hands_landmarks) - 1):
        for j in range(i + 1, len(hands_landmarks)):
            if is_yuji_pair(hands_landmarks[i], hands_landmarks[j]):
                return True
    return False

def match_self_embodiment_of_perfection(hands_landmarks):
    if len(hands_landmarks) < 2:
        return False

    for first_index in range(len(hands_landmarks) - 1):
        for second_index in range(first_index + 1, len(hands_landmarks)):
            if is_mahito_pair(hands_landmarks[first_index], hands_landmarks[second_index]):
                return True
    return False

def match_malevolent_shrine(hands_landmarks):
    if len(hands_landmarks) < 2:
        return False

    for first_index in range(len(hands_landmarks) - 1):
        for second_index in range(first_index + 1, len(hands_landmarks)):
            if is_sukuna_pair(hands_landmarks[first_index], hands_landmarks[second_index]):
                return True

    return False

def match_unlimited_void(hands_landmarks):
    if len(hands_landmarks) != 1:
        return False
    return is_gojo_hand(hands_landmarks[0])

# Individual Hand Checks

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

# All Domains

GESTURE_RULES = sorted(
    [
        GestureRule(
            name="Authentic Mutual Love",
            min_hands=2,
            max_hands=2,
            priority=130,
            color=(180, 100, 255),
            matcher=match_authentic_mutual_love,
        ),
        GestureRule(
            name="Idle Death Gamble",
            min_hands=2,
            max_hands=2,
            priority=120,
            color=(0, 200, 255),
            matcher=match_idle_death_gamble,
        ),
        GestureRule(
            name="Malevolent Shrine",
            min_hands=2,
            max_hands=2,
            priority=110,
            color=(0, 0, 255),
            matcher=match_malevolent_shrine,
        ),
        GestureRule(
            name="Yuji Itadori",
            min_hands=2,
            max_hands=2,
            priority=100,
            color=(0, 255, 0),
            matcher=match_yuji_itadori,
        ),
        GestureRule(
            name="Self-Embodiment of Perfection",
            min_hands=2,
            max_hands=2,
            priority=90,
            color=(255, 0, 255),
            matcher=match_self_embodiment_of_perfection,
        ),
        GestureRule(
            name="Unlimited Void",
            min_hands=1,
            max_hands=1,
            priority=80,
            color=(255, 255, 255),
            matcher=match_unlimited_void,
        ),
    ],
    key=lambda rule: rule.priority,
    reverse=True,
)

GESTURE_RULES_BY_NAME = {rule.name: rule for rule in GESTURE_RULES}

# Domain Detection and Smoothing

def is_label_allowed_for_hand_count(label, hand_count):
    if not label:
        return False
    rule = GESTURE_RULES_BY_NAME.get(label)
    if not rule:
        return False
    return rule.min_hands <= hand_count <= rule.max_hands

def smooth_domain_prediction(domain, hand_count):
    if not is_label_allowed_for_hand_count(domain, hand_count):
        domain = None

    PREDICTION_HISTORY.append(domain if domain else "")
    votes = Counter(label for label in PREDICTION_HISTORY if label)
    if not votes:
        return None

    top_label, top_count = votes.most_common(1)[0]
    if not is_label_allowed_for_hand_count(top_label, hand_count):
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

    hand_count = len(hands_landmarks)
    for rule in GESTURE_RULES:
        if rule.min_hands <= hand_count <= rule.max_hands and rule.matcher(hands_landmarks):
            return rule.name

    return None

# Visual Effects

# Gojo
STARS = []
SYMBOLS = []

def init_stars(width, height, count=150):
    global STARS, SYMBOLS
    STARS = [[random.randint(0, width), random.randint(0, height), random.uniform(0.5, 3.0)] for _ in range(count)]
    # Create scrolling symbols (numbers/chars)
    for _ in range(30):
        SYMBOLS.append([random.randint(0, width), random.randint(0, height), random.uniform(2, 6), str(random.randint(0, 9))])

def apply_infinite_void(frame):
    h, w = frame.shape[:2]
    
    # dizzy effect
    shift = 4
    b = frame[:, :, 0]
    g = np.roll(frame[:, :, 1], shift, axis=1)
    r = np.roll(frame[:, :, 2], -shift, axis=1)
    frame = cv2.merge([b, g, r])

    # darken screen
    overlay = cv2.GaussianBlur(frame, (15, 15), 0)
    frame = cv2.addWeighted(frame, 0.2, overlay, 0.8, 0)

    for star in STARS:
        star[1] = (star[1] + star[2]) % h
        cv2.circle(frame, (int(star[0]), int(star[1])), 1, (255, 255, 255), -1)
    
    for sym in SYMBOLS:
        sym[1] = (sym[1] + sym[2]) % h
        cv2.putText(frame, sym[3], (int(sym[0]), int(sym[1])), FONT, 0.5, (255, 255, 255), 1)

    return frame

# Sukuna

SLASHES = []
FLASH_COUNTER = 0

def spawn_slash(width, height):
    x1 = random.randint(0, width)
    y1 = random.randint(0, height)

    length = random.randint(80, 200)
    angle = random.uniform(-0.8, 0.8) # diagonal

    x2 = int(x1 + length * math.cos(angle))
    y2 = int(y1 + length * math.sin(angle))

    life = random.randint(3, 6)

    SLASHES.append([x1, y1, x2, y2, life])

def apply_malevolent_shrine(frame):
    global FLASH_COUNTER
    h, w = frame.shape[:2]

    # red tint
    red_tint = np.zeros_like(frame)
    red_tint[:, :, 2] = 120
    frame = cv2.addWeighted(frame, 0.7, red_tint, 0.3, 0)

    # flashing
    FLASH_COUNTER += 1
    if FLASH_COUNTER % 10 == 0:
        return np.ones_like(frame) * 255

    # grid cut
    if random.random() < 0.3:
        pos = random.randint(0, h if random.random() > 0.5 else w)
        if random.random() > 0.5:
            cv2.line(frame, (0, pos), (w, pos), (255, 255, 255), 1)
        else:
            cv2.line(frame, (pos, 0), (pos, h), (255, 255, 255), 1)

    # rand slahes
    if random.random() < 0.6:
        spawn_slash(w, h)
    
    new_slashes = []
    for x1, y1, x2, y2, life in SLASHES:
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), life)
        if life - 1 > 0: new_slashes.append([x1, y1, x2, y2, life - 1])
    SLASHES[:] = new_slashes

    # screen shake
    M = np.float32([[1, 0, random.randint(-15, 15)], [0, 1, random.randint(-15, 15)]])
    frame = cv2.warpAffine(frame, M, (w, h))

    return frame

# Main Loop

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

ret, test_frame = cap.read()
if ret:
    init_stars(test_frame.shape[1], test_frame.shape[0])

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

        if stable_domain == "Malevolent Shrine":
            frame = apply_malevolent_shrine(frame)
        elif stable_domain == "Unlimited Void":
            frame = apply_infinite_void(frame)

        if stable_domain:
            cv2.putText(
                frame,
                f"Domain Expansion: {stable_domain}",
                TEXT_POSITION,
                FONT,
                1,
                GESTURE_RULES_BY_NAME[stable_domain].color,
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