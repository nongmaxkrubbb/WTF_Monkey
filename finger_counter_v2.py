"""
ระบบตรวจจับ "ท่ามือ" เฉพาะจากกล้อง แล้วโชว์ภาพตามท่าที่ทำ
============================================================
ต่างจากการนับจำนวนนิ้วตรงที่ระบบนี้ดูว่า "นิ้วไหนชูบ้าง" ไม่ใช่แค่ "กี่นิ้ว"
เช่น ชี้นิ้วเดียว (นิ้วชี้) กับ ยกโป้งเดียว (thumbs up) ต่างก็เป็น "1 นิ้ว"
เหมือนกัน แต่เป็นคนละท่า และโชว์คนละภาพได้

วิธีติดตั้ง:
    pip install -r requirements.txt

วิธีใช้:
    python finger_counter.py
    (ครั้งแรกจะดาวน์โหลดโมเดล hand_landmarker.task อัตโนมัติ ~10MB ต้องต่อเน็ต)

ควบคุม:
    q = ออกโปรแกรม
    s = บันทึกภาพหน้าจอปัจจุบัน (screenshot) ลงโฟลเดอร์ captures/
    +/- = ปรับความไวในการตรวจว่านิ้วชูหรือไม่

การใส่ภาพของคุณเอง:
    ใส่ไฟล์ชื่อตามท่า เช่น fist.gif, point.png, peace.jpg ไว้ในโฟลเดอร์ images/
    (ดูรายชื่อท่าทั้งหมดที่รองรับได้ในโฟลเดอร์ images/README.txt หรือ list GESTURES ด้านล่าง)
    รองรับไฟล์ .gif (เล่นเป็นภาพเคลื่อนไหว), .png, .jpg, .jpeg, .webp
    ถ้าไม่มีไฟล์สำหรับท่าไหน โปรแกรมจะโชว์ placeholder ชื่อท่านั้นแทน
"""

import os
import ssl
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from PIL import Image, ImageSequence

# ----------------------------------------------------------------------------
# ตั้งค่า
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

MAX_HANDS = 1                      # ตรวจจับ 1 มือ (โฟกัสท่าเฉพาะ ชัดเจนกว่า)
DETECTION_CONFIDENCE = 0.6
PRESENCE_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.6
CAM_INDEX = 0
PANEL_WIDTH = 360
FINGER_MARGIN = 1.05   # ค่ายิ่งสูง = ต้องเหยียดนิ้วชัดเจนมากขึ้นถึงจะนับว่า "ชู"
                        # ถ้ายังจับท่าไม่ตรง ลองปรับด้วยปุ่ม +/- ตอนรันโปรแกรม

FINGER_TIP_IDS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
FINGER_PIP_IDS = [3, 6, 10, 14, 18]

# ----------------------------------------------------------------------------
# รายชื่อ "ท่ามือ" ที่รองรับ: (slug สำหรับตั้งชื่อไฟล์ภาพ, คำอธิบาย, pattern นิ้ว)
# pattern เรียงตาม (โป้ง, ชี้, กลาง, นาง, ก้อย) — 1 = ชู, 0 = งอ
# แก้ไข/เพิ่มท่าของคุณเองได้ตรงนี้เลย
# ----------------------------------------------------------------------------
GESTURES = [
    ("monkey_think",  "ท่าครุ่นคิด (ชู 1 นิ้วชี้)",        (0, 1, 0, 0, 0)),
    ("monkey_shock",  "ท่ากุมหัว (กาง 5 นิ้ว)",          (1, 1, 1, 1, 1)),
    ("monkey_beer",   "ท่าถือแก้ว (กำหมัด)",             (0, 0, 0, 0, 0)),
    ("monkey_tongue", "ท่าแลบลิ้น (ชู 2 นิ้ว V)",        (0, 1, 1, 0, 0)),
    ("monkey_music",  "ท่าฟังเพลง (ชี้+ก้อย ชาวร็อค)",    (0, 1, 0, 0, 1)),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
]


def _make_ssl_context():
    """สร้าง SSL context โดยพยายามใช้ certificate bundle จาก certifi ก่อน
    (แก้ปัญหา SSL: CERTIFICATE_VERIFY_FAILED ที่พบบ่อยบน macOS)"""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def ensure_model():
    """ดาวน์โหลดไฟล์โมเดล hand_landmarker.task ถ้ายังไม่มี"""
    if os.path.exists(MODEL_PATH):
        return
    print("กำลังดาวน์โหลดโมเดล hand landmark (ครั้งแรกครั้งเดียว)...")
    try:
        ctx = _make_ssl_context()
        with urllib.request.urlopen(MODEL_URL, context=ctx) as resp, \
                open(MODEL_PATH, "wb") as out_file:
            out_file.write(resp.read())
        print("ดาวน์โหลดเสร็จแล้ว:", MODEL_PATH)
    except Exception as e:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        print("ดาวน์โหลดโมเดลไม่สำเร็จ:", e)
        print()
        print("วิธีแก้ (macOS + Python จาก python.org มักเจอปัญหา SSL certificate):")
        print('  1) รัน: open "/Applications/Python 3.x/Install Certificates.command"')
        print("  2) หรือติดตั้ง certifi แล้วรันใหม่: pip install certifi")
        print("  3) หรือดาวน์โหลดไฟล์เองจากลิงก์นี้แล้ววางไว้ในโฟลเดอร์เดียวกับสคริปต์")
        print(f"     ชื่อไฟล์ hand_landmarker.task : {MODEL_URL}")
        raise SystemExit(1)


# ----------------------------------------------------------------------------
# ตรวจจับว่านิ้วไหนชูขึ้นบ้าง จาก landmark ของมือหนึ่งข้าง
# ----------------------------------------------------------------------------
def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def get_finger_states(landmarks, margin=1.05):
    """landmarks: list ของจุด (x, y) แบบ normalized (0-1), 21 จุดตามลำดับ mediapipe

    ใช้วิธี "ระยะห่างจากข้อมือ (wrist)" แทนการเทียบแกน y ตรงๆ เพราะทนทานต่อ
    การเอียง/หมุนมือได้ดีกว่ามาก

    คืนค่า tuple สถานะ 5 นิ้ว (โป้ง, ชี้, กลาง, นาง, ก้อย) แต่ละค่าเป็น 1/0
    """
    wrist = landmarks[0]
    states = []

    for tip_id, pip_id in zip(FINGER_TIP_IDS[1:], FINGER_PIP_IDS[1:]):
        tip_dist = _dist(landmarks[tip_id], wrist)
        pip_dist = _dist(landmarks[pip_id], wrist)
        states.append(1 if tip_dist > pip_dist * margin else 0)

    pinky_mcp = landmarks[17]
    thumb_tip_dist = _dist(landmarks[4], pinky_mcp)
    thumb_base_dist = _dist(landmarks[2], pinky_mcp)
    thumb_up = 1 if thumb_tip_dist > thumb_base_dist * margin else 0
    states.insert(0, thumb_up)

    return tuple(states)


def match_gesture(pattern):
    """จับคู่ pattern ที่ตรวจจับได้กับท่าที่รู้จักใน GESTURES
    - ตรงเป๊ะ -> ใช้เลย
    - ไม่ตรงเป๊ะ -> หาอันที่ใกล้เคียงที่สุด (ผิดไม่เกิน 1 นิ้ว) กันภาพกระพริบตอนนิ้วสั่นนิดหน่อย
    - ไม่เข้าเกณฑ์ไหนเลย -> คืนค่า None (ยังไม่รู้จักท่านี้)
    คืนค่า (slug, description) หรือ (None, None)
    """
    for slug, desc, pat in GESTURES:
        if pat == pattern:
            return slug, desc

    best = None
    best_diff = None
    for slug, desc, pat in GESTURES:
        diff = sum(1 for a, b in zip(pat, pattern) if a != b)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best = (slug, desc)
    if best is not None and best_diff <= 1:
        return best
    return None, None


# ----------------------------------------------------------------------------
# โหลดภาพ (รองรับ gif เคลื่อนไหว)
# ----------------------------------------------------------------------------
def _pil_frame_to_bgr(pil_frame, size):
    frame = pil_frame.convert("RGBA")
    bg = Image.new("RGBA", frame.size, (0, 0, 0, 255))
    bg.paste(frame, mask=frame.split()[3])
    rgb = bg.convert("RGB")
    arr = np.array(rgb)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, size)


def load_image_frames(path, size):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gif":
        pil_img = Image.open(path)
        frames = [
            _pil_frame_to_bgr(f.copy(), size)
            for f in ImageSequence.Iterator(pil_img)
        ]
        return frames if frames else None
    else:
        pil_img = Image.open(path)
        return [_pil_frame_to_bgr(pil_img, size)]


def build_placeholder(title, subtitle="", size=(PANEL_WIDTH, PANEL_WIDTH), color=(90, 90, 90)):
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    cv2.circle(img, (w // 2, h // 2 - 20), min(w, h) // 2 - 40, (255, 255, 255), 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    words = title.split(" ")
    lines, cur = [], ""
    for word in words:
        test = (cur + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(test, font, 0.7, 2)
        if tw > w - 40 and cur:
            lines.append(cur)
            cur = word
        else:
            cur = test
    if cur:
        lines.append(cur)

    y = h // 2 + 20
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, 0.7, 2)
        cv2.putText(img, line, (w // 2 - tw // 2, y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += th + 12

    if subtitle:
        cv2.putText(img, subtitle, (20, h - 20), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return img


def load_image_bank(size=(PANEL_WIDTH, PANEL_WIDTH)):
    """คืนค่า dict: gesture_slug -> list ของเฟรม (BGR numpy array)
    หาไฟล์ <slug>.* ในโฟลเดอร์ images/ รองรับ .gif .png .jpg .jpeg .webp
    ถ้าไม่มีไฟล์ให้ใช้ placeholder (เฟรมเดียว) แทน"""
    bank = {}
    for slug, desc, _pat in GESTURES:
        frames = None
        for ext in ("gif", "png", "jpg", "jpeg", "webp"):
            p = os.path.join(IMAGES_DIR, f"{slug}.{ext}")
            if os.path.exists(p):
                try:
                    frames = load_image_frames(p, size)
                except Exception as e:
                    print(f"โหลดภาพ {p} ไม่สำเร็จ: {e}")
                if frames:
                    break
        if not frames:
            frames = [build_placeholder(desc, subtitle=f"({slug}.png/gif ยังไม่มี)", size=size)]
        bank[slug] = frames
    bank["_unknown"] = [build_placeholder("ยังไม่รู้จักท่านี้", size=size, color=(50, 50, 50))]
    return bank


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    ensure_model()

    image_bank = load_image_bank()

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_hand_presence_confidence=PRESENCE_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("เปิดกล้องไม่ได้ ลองเปลี่ยนค่า CAM_INDEX ในไฟล์นี้ดูครับ (0, 1, 2, ...)")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    shown_slug = "_unknown"
    shown_desc = "ยังไม่รู้จักท่านี้"
    pending_slug = "_unknown"
    pending_desc = shown_desc
    last_change_time = time.time()
    STABLE_SECONDS = 0.15
    start_time = time.time()
    finger_margin = FINGER_MARGIN
    GIF_FPS = 12
    anim_start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("อ่านภาพจากกล้องไม่ได้")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w = frame.shape[:2]
        cur_slug, cur_desc = "_unknown", "ไม่พบมือในกล้อง"

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            pts = [(lm.x, lm.y) for lm in hand_landmarks]
            pattern = get_finger_states(pts, margin=finger_margin)
            slug, desc = match_gesture(pattern)
            if slug is not None:
                cur_slug, cur_desc = slug, desc
            else:
                cur_slug, cur_desc = "_unknown", "ยังไม่รู้จักท่านี้"

            debug_txt = " ".join(
                f"{lbl}:{s}" for lbl, s in zip(["T", "I", "M", "R", "P"], pattern)
            )
            cv2.putText(frame, debug_txt, (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            px = [(int(x * w), int(y * h)) for x, y in pts]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, px[a], px[b], (0, 200, 0), 2)
            for x, y in px:
                cv2.circle(frame, (x, y), 4, (0, 100, 255), -1)

        if cur_slug != pending_slug:
            pending_slug = cur_slug
            pending_desc = cur_desc
            last_change_time = time.time()
        if time.time() - last_change_time >= STABLE_SECONDS and shown_slug != pending_slug:
            shown_slug = pending_slug
            shown_desc = pending_desc
            anim_start_time = time.time()

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        cv2.rectangle(frame, (0, 0), (max(280, len(shown_desc) * 12), 60), (0, 0, 0), -1)
        cv2.putText(frame, shown_desc, (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"margin: {finger_margin:.2f} (+/- ปรับ)", (w - 260, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        panel_frames = image_bank[shown_slug]
        if len(panel_frames) > 1:
            frame_idx = int((time.time() - anim_start_time) * GIF_FPS) % len(panel_frames)
        else:
            frame_idx = 0
        panel = cv2.resize(panel_frames[frame_idx], (PANEL_WIDTH, h))
        combined = np.hstack([frame, panel])
        cv2.imshow("Gesture Matcher - q=ออก / s=บันทึกภาพ", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = os.path.join(CAPTURES_DIR, f"capture_{int(time.time())}.png")
            cv2.imwrite(fname, combined)
            print(f"บันทึกภาพแล้ว: {fname}")
        elif key == ord("+") or key == ord("="):
            finger_margin = max(0.80, finger_margin - 0.02)
            print(f"FINGER_MARGIN = {finger_margin:.2f} (นับง่ายขึ้น)")
        elif key == ord("-") or key == ord("_"):
            finger_margin = min(1.30, finger_margin + 0.02)
            print(f"FINGER_MARGIN = {finger_margin:.2f} (นับยากขึ้น)")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()