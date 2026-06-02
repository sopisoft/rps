import cv2
import mediapipe as mp
import time
import random
from sklearn.svm import SVC


def process_landmarks(landmarks):
    base_x, base_y = landmarks[0].x, landmarks[0].y
    processed = []
    for lm in landmarks:
        processed.extend([lm.x - base_x, lm.y - base_y])
    max_val = max(map(abs, processed))
    if max_val > 0:
        processed = [p / max_val for p in processed]
    return processed


def draw_landmarks(image, landmarks):
    h, w, _ = image.shape
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)


def draw_center_text(
    img,
    text,
    y_ratio=0.5,
    font_scale=1.5,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    alpha=0.5,
):
    h, w, _ = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(2, int(font_scale * 2))
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    x = (w - text_w) // 2
    y = int(h * y_ratio)

    overlay = img.copy()
    cv2.rectangle(
        overlay, (x - 10, y - text_h - 10), (x + text_w + 10, y + 10), bg_color, -1
    )
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)


def main():
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))

    X = []
    y_data = []
    model = SVC(kernel="linear")
    labels = {0: "Gu", 1: "Choki", 2: "Pa"}

    state = "DATA_COLLECTION"

    countdown_start = 0
    result_start = 0

    user_hand = ""
    pc_hand = ""
    res = ""

    timestamp_ms = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms += 1
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        features = None
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                draw_landmarks(frame, hand_landmarks)
                features = process_landmarks(hand_landmarks)

        if state == "DATA_COLLECTION":
            count_g = y_data.count(0)
            count_c = y_data.count(1)
            count_p = y_data.count(2)

            draw_center_text(
                frame,
                "Hold [g]Gu [c]Choki [p]Pa to collect",
                y_ratio=0.15,
                font_scale=0.8,
            )
            draw_center_text(
                frame,
                f"Gu:{count_g} | Choki:{count_c} | Pa:{count_p}",
                y_ratio=0.25,
                font_scale=1.0,
            )
            draw_center_text(
                frame, "Press [t] to Train Model", y_ratio=0.85, font_scale=0.8
            )

            if features:
                if key == ord("g"):
                    X.append(features)
                    y_data.append(0)
                    draw_center_text(
                        frame,
                        "Collecting Gu...",
                        y_ratio=0.55,
                        font_scale=1.2,
                        text_color=(0, 255, 255),
                    )
                elif key == ord("c"):
                    X.append(features)
                    y_data.append(1)
                    draw_center_text(
                        frame,
                        "Collecting Choki...",
                        y_ratio=0.55,
                        font_scale=1.2,
                        text_color=(0, 255, 255),
                    )
                elif key == ord("p"):
                    X.append(features)
                    y_data.append(2)
                    draw_center_text(
                        frame,
                        "Collecting Pa...",
                        y_ratio=0.55,
                        font_scale=1.2,
                        text_color=(0, 255, 255),
                    )

            if key == ord("t"):
                if len(set(y_data)) >= 2:
                    state = "TRAINING"
                else:
                    draw_center_text(
                        frame,
                        "Need at least 2 signs!",
                        y_ratio=0.7,
                        font_scale=1.0,
                        text_color=(0, 0, 255),
                    )

        elif state == "TRAINING":
            draw_center_text(frame, "Training Model...", y_ratio=0.5)
            cv2.imshow("Janken System", frame)
            cv2.waitKey(1)
            model.fit(X, y_data)
            state = "PLAY_IDLE"

        elif state == "PLAY_IDLE":
            draw_center_text(frame, "Show hand to start", y_ratio=0.8, font_scale=1.0)
            if features:
                state = "COUNTDOWN"
                countdown_start = time.time()

        elif state == "COUNTDOWN":
            elapsed = time.time() - countdown_start
            remain = 3 - int(elapsed)

            if remain > 0:
                draw_center_text(frame, str(remain), y_ratio=0.5, font_scale=4.0)
            else:
                if features:
                    pred = model.predict([features])[0]
                    user_hand = labels[pred]
                    pc_hand_val = random.randint(0, 2)
                    pc_hand = labels[pc_hand_val]

                    if pred == pc_hand_val:
                        res = "DRAW"
                    elif (
                        (pred == 0 and pc_hand_val == 1)
                        or (pred == 1 and pc_hand_val == 2)
                        or (pred == 2 and pc_hand_val == 0)
                    ):
                        res = "YOU WIN!"
                    else:
                        res = "YOU LOSE..."

                    state = "RESULT"
                    result_start = time.time()
                else:
                    state = "PLAY_IDLE"

        elif state == "RESULT":
            draw_center_text(
                frame, f"You: {user_hand} VS PC: {pc_hand}", y_ratio=0.3, font_scale=1.5
            )

            if res == "YOU WIN!":
                t_col = (0, 255, 0)
            elif res == "YOU LOSE...":
                t_col = (255, 0, 0)
            else:
                t_col = (0, 255, 0)

            draw_center_text(
                frame, res, y_ratio=0.6, font_scale=3.0, text_color=t_col, alpha=0.8
            )

            if time.time() - result_start > 3.0:
                state = "PLAY_IDLE"

        cv2.imshow("Janken System", frame)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
