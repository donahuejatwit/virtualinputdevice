import time
import cv2
import mediapipe as mp
import util
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Init Vars
frame_counter = 0
fps = 0
start_time = time.time()
mouse_controller = MouseController()
keyboard_controller = KeyboardController()


# Click state
is_mouse_pressed = False
gesture_start_time = None
last_seen_time = 0

# Taiko mode state
taiko_mode = False
taiko_detect_counter = 0
last_mode_toggle_time = 0
taiko_toggle_frames = 4
taiko_toggle_cooldown = 0.8

taiko_key_bindings = {
    "Left_Index": "x",
    "Left_Middle": "z",
    "Right_Index": "c",
    "Right_Middle": "v",
}
taiko_key_states = {name: False for name in taiko_key_bindings}

# Clicking Logic
grace_period = 0.05    # small release grace to absorb single-frame detection drops
hold_threshold = .15 # start hold quickly while still allowing short tap clicks


def reset_mouse_click_state():
    global is_mouse_pressed, gesture_start_time

    if is_mouse_pressed:
        mouse_controller.release(Button.left)
    is_mouse_pressed = False
    gesture_start_time = None


def release_all_taiko_keys():
    for key_name, is_pressed in taiko_key_states.items():
        if is_pressed:
            keyboard_controller.release(taiko_key_bindings[key_name])
            taiko_key_states[key_name] = False


def get_hand_label(result, hand_index, hand_landmarks):
    if result.handedness and hand_index < len(result.handedness) and result.handedness[hand_index]:
        label = result.handedness[hand_index][0].category_name
        if label in ("Left", "Right"):
            return label

    # Fallback if handedness is unavailable: estimate by image x-position.
    return "Left" if hand_landmarks[0].x < 0.5 else "Right"


def update_taiko_key_state(key_name, finger_down):
    currently_pressed = taiko_key_states[key_name]
    bound_key = taiko_key_bindings[key_name]

    if finger_down and not currently_pressed:
        keyboard_controller.press(bound_key)
        taiko_key_states[key_name] = True
    elif not finger_down and currently_pressed:
        keyboard_controller.release(bound_key)
        taiko_key_states[key_name] = False


def get_normalized_finger_distance(
        hand_landmarks,
        tip_idx,
        base_idx,
        palm_left_idx=5,
        palm_right_idx=17):
    tip = hand_landmarks[tip_idx]
    base = hand_landmarks[base_idx]
    palm_left = hand_landmarks[palm_left_idx]
    palm_right = hand_landmarks[palm_right_idx]

    finger_distance = util.get_distance([(tip.x, tip.y), (base.x, base.y)])
    palm_width = util.get_distance([(palm_left.x, palm_left.y), (palm_right.x, palm_right.y)])

    if finger_distance is None or palm_width is None or palm_width <= 1e-6:
        return None

    return finger_distance / palm_width


def is_finger_down_by_distance(
        hand_landmarks,
        tip_idx,
        base_idx,
        previously_down,
        press_ratio=0.95,
        release_ratio=1.10):
    normalized_distance = get_normalized_finger_distance(hand_landmarks, tip_idx, base_idx)

    if normalized_distance is None:
        return False

    # Bent finger has smaller normalized distance; hysteresis prevents flicker.
    if previously_down:
        return normalized_distance <= release_ratio
    return normalized_distance <= press_ratio


def handle_click(gesture_detected, current_time):
    global is_mouse_pressed
    global gesture_start_time
    global last_seen_time

    if gesture_detected:
        last_seen_time = current_time

        if gesture_start_time is None:
            gesture_start_time = current_time

        # Press immediately when bend is detected so click starts without hold delay.
        if not is_mouse_pressed:
            mouse_controller.press(Button.left)
            is_mouse_pressed = True

    else:

        if gesture_start_time is None:
            return

        if current_time - last_seen_time < grace_period:
            return

        if is_mouse_pressed:
            mouse_controller.release(Button.left)
            is_mouse_pressed = False

        gesture_start_time = None


def run(model,
        num_hands,
        min_hand_detection_confidence,
        min_hand_presence_confidence,
        min_tracking_confidence,
        cam_id,
        cam_width,
        cam_height,
        width_buffer,
        height_buffer,
        click_gesture_name,
        smoothing_factor,
        click_on_score_threshold,
        click_off_score_threshold,
        click_on_frames,
        click_off_frames):

    global frame_counter, fps, start_time

    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

    recognition_result_list = []
    prev_mouse_x, prev_mouse_y = 0, 0
    mouse_index_down = False
    mouse_middle_down = False
    stable_mouse_click_detected = False
    mouse_click_on_counter = 0
    mouse_click_off_counter = 0

    def save_result(result,
                    output_image,
                    timestamp_ms):

        global frame_counter, fps, start_time

        frame_counter += 1
        fps = frame_counter / (time.time() - start_time)

        recognition_result_list.append(result)

    base_options = python.BaseOptions(model_asset_path=model)

    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        result_callback=save_result
    )

    recognizer = vision.GestureRecognizer.create_from_options(options)

    try:
        import pyautogui
        screen_width, screen_height = pyautogui.size()
    except:
        screen_width, screen_height = 1920, 1080
        print("Could not determine screen size, using 1080p")

    x_buffer_norm = width_buffer / cam_width
    y_buffer_norm = height_buffer / cam_height

    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            print("cam error")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        recognizer.recognize_async(mp_image, int(time.time() * 1000))

        if recognition_result_list:

            result = recognition_result_list.pop(0)

            current_time = time.time()
            taiko_detected = False

            if result.gestures:
                for gesture_list in result.gestures:
                    gesture = gesture_list[0]
                    gesture_name = gesture.category_name
                    score = gesture.score

                    cv2.putText(
                        frame,
                        f"{gesture_name} ({score:.2f})",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

                    if gesture_name == "ILoveYou":
                        taiko_detected = True

            global taiko_mode, taiko_detect_counter, last_mode_toggle_time
            if taiko_detected:
                taiko_detect_counter += 1
            else:
                taiko_detect_counter = 0


            if (
                taiko_detect_counter >= taiko_toggle_frames
                and current_time - last_mode_toggle_time >= taiko_toggle_cooldown
            ):
                taiko_mode = not taiko_mode
                last_mode_toggle_time = current_time
                taiko_detect_counter = 0

                if taiko_mode:
                    reset_mouse_click_state()
                    stable_mouse_click_detected = False
                    mouse_click_on_counter = 0
                    mouse_click_off_counter = 0
                    mouse_index_down = False
                    mouse_middle_down = False
                else:
                    release_all_taiko_keys()

            raw_mouse_click_detected = False
            if result.hand_landmarks:

                h, w, _ = frame.shape
                taiko_frame_finger_state = {name: False for name in taiko_key_bindings}
                mouse_click_detected = False

                for hand_index, hand_landmarks in enumerate(result.hand_landmarks):

                    weighted_x = hand_landmarks[0].x * 10
                    weighted_y = hand_landmarks[0].y * 10
                    total_weight = 10

                    for i, lm in enumerate(hand_landmarks):
                        if i == 0:
                            continue

                        weighted_x += lm.x
                        weighted_y += lm.y
                        total_weight += 1

                    avg_x = weighted_x / total_weight
                    avg_y = weighted_y / total_weight

                    if not taiko_mode:
                        target_x = int((avg_x - x_buffer_norm) / (1 - 2 * x_buffer_norm) * screen_width)
                        target_y = int((avg_y - y_buffer_norm) / (1 - 2 * y_buffer_norm) * screen_height)

                        new_x = int(prev_mouse_x * (1 - smoothing_factor) + target_x * smoothing_factor)
                        new_y = int(prev_mouse_y * (1 - smoothing_factor) + target_y * smoothing_factor)

                        mouse_controller.position = (new_x, new_y)
                        prev_mouse_x, prev_mouse_y = new_x, new_y

                        # Mouse mode click now shares taiko bend logic (index or middle curl).
                        index_bent = is_finger_down_by_distance(
                            hand_landmarks,
                            tip_idx=8,
                            base_idx=5,
                            previously_down=mouse_index_down
                        )
                        middle_bent = is_finger_down_by_distance(
                            hand_landmarks,
                            tip_idx=12,
                            base_idx=9,
                            previously_down=mouse_middle_down
                        )
                        mouse_index_down = index_bent
                        mouse_middle_down = middle_bent
                        mouse_click_detected = mouse_click_detected or index_bent or middle_bent
                        raw_mouse_click_detected = raw_mouse_click_detected or index_bent or middle_bent
                    else:
                        hand_label = get_hand_label(result, hand_index, hand_landmarks)

                        index_key = f"{hand_label}_Index"
                        middle_key = f"{hand_label}_Middle"

                        taiko_frame_finger_state[index_key] = is_finger_down_by_distance(
                            hand_landmarks,
                            tip_idx=8,
                            base_idx=5,
                            previously_down=taiko_key_states[index_key]
                        )
                        taiko_frame_finger_state[middle_key] = is_finger_down_by_distance(
                            hand_landmarks,
                            tip_idx=12,
                            base_idx=9,
                            previously_down=taiko_key_states[middle_key]
                        )

                    for lm in hand_landmarks:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

                if taiko_mode:
                    for key_name, is_down in taiko_frame_finger_state.items():
                        update_taiko_key_state(key_name, is_down)
                else:
                    if raw_mouse_click_detected:
                        stable_mouse_click_detected = True
                        mouse_click_on_counter = click_on_frames
                        mouse_click_off_counter = 0
                    else:
                        mouse_click_off_counter += 1
                        mouse_click_on_counter = 0

                    if stable_mouse_click_detected and mouse_click_off_counter >= click_off_frames:
                        stable_mouse_click_detected = False

                    handle_click(stable_mouse_click_detected, current_time)
            elif taiko_mode:
                release_all_taiko_keys()
            else:
                mouse_click_off_counter += 1
                mouse_click_on_counter = 0
                if stable_mouse_click_detected and mouse_click_off_counter >= click_off_frames:
                    stable_mouse_click_detected = False
                mouse_index_down = False
                mouse_middle_down = False
                handle_click(stable_mouse_click_detected, current_time)

            if taiko_mode:
                reset_mouse_click_state()

            mode_text = "TAIKO MODE" if taiko_mode else "MOUSE MODE"
            mode_color = (0, 200, 255) if taiko_mode else (0, 255, 0)
            cv2.putText(
                frame,
                mode_text,
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                mode_color,
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            cv2.imshow("Hand Gesture Mouse Control", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    recognizer.close()
    reset_mouse_click_state()
    release_all_taiko_keys()
    cap.release()
    cv2.destroyAllWindows()