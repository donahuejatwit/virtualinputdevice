import time
import cv2
import mediapipe as mp
from pynput.mouse import Controller, Button
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Init Vars
frame_counter = 0
fps = 0
start_time = time.time()
mouse_controller = Controller()


# Click state
is_mouse_pressed = False
gesture_start_time = None
last_seen_time = 0
# Clicking Logic
grace_period = 0.25    # when holding if the gesture stops it does not release for x seconds
hold_threshold = 0.5  # if the gesture for click lasts x seconds it begins a hold
def handle_click(gesture_detected, current_time):
    global is_mouse_pressed
    global gesture_start_time
    global last_seen_time

    if gesture_detected:
        last_seen_time = current_time

        if gesture_start_time is None:
            gesture_start_time = current_time
            return

        if not is_mouse_pressed and current_time - gesture_start_time >= hold_threshold:
            mouse_controller.press(Button.left)
            is_mouse_pressed = True

    else:

        if gesture_start_time is None:
            return

        if current_time - last_seen_time < grace_period:
            return

        duration = last_seen_time - gesture_start_time

        if is_mouse_pressed:
            mouse_controller.release(Button.left)
            is_mouse_pressed = False

        elif duration < hold_threshold:
            mouse_controller.click(Button.left)

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
        smoothing_factor):

    global frame_counter, fps, start_time

    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

    recognition_result_list = []
    prev_mouse_x, prev_mouse_y = 0, 0

    def save_result(result: vision.GestureRecognizerResult,
                    output_image: mp.Image,
                    timestamp_ms: int):

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

            if result.hand_landmarks:

                h, w, _ = frame.shape

                for hand_landmarks in result.hand_landmarks:

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

                    target_x = int((avg_x - x_buffer_norm) / (1 - 2 * x_buffer_norm) * screen_width)
                    target_y = int((avg_y - y_buffer_norm) / (1 - 2 * y_buffer_norm) * screen_height)

                    new_x = int(prev_mouse_x * (1 - smoothing_factor) + target_x * smoothing_factor)
                    new_y = int(prev_mouse_y * (1 - smoothing_factor) + target_y * smoothing_factor)

                    mouse_controller.position = (new_x, new_y)
                    prev_mouse_x, prev_mouse_y = new_x, new_y

                    for lm in hand_landmarks:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

            gesture_detected = False

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

                    if gesture_name == click_gesture_name and score > 0.7:
                        gesture_detected = True
                        break

            handle_click(gesture_detected, time.time())

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
    cap.release()
    cv2.destroyAllWindows()
