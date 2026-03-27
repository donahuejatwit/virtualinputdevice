import importlib.util

# Load modules
spec = importlib.util.spec_from_file_location("hands_module", "01_Hands.py")
hands_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hands_module)

# Camera Size (effects FPS/latency)
cam_width = 480
cam_height = 360

# Run gesture recognizer
hands_module.run(
    model='gesture_recognizer.task',
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    cam_id=0,
    cam_width=cam_width,
    cam_height=cam_height,
    width_buffer=int(cam_width * 0.1),
    height_buffer=int(cam_height * 0.1),
    click_gesture_name = 'Open_Palm',
    smoothing_factor=0.8,
    click_on_score_threshold=0.58,
    click_off_score_threshold=0.38,
    click_on_frames=2,
    click_off_frames=3
    )
