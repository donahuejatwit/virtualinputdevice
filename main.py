# Jack Donahue & Giang Do 

import sys
import importlib.util

# Load module with numeric name
spec = importlib.util.spec_from_file_location("hands_module", "01_Hands.py")
hands_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hands_module)

# Run the gesture recognizer with hand-to-mouse tracking
hands_module.run(model='gesture_recognizer.task', 
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    camera_id=0,
    width=1024,
    height=480)
