Wentworth Institute of Technology
COMP 5500 Senior Project

Overview
A small Python script that uses MediaPipe to move the mouse and optionally send keyboard keys in "Taiko mode". Move your hand to move the cursor. Curl index or middle finger to click while in mouse the 🤟 gesture to toggle Taiko mode, where finger curls send key inputs for the Taiko gamemode in OSU.

Requirements

    Python 3.8+
        opencv-python
        mediapipe
        pynput
        pyautogui

Files

    main.py
    util.py
    01_Hands.py
    README.md

Quick config

    cam_id, cam_width, cam_height
    smoothing_factor, width_buffer, height_buffer
    grace_period, hold_threshold
    click_on_frames, click_off_frames
    taiko_key_bindings, taiko_toggle_frames, taiko_toggle_cooldown
    