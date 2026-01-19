import argparse
import sys
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()

# Global variable. True when showing the Radial Menu.
ACTIVE=False



def run(model: str, num_hands: int,
        min_hand_detection_confidence: float,
        min_hand_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
      model: Name of the gesture recognition model bundle.
      num_hands: Max number of hands can be detected by the recognizer.
      min_hand_detection_confidence: The minimum confidence score for hand
        detection to be considered successful.
      min_hand_presence_confidence: The minimum confidence score of hand
        presence score in the hand landmark detection.
      min_tracking_confidence: The minimum confidence score for the hand
        tracking to be considered successful.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
  """

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 50  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 0)  # black
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Label box parameters
  label_text_color = (255, 255, 255)  # white
  label_font_size = 1
  label_thickness = 2

  recognition_frame = None
  recognition_result_list = []

  def save_result(result: vision.GestureRecognizerResult,
                  unused_output_image: mp.Image, timestamp_ms: int):
      global FPS, COUNTER, START_TIME

      # Calculate the FPS
      if COUNTER % fps_avg_frame_count == 0:
          FPS = fps_avg_frame_count / (time.time() - START_TIME)
          START_TIME = time.time()

      recognition_result_list.append(result)
      COUNTER += 1

  # Initialize the gesture recognizer model
  base_options = python.BaseOptions(model_asset_path=model)
  options = vision.GestureRecognizerOptions(base_options=base_options,
                                          running_mode=vision.RunningMode.LIVE_STREAM,
                                          num_hands=num_hands,
                                          min_hand_detection_confidence=min_hand_detection_confidence,
                                          min_hand_presence_confidence=min_hand_presence_confidence,
                                          min_tracking_confidence=min_tracking_confidence,
                                          result_callback=save_result)
  recognizer = vision.GestureRecognizer.create_from_options(options)
           

           
  import math
  def getDistance(p1, p2):
    dist=math.dist(p1,p2)
    return dist

  def getCenter(p1, p2):
    cnr = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))
    return cnr

  def getEvent(cnr, cnrCur):
    #
    # Now figure out in which of the 4 menus the hand is, to initiate event.
    #
    
    L=1 
  
    v1 = (-L, -L) # point top left
    v2 = ( L, -L) # point top right
    v3 = ( L,  L) # point bottom right
    v4 = (-L,  L) # point bottom left
   
    v1p = ( L, -L) # perpendicular vector to v1
    v2p = ( L,  L) # perpendicular vector to v2
    v3p = (-L,  L) # perpendicular vector to v3
    v4p = (-L, -L) # perpendicular vector to v4

    # vector pointing from center of radial menu, to hand
    e = (cnrCur[0] - cnr[0], cnrCur[1] - cnr[1]) 

    v1pDOTe = v1p[0] * e[0] + v1p[1] * e[1] # v1p DOT e
    v2pDOTe = v2p[0] * e[0] + v2p[1] * e[1] # v2p DOT e
    v3pDOTe = v3p[0] * e[0] + v3p[1] * e[1] # v3p DOT e
    v4pDOTe = v4p[0] * e[0] + v4p[1] * e[1] # v4p DOT e

    
    if v1pDOTe > 0 and v2pDOTe < 0:   # hand on top of v1 and below v2
      return "UP"
    elif v2pDOTe > 0 and v3pDOTe < 0: # hand on top of v2 and below v3
      return "RIGHT"
    elif v3pDOTe > 0 and v4pDOTe < 0: # hand on top of v3 and below v4
      return "DOWN"    
    elif v4pDOTe > 0 and v1pDOTe < 0: # hand on top of v4 and below v1
      return "LEFT"
    else:
      return "DO NOT KNOW" # I guess when we are exactly on one of the lines?


  def drawUI(current_frame, c2, d2, cnr, dist, cnrCur):
    overlay = current_frame.copy()
    r1=int((dist/2)+dist/2 *340/100)
    r2=int((dist/2)+dist/2 *10/100)

    cv2.ellipse(current_frame, cnr,   (r1,r1), 0, 0, 360, (255, 155, 100), -1)
    cv2.ellipse(current_frame, cnr,   (r2,r2), 0, 0, 360, (255, 50, 50), -1)

    cv2.addWeighted(overlay, 0.5, current_frame, 0.5, 0, current_frame)

    cv2.circle(current_frame, cnr,   r2, (255, 50, 50), 2)

    L=200
    cv2.line(current_frame, cnr, (cnr[0]-L, cnr[1]-L), (255,255,255), 1) # top left
    cv2.line(current_frame, cnr, (cnr[0]+L, cnr[1]-L), (255,255,255), 1) # top right
    cv2.line(current_frame, cnr, (cnr[0]+L, cnr[1]+L), (255,255,255), 1) # bottom right
    cv2.line(current_frame, cnr, (cnr[0]-L, cnr[1]+L), (255,255,255), 1) # bottom left


    if getDistance(cnr,cnrCur) > dist:
      overlay = current_frame.copy()
      cv2.circle(current_frame, cnrCur,   r2, (50, 255, 50), 20)
      cv2.addWeighted(overlay, 0.3, current_frame, 0.7, 0, current_frame)


           
  
  cnr=(0,0)   # holds the center point of the Radial Menu (defined by the midpoint of index_pip and wrist)
  dist=0      # distance between the two points (index_pip and wrist)
  ACTIVEcntr=0  # When ACTIVE (menu is shown) we get some flickering as the recognizers skips some frames.
                # This counter is incremented each time, when over a threshold, we hide the menu.
  evnt="None"   # Holds the event generated (which menu item we selected: UP, RIGHT, DOWN, LEFT)

  global ACTIVE # When True, we show the menu

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Run gesture recognizer using the model.
    recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(FPS)
    text_location = (left_margin, row_size)
    current_frame = image
    cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                font_size, text_color, font_thickness, cv2.LINE_AA)

    if recognition_result_list:
      # Draw landmarks and write the text for each hand.
      for hand_index, hand_landmarks in enumerate(
          recognition_result_list[0].hand_landmarks):
        # Calculate the bounding box of the hand
        x_min = min([landmark.x for landmark in hand_landmarks])
        y_min = min([landmark.y for landmark in hand_landmarks])
        y_max = max([landmark.y for landmark in hand_landmarks])

        # Convert normalized coordinates to pixel values
        frame_height, frame_width = current_frame.shape[:2]
        x_min_px = int(x_min * frame_width)
        y_min_px = int(y_min * frame_height)
        y_max_px = int(y_max * frame_height)

        # Get gesture classification results
        if recognition_result_list[0].gestures:
          gesture = recognition_result_list[0].gestures[hand_index]
          category_name = gesture[0].category_name
          score = round(gesture[0].score, 2)
          result_text = f'{category_name} ({score})'
          
          #
          # Can be used if we are interested in Left/Right hand
          #
          #print(recognition_result_list[0].handedness[hand_index])

          # Compute text size
          text_size = \
          cv2.getTextSize(result_text, cv2.FONT_HERSHEY_DUPLEX, label_font_size,
                          label_thickness)[0]
          text_width, text_height = text_size

          # Calculate text position (above the hand)
          text_x = x_min_px
          text_y = y_min_px - 10  # Adjust this value as needed

          # Make sure the text is within the frame boundaries
          if text_y < 0:
            text_y = y_max_px + text_height

          # Draw the text
          cv2.putText(current_frame, result_text, (text_x, text_y),
                      cv2.FONT_HERSHEY_DUPLEX, label_font_size,
                      label_text_color, label_thickness, cv2.LINE_AA)
          
          #print(gesture[0].category_name)

          

          

        # Draw hand landmarks on the frame
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                          z=landmark.z) for landmark in
          hand_landmarks
        ])
        
        whichHand = recognition_result_list[0].handedness[hand_index][0].display_name
        
        if(whichHand == "Left"):
          if category_name == "Closed_Fist": 
            image_rows, image_cols, _ = current_frame.shape

            c1=hand_landmarks_proto.landmark[mp_hands.HandLandmark.WRIST]
            c2 = mp_drawing._normalized_to_pixel_coordinates(c1.x, c1.y, image_cols, image_rows)

            d1=hand_landmarks_proto.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            d2 = mp_drawing._normalized_to_pixel_coordinates(d1.x, d1.y, image_cols, image_rows)

            if c2 != None and d2 != None:
              if( ACTIVE == False ):
                ACTIVE = True
                cnr = getCenter(c2,d2)
                dist = getDistance(c2,d2)

              drawUI(current_frame, c2, d2, cnr, dist, getCenter(c2,d2))
              ACTIVEcntr=0
              
          elif category_name == "Open_Palm" and ACTIVE:
            image_rows, image_cols, _ = current_frame.shape

            c1=hand_landmarks_proto.landmark[mp_hands.HandLandmark.WRIST]
            c2 = mp_drawing._normalized_to_pixel_coordinates(c1.x, c1.y, image_cols, image_rows)

            d1=hand_landmarks_proto.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            d2 = mp_drawing._normalized_to_pixel_coordinates(d1.x, d1.y, image_cols, image_rows)

            if c2 != None and d2 != None:
              cnrCur = getCenter(c2,d2)
              if getDistance(cnr,cnrCur) > dist:
                evnt = getEvent(cnr, cnrCur)
              ACTIVEcntr=0
              ACTIVE = False

          else:
            if ACTIVE:
              ACTIVEcntr=ACTIVEcntr+1
              if ACTIVEcntr > 15:  # to avoid flickering
                ACTIVEcntr=0
                ACTIVE=False        


        
        cv2.putText(current_frame, f"(Left Hand) LAST EVENT: {evnt}", (22, 80),
                      cv2.FONT_HERSHEY_DUPLEX, label_font_size,
                      (128,0,0), label_thickness, cv2.LINE_AA)      





        mp_drawing.draw_landmarks(
          current_frame,
          hand_landmarks_proto,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

      recognition_frame = current_frame
      recognition_result_list.clear()

    if recognition_frame is not None:
        cv2.imshow('GestureRecognition / UI', recognition_frame)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
        break

  recognizer.close()
  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of gesture recognition model.',
      required=False,
      default='gesture_recognizer.task')
  parser.add_argument(
      '--numHands',
      help='Max number of hands that can be detected by the recognizer.',
      required=False,
      default=2)
  parser.add_argument(
      '--minHandDetectionConfidence',
      help='The minimum confidence score for hand detection to be considered '
           'successful.',
      required=False,
      default=0.5)
  parser.add_argument(
      '--minHandPresenceConfidence',
      help='The minimum confidence score of hand presence score in the hand '
           'landmark detection.',
      required=False,
      default=0.5)
  parser.add_argument(
      '--minTrackingConfidence',
      help='The minimum confidence score for the hand tracking to be '
           'considered successful.',
      required=False,
      default=0.5)
  # Finding the camera ID can be very reliant on platform-dependent methods.
  # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0.
  # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
  # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      default=1024)  #640
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      default=480)
  args = parser.parse_args()

  run(args.model, int(args.numHands), args.minHandDetectionConfidence,
      args.minHandPresenceConfidence, args.minTrackingConfidence,
      int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
  main()