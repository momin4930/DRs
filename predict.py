import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO models
model_ball = YOLO('D:\\STUDY\\Semester 7\\DRS\\Ball\\BallModel1.pt')  # Ball detection model
model_stump = YOLO('D:\\STUDY\\Semester 7\\DRS\\Ball\\bestStump.pt')  # Stump detection model

# Open video
video_path = "D:\\STUDY\\Semester 7\\DRS\\MyDRS2\\vids\\football.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Set the desired dimensions for the output window
window_width = 640
window_height = 640
cv2.namedWindow("Ball and Stump Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Ball and Stump Detection", window_width, window_height)

trajectory_points = []
stump_saved = False
stump_x1, stump_x2, stump_y2 = None, None, None
prediction_started = False
predicted_trajectory = []
prediction_result = ""

# Load "Pitching in Line" picture
pitching_image = cv2.imread("D:\\STUDY\\Semester 7\\DRS\\MyDRS2\\vids\\inline.jpeg")

# Resize and position "Pitching in Line" picture
resize_width = 250  # Desired width
resize_height = 250  # Desired height
pitching_image_resized = cv2.resize(pitching_image, (resize_width, resize_height))
inline_picture_shown = False  # Track whether the picture is already displayed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    overlay = frame.copy()
    results_ball = model_ball.predict(source=[frame], conf=0.9, save=False)
    ball_detected = False

    for r in results_ball[0].boxes:
        box = r.xyxy[0]
        clsID = int(r.cls)
        confidence = float(r.conf)

        if clsID == 0:
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if not prediction_started:
                trajectory_points.append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Ball {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            ball_detected = True
            break

    results_stump = model_stump.predict(source=[frame], conf=0.9, save=False)
    for r in results_stump[0].boxes:
        box = r.xyxy[0]
        clsID = int(r.cls)
        confidence = float(r.conf)

        if clsID == 0 and not stump_saved:
            x1, y1, x2, y2 = map(int, box)
            stump_x1, stump_x2, stump_y2 = x1, x2, y2
            stump_saved = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Stump {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw the vertical stump line (midpoint between stump_x1 and stump_x2)
    if stump_saved:
        center_x = (stump_x1 + stump_x2) // 2
        line_start_y = stump_y2 + 20
        line_end_y = stump_y2 + 1000

        # Draw the stump line (in red)
        red_color = (0, 0, 255)
        cv2.line(overlay, (center_x, stump_y2), (center_x, line_end_y), red_color, 32)

    # Keep inline picture displayed after pitching
    if len(trajectory_points) >= 2 and not prediction_started:
        dy_previous = trajectory_points[-2][1] - trajectory_points[-1][1]
        if dy_previous > 0:  # Ball moving downward
            prediction_started = True
            inline_picture_shown = True  # Flag to keep the picture shown
            predicted_trajectory = []
            dx = trajectory_points[-1][0] - trajectory_points[-2][0]
            dy = trajectory_points[-1][1] - trajectory_points[-2][1]
            last_point = trajectory_points[-1]

            # Predict trajectory
            while last_point[1] < stump_y2 + 1000:  # Ball predicted to travel below stump
                last_point = (last_point[0] + dx, last_point[1] + dy)
                if not (0 <= last_point[0] < frame.shape[1] and 0 <= last_point[1] < frame.shape[0]):
                    break
                predicted_trajectory.append(last_point)

                if stump_x1 <= last_point[0] <= stump_x2 and last_point[1] >= stump_y2:
                    prediction_result = "Hitting the Stumps"
                    break
                else:
                    prediction_result = "Missing the Stumps"

    # Display "Pitching in Line" picture
    if inline_picture_shown:
        frame[10:10 + resize_height, 10:10 + resize_width] = pitching_image_resized
        cv2.putText(frame, prediction_result, (10, 10 + resize_height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0 , 255), 2)

    # Draw actual trajectory
    if len(trajectory_points) > 1:
        for i in range(1, len(trajectory_points)):
            cv2.line(overlay, trajectory_points[i - 1], trajectory_points[i], (0, 255, 0), 32)

    # Draw predicted trajectory
    if prediction_started:
        for i in range(1, len(predicted_trajectory)):
            cv2.line(overlay, predicted_trajectory[i - 1], predicted_trajectory[i], (255, 0, 0), 32)

    # Overlay the visualization on the frame
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    cv2.imshow("Ball and Stump Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()









# import cv2
# import numpy as np
# from ultralytics import YOLO

# # Load YOLO models
# model_ball = YOLO('D:\\STUDY\\Semester 7\\DRS\\Ball\\BallModel.pt')  # Ball detection model
# model_stump = YOLO('D:\\STUDY\\Semester 7\\DRS\\Ball\\bestStump.pt')  # Stump detection model

# # Open video
# video_path = "D:\\STUDY\\Semester 7\\DRS\\MyDRS2\\vids\\30.mp4"
# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# # Set the desired dimensions for the output window
# window_width = 640
# window_height = 640
# cv2.namedWindow("Ball and Stump Detection", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Ball and Stump Detection", window_width, window_height)

# trajectory_points = []
# stump_saved = False
# stump_x1, stump_x2, stump_y2 = None, None, None
# pitching_in_line = False

# # Load the image to display for "Pitching in Line"
# pitching_image = cv2.imread("D:\\STUDY\\Semester 7\\DRS\\MyDRS2\\vids\\inline.jpeg")
# pitching_image = cv2.resize(pitching_image, (240, 240))  # Resize to a small image (80x80)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video.")
#         break

#     overlay = frame.copy()
#     results_ball = model_ball.predict(source=[frame], conf=0.1, save=False)
#     ball_detected = False

#     for r in results_ball[0].boxes:
#         box = r.xyxy[0]
#         clsID = int(r.cls)
#         confidence = float(r.conf)

#         if clsID == 0:
#             x1, y1, x2, y2 = map(int, box)
#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#             trajectory_points.append((cx, cy))
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"Ball {confidence:.2f}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             ball_detected = True
#             break

#     results_stump = model_stump.predict(source=[frame], conf=0.3, save=False)
#     for r in results_stump[0].boxes:
#         box = r.xyxy[0]
#         clsID = int(r.cls)
#         confidence = float(r.conf)

#         if clsID == 0 and not stump_saved:
#             x1, y1, x2, y2 = map(int, box)
#             stump_x1, stump_x2, stump_y2 = x1, x2, y2
#             stump_saved = True

#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(frame, f"Stump {confidence:.2f}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#     if stump_saved:
#         center_x = (stump_x1 + stump_x2) // 2
#         line_start_y = stump_y2 + 20
#         line_end_y = stump_y2 + 1000
        
#         # Draw the stump line
#         red_color = (0, 0, 255)
#         cv2.line(overlay, (center_x, stump_y2), (center_x, line_end_y), red_color, 32)

#         if ball_detected:
#             for (cx, cy) in trajectory_points:
#                 if stump_x1 <= cx <= stump_x2 and line_start_y <= cy <= line_end_y:
#                     pitching_in_line = True
#                     break

#     if pitching_in_line:
#         # Overlay the pitching image in the top-left corner
#         x_offset, y_offset = 10, 10  # Top-left corner offsets
#         frame[y_offset:y_offset+pitching_image.shape[0], x_offset:x_offset+pitching_image.shape[1]] = pitching_image

#     if len(trajectory_points) > 1:
#         for i in range(1, len(trajectory_points)):
#             cv2.line(overlay, trajectory_points[i - 1], trajectory_points[i], (0, 255, 0), 32)

#     cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
#     cv2.imshow("Ball and Stump Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



