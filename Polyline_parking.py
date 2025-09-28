import cv2
import numpy as np
import torch
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO('yolov8x.pt')
model.to(device)

drawing = False
parking_spaces = []
current_polygon = []
occupied_parking_lots = 0
lot_status = []

def draw_polygon(event, x, y, flags, param):
    global drawing, current_polygon, lot_status

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_polygon = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_polygon.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_polygon.append((x, y))
        parking_spaces.append(np.array(current_polygon, np.int32))
        lot_status.append(False)
        current_polygon = []

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

video_path = "C:/Users/Adi/OneDrive/Desktop/chinmay/VIDEO-2025-09-24-22-42-38.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret, frame = cap.read()
if not ret or frame is None:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()
cv2.namedWindow('Parking Detection')
cv2.setMouseCallback('Parking Detection', draw_polygon)

while True:
    ret, frame = cap.read()
    # if not ret:
    #     print("Error: Failed to capture frame. Retrying...")
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #     continue
    # if not ret or frame is None:
    #     print("End of video or failed to read frame. Exiting...")
    #     break


    frame = cv2.resize(frame, (640, 360))

    results = model.predict(frame, device=device,verbose=False)

    boxes = results[0].boxes.data.cpu().numpy()

    occupied_parking_lots = 0

    for i, space in enumerate(parking_spaces):
        color = (0, 255, 0)
        lot_occupied = False

        for box in boxes:
            x1, y1, x2, y2, conf, cls_id = map(int, box)
            class_name = model.names[cls_id]

            if class_name in ['car', 'truck', 'bus']:
                car_center = (x1 + x2) // 2, (y1 + y2) // 2

                if point_in_polygon(car_center, space):
                    color = (0, 0, 255)
                    lot_occupied = True

        lot_status[i] = lot_occupied
        if lot_occupied:
            occupied_parking_lots += 1

        cv2.polylines(frame, [space], isClosed=True, color=color, thickness=2)

        M = cv2.moments(space)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            lot_number_position = (cx - 30, cy)

            cv2.rectangle(frame, (lot_number_position[0] - 10, lot_number_position[1] - 10),
                          (lot_number_position[0] + 10, lot_number_position[1] + 10), (0, 0, 0), -1)

            cv2.putText(frame, str(i + 1), lot_number_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

    if len(current_polygon) > 1:
        cv2.polylines(frame, [np.array(current_polygon, np.int32)], isClosed=False, color=(255, 255, 0), thickness=2)

    total_parking_lots = len(parking_spaces)

    cv2.rectangle(frame, (10, 10), (230, 50), (0, 0, 0), -1)

    cv2.putText(frame, f"Parking Lots: {occupied_parking_lots}/{total_parking_lots}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Parking Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()