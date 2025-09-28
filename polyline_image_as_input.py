import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLO model
model = YOLO('yolov8x.pt')
model.to(device)

# Initialize variables
drawing = False
parking_spaces = []
current_polygon = []
occupied_parking_lots = 0
lot_status = []

# Function to draw a polygon for parking spaces
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

# Function to check if a point is within a polygon
def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# Load and resize image to 5:4 ratio
image_path = "C:/Users/Adi/OneDrive/Desktop/chinmay/PHOTO-2025-09-24-22-42-00.jpg"
original_frame = cv2.imread(image_path)

if original_frame is None:
    print("Error: Could not open image.")
    exit()

# Resize the image to 800x640 (5:4 aspect ratio)
original_frame = cv2.resize(original_frame, (800, 640))
cv2.namedWindow('Parking Detection')
cv2.setMouseCallback('Parking Detection', draw_polygon)

# Run object detection on the image
results = model.predict(original_frame, device=device)
boxes = results[0].boxes.data.cpu().numpy()

# Main loop to keep updating the display
while True:
    # Copy original frame to reset the drawing
    frame = original_frame.copy()

    # Draw incomplete polygon while moving the cursor
    if len(current_polygon) > 1:
        cv2.polylines(frame, [np.array(current_polygon, np.int32)], isClosed=False, color=(255, 255, 0), thickness=2)

    # Process each defined parking space
    occupied_parking_lots = 0
    for i, space in enumerate(parking_spaces):
        color = (0, 255, 0)  # Green for unoccupied
        lot_occupied = False

        # Check if any detected object is inside the parking space
        for box in boxes:
            x1, y1, x2, y2, conf, cls_id = map(int, box)
            class_name = model.names[cls_id]

            # Only consider vehicle classes
            if class_name in ['car', 'truck', 'bus']:
                car_center = (x1 + x2) // 2, (y1 + y2) // 2

                # If car is in parking space, mark as occupied
                if point_in_polygon(car_center, space):
                    color = (0, 0, 255)  # Red for occupied
                    lot_occupied = True

        lot_status[i] = lot_occupied
        if lot_occupied:
            occupied_parking_lots += 1

        # Draw the polygon on the image
        cv2.polylines(frame, [space], isClosed=True, color=color, thickness=2)

        # Display lot number at the center of the parking space
        M = cv2.moments(space)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            lot_number_position = (cx - 30, cy)

            cv2.rectangle(frame, (lot_number_position[0] - 10, lot_number_position[1] - 10),
                          (lot_number_position[0] + 10, lot_number_position[1] + 10), (0, 0, 0), -1)

            cv2.putText(frame, str(i + 1), lot_number_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

    # Display total parking lot status on the image
    total_parking_lots = len(parking_spaces)
    cv2.rectangle(frame, (10, 10), (230, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"Parking Lots: {occupied_parking_lots}/{total_parking_lots}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the image with drawn parking spaces
    cv2.imshow('Parking Detection', frame)

    # Break loop on pressing 'space' or close on 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') or key == ord('q'):
        break

cv2.destroyAllWindows()
