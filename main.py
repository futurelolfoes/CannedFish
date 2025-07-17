import cv2
from ultralytics import YOLO

# Confidence threshold: only detections with confidence >= CONF_THRESHOLD are considered
CONF_THRESHOLD = 0.4  # Adjust this value (0.0 - 1.0) for sensitivity

# Load a YOLOv8 model (replace with your custom seat detection weights if available)
model = YOLO('yolov8n.pt')  # or 'path/to/custom_seat_weights.pt'

# Define class mappings (0: person, 1: empty_seat, 2: occupied_seat)
class_names = {0: 'person', 1: 'empty_seat', 2: 'occupied_seat'}
target_classes = set(class_names.keys())

# Video source: replace 'subway.mp4' with camera index (0) or RTSP stream
cap = cv2.VideoCapture('subway.mp4')
if not cap.isOpened():
    print("Error: Cannot open video source")
    exit()

# Prepare video writer to export processed video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 files
output_path = 'output.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)[0]

    # Counters
    person_count = 0
    empty_seat_count = 0
    occupied_seat_count = 0

    # Draw detections above confidence threshold, only for target classes
    for *box, conf, cls in results.boxes.data:
        conf = float(conf)
        cls = int(cls)
        if conf < CONF_THRESHOLD or cls not in target_classes:
            continue

        # Convert box coords to integers
        x1, y1, x2, y2 = map(int, box)
        label = class_names[cls]

        # Assign colors and count
        if label == 'person':
            person_count += 1
            color = (0, 255, 0)
        elif label == 'empty_seat':
            empty_seat_count += 1
            color = (255, 255, 0)
        else:  # occupied_seat
            occupied_seat_count += 1
            color = (0, 0, 255)

        # Draw box and label with confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display counts and threshold on frame
    cv2.putText(frame, f"People: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Empty Seats: {empty_seat_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Occupied Seats: {occupied_seat_count}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Threshold: {CONF_THRESHOLD}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Write the processed frame to output video
    out.write(frame)

    # Optional: show live preview
    cv2.imshow('YOLO Seat & Human Monitor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Exported processed video to {output_path}")
