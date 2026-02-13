from ultralytics import YOLO
from picamera2 import Picamera2
import cv2 as cv
import time
import numpy as np

# Load YOLOv8-nano (smallest, fastest for Pi)
print("Loading YOLOv8-nano model...")
model = YOLO('yolov8n.pt')

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

time.sleep(2)
print("Search and Rescue Mode - Press 'q' to quit, 's' to save")

frame_count = 0

try:
    while True:
        frame = picam2.capture_array()
        
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        
        # Run YOLO detection
        results = model(frame, classes=[0], verbose=False)  # class 0 = person
        
        # Get detections
        boxes = results[0].boxes
        person_count = len(boxes)
        
        # Convert RGB to BGR for display
        display_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        
        # Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Draw box
            cv.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Person {conf:.2f}"
            cv.putText(display_frame, label, (x1, y1-10), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display count
        status = f"SURVIVORS: {person_count}" if person_count > 0 else "SEARCHING..."
        color = (0, 255, 0) if person_count > 0 else (0, 0, 255)
        cv.putText(display_frame, status, (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv.imshow('Search and Rescue', display_frame)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"rescue_{frame_count}.jpg"
            cv.imwrite(filename, display_frame)
            print(f"Saved {filename} - {person_count} person(s) detected")
            frame_count += 1

except KeyboardInterrupt:
    pass
finally:
    picam2.stop()
    cv.destroyAllWindows()
