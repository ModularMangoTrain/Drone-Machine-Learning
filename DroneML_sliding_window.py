import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from picamera2 import Picamera2
import cv2 as cv
import time

spatial = torch.load("spatial_person_detector_quantized.pth", map_location=torch.device('cpu'), weights_only=False)
device = torch.device("cpu")
spatial.to(device)
spatial.eval()

preprocess = transforms.Compose([
    transforms.Resize(144),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def sliding_window_detect(frame, window_size=128, stride=64, threshold=0.7):
    """Slide window across frame to find people"""
    h, w = frame.shape[:2]
    detections = []
    
    for y in range(0, h - window_size, stride):
        for x in range(0, w - window_size, stride):
            # Extract window
            window = frame[y:y+window_size, x:x+window_size]
            
            # Classify window
            pil_img = Image.fromarray(window)
            input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
            
            with torch.inference_mode():
                output = spatial(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                person_conf = probs[1].item()
            
            # If person detected, save bounding box
            if person_conf > threshold:
                detections.append({
                    'box': (x, y, x+window_size, y+window_size),
                    'conf': person_conf
                })
    
    # Non-maximum suppression to remove overlapping boxes
    detections = nms(detections, iou_threshold=0.3)
    return detections

def nms(detections, iou_threshold=0.3):
    """Remove overlapping boxes"""
    if not detections:
        return []
    
    detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
    keep = []
    
    while detections:
        best = detections.pop(0)
        keep.append(best)
        
        detections = [d for d in detections if iou(best['box'], d['box']) < iou_threshold]
    
    return keep

def iou(box1, box2):
    """Calculate intersection over union"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter / (area1 + area2 - inter)

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

time.sleep(2)
print("Search and Rescue - Aerial Detection")
print("Press 'q' to quit, 's' to save")

frame_count = 0

try:
    while True:
        frame = picam2.capture_array()
        
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        
        # Detect people with sliding window
        detections = sliding_window_detect(frame)
        
        display_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        
        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det['box']
            conf = det['conf']
            
            cv.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person {conf:.2f}"
            cv.putText(display_frame, label, (x1, y1-10),
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display count
        person_count = len(detections)
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
            print(f"Saved {filename} - {person_count} person(s)")
            frame_count += 1

except KeyboardInterrupt:
    pass
finally:
    picam2.stop()
    cv.destroyAllWindows()
