import cv2
import os

folder = "dataset"
os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)

def get_saved_images():
    return [f for f in os.listdir(folder) if f.startswith("image_") and f.endswith(".jpg")]

def get_max_index():
    files = get_saved_images()
    if not files:
        return -1
    return max(int(f.split('_')[1].split('.')[0]) for f in files)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    size = min(h, w)
    start_y = (h - size) // 2
    start_x = (w - size) // 2
    
    square_frame = frame[start_y:start_y+size, start_x:start_x+size]
    final_frame = cv2.resize(square_frame, (200, 200))

    display_frame = final_frame.copy()

    count = len(get_saved_images())
    cv2.putText(display_frame, str(count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Data Collection', display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 32:
        idx = get_max_index() + 1
        cv2.imwrite(os.path.join(folder, f"image_{idx}.jpg"), final_frame)
    elif key == ord('r'):
        idx = get_max_index()
        if idx >= 0:
            os.remove(os.path.join(folder, f"image_{idx}.jpg"))
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
