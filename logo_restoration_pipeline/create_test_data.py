import cv2
import numpy as np
import os

def create_dummy_data():
    # Create directories
    os.makedirs("input", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    
    # 1. Create a dummy "Car" image (Input)
    # Gray background with a darker rectangle representing a "grille" or feature
    img = np.full((640, 640, 3), 200, dtype=np.uint8)
    # Draw a "hood" area
    cv2.rectangle(img, (100, 100), (540, 540), (180, 180, 180), -1)
    # Draw a "logo" placeholder (black circle) that YOLO might detect if we are lucky, 
    # but for the mock test we might need to mock detection too if YOLO isn't trained.
    # Let's draw a simple shape.
    cv2.circle(img, (320, 320), 50, (50, 50, 50), -1)
    
    cv2.imwrite("input/test_car_bmw.jpg", img)
    print("Created input/test_car_bmw.jpg")
    
    # 2. Create a dummy "BMW" Logo (Asset)
    # Blue and white circle
    logo = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(logo, (100, 100), 100, (255, 255, 255), -1) # White base
    cv2.ellipse(logo, (100, 100), (100, 100), 0, 0, 90, (255, 0, 0), -1) # Blue quadrant
    cv2.ellipse(logo, (100, 100), (100, 100), 0, 180, 270, (255, 0, 0), -1) # Blue quadrant
    
    cv2.imwrite("assets/bmw_logo.png", logo)
    print("Created assets/bmw_logo.png")

if __name__ == "__main__":
    create_dummy_data()
