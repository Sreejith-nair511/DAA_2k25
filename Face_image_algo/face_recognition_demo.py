import cv2
import numpy as np
import os
from datetime import datetime

class FaceDetectionSystem:
    def __init__(self):
        # Load the pre-trained face detection classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load the eye cascade classifier for better accuracy
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        if self.face_cascade.empty():
            raise ValueError("Error: Could not load face cascade classifier")
        if self.eye_cascade.empty():
            raise ValueError("Error: Could not load eye cascade classifier")

    def detect_faces(self, frame):
        """Detect faces in the given frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Get the face region for eye detection
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes within the face region
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            
            # Add label
            cv2.putText(frame, 
                       f'Face detected', 
                       (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, 
                       (0, 255, 0), 
                       2)
        
        # Add count of faces detected
        cv2.putText(frame,
                   f'Faces detected: {len(faces)}',
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1,
                   (0, 255, 0),
                   2)
        
        return frame

    def save_snapshot(self, frame):
        """Save a snapshot when a face is detected"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"face_detection_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved: {filename}")

    def run_webcam(self):
        """Run face detection on webcam feed"""
        video_capture = cv2.VideoCapture(0)
        
        # Check if webcam opened successfully
        if not video_capture.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting webcam feed...")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save a snapshot")
        
        while True:
            # Read frame from webcam
            ret, frame = video_capture.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            processed_frame = self.detect_faces(frame)
            
            # Display result
            cv2.imshow('Face Detection', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_snapshot(processed_frame)
        
        # Clean up
        video_capture.release()
        cv2.destroyAllWindows()

def main():
    try:
        # Initialize and run system
        face_system = FaceDetectionSystem()
        face_system.run_webcam()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()