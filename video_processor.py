import cv2
import numpy as np
from datetime import datetime

class VideoProcessor:
    def __init__(self):
        self.recording = False
        self.video_writer = None
        
    def start_recording(self, output_path, frame_size, fps=30):
        """Start recording video"""
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.recording = True
        
    def stop_recording(self):
        """Stop recording video"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        
    def write_frame(self, frame):
        """Write frame to video if recording"""
        if self.recording and self.video_writer:
            self.video_writer.write(frame)
            
    def draw_analysis_info(self, image, emotions, gestures, pose_detected, face_count):
        """Draw analysis information on the image"""
        h, w = image.shape[:2]
        
        # Create overlay for information
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        # Display information
        y_offset = 40
        line_height = 30
        
        # Emotions
        for i, (emotion, confidence) in enumerate(emotions):
            text = f"Face {i+1}: {emotion} ({confidence:.2f})"
            cv2.putText(image, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += line_height
            
        # Gestures
        for i, (gesture, confidence) in enumerate(gestures):
            text = f"Hand {i+1}: {gesture} ({confidence:.2f})"
            cv2.putText(image, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += line_height
            
        # Body detection status
        body_status = "Body: Detected" if pose_detected else "Body: Not Detected"
        cv2.putText(image, body_status, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        y_offset += line_height
        
        # Face count
        cv2.putText(image, f"Faces: {face_count}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return image