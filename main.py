import cv2
import time
import numpy as np
from datetime import datetime

from body_detector import BodyDetector
from emotion_detector import EmotionDetector
from gesture_recognizer import GestureRecognizer
from video_processor import VideoProcessor

class HumanAnalysisSystem:
    def __init__(self):
        self.body_detector = BodyDetector()
        self.emotion_detector = EmotionDetector()
        self.gesture_recognizer = GestureRecognizer()
        self.video_processor = VideoProcessor()
        
        self.cap = None
        self.is_running = False
        
    def initialize_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return False
            
        # Test camera
        ret, test_frame = self.cap.read()
        if not ret:
            print("Error: Could not read from camera")
            return False
            
        print("Camera initialized successfully")
        return True

    def process_frame(self, frame):
        """Process a single frame for enhanced analysis"""
        try:
            # Detect components with enhanced features - with error handling
            faces = self.body_detector.detect_face(frame)
            pose_landmarks, pose_data = self.body_detector.detect_pose(frame)
            
            # FIXED: Handle hand detection properly to avoid unpacking errors
            hands_result = self.body_detector.detect_hands(frame)
            if hands_result is None:
                hands_data, hand_landmarks = [], None
            else:
                hands_data, hand_landmarks = hands_result
                
            face_meshes = self.body_detector.detect_face_mesh(frame)
            
            # Initialize emotions and gestures lists
            emotions = []
            gestures = []
            
            # Analyze emotions for each face - only process first face
            if face_meshes:
                # Only use the first face mesh to avoid multiple detections
                face_mesh = face_meshes[0]
                x, y, w, h = face_mesh['bbox']
                
                # Ensure ROI coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                
                if w > 0 and h > 0:  # Only process if valid ROI
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Enhanced emotion detection with facial landmarks
                    emotion, confidence = self.emotion_detector.detect_emotion(face_roi, face_mesh)
                    emotions.append((emotion, confidence))
                    
                    # Get facial regions for detailed drawing
                    facial_regions = self.body_detector.get_facial_regions(face_mesh)
                    
                    # Display enhanced emotion info
                    frame = self.emotion_detector.draw_emotion_info(
                        frame, (x, y, w, h), emotion, confidence, facial_regions
                    )
            
            # Fallback for basic face detection if no face mesh found
            elif faces:
                # Only process the first face
                face = faces[0]
                x, y, w, h = face['bbox']
                
                # Ensure ROI coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                
                if w > 0 and h > 0:  # Only process if valid ROI
                    face_roi = frame[y:y+h, x:x+w]
                    
                    emotion, confidence = self.emotion_detector.detect_emotion(face_roi)
                    emotions.append((emotion, confidence))
                    
                    frame = self.emotion_detector.draw_emotion_info(
                        frame, (x, y, w, h), emotion, confidence
                    )
            
            # Analyze gestures for each hand with enhanced recognition
            if hands_data:
                for i, hand in enumerate(hands_data):
                    gesture, confidence = self.gesture_recognizer.recognize_gesture(hand)
                    gestures.append((gesture, confidence))
                    
                    # Display enhanced gesture info
                    if hand and hand['landmarks']:
                        wrist = hand['landmarks'][0]
                        text = f"{gesture} ({confidence:.2f})"
                        cv2.putText(frame, text, (wrist['x'], wrist['y']-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Draw all enhanced detections
            frame = self.body_detector.draw_detections(
                frame, faces, pose_data, hand_landmarks, face_meshes
            )
            
            # Calculate face count properly
            face_count = len(face_meshes) if face_meshes else len(faces)
            pose_detected = pose_landmarks is not None and len(pose_landmarks) > 0
            
            # Add analysis information overlay
            frame = self.video_processor.draw_analysis_info(
                frame, emotions, gestures, pose_detected, face_count
            )
            
            return frame, emotions, gestures
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            # Return the original frame with empty emotions and gestures
            return frame, [], []

    def run(self):
        """Main application loop"""
        if not self.initialize_camera():
            return
            
        self.is_running = True
        print("Starting Human Analysis System...")
        print("Press 'q' to quit, 'r' to start/stop recording, 's' to save screenshot")
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            try:
                processed_frame, emotions, gestures = self.process_frame(frame)
            except Exception as e:
                print(f"Error processing frame: {e}")
                processed_frame = frame
                emotions = []
                gestures = []
            
            # Record frame if recording
            self.video_processor.write_frame(processed_frame)
            
            # Display recording status
            if self.video_processor.recording:
                cv2.putText(processed_frame, "RECORDING", (10, processed_frame.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('AI Human Analysis System', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not self.video_processor.recording:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"output/recordings/analysis_{timestamp}.avi"
                    self.video_processor.start_recording(output_path, 
                                                       (processed_frame.shape[1], processed_frame.shape[0]))
                    print(f"Started recording: {output_path}")
                else:
                    self.video_processor.stop_recording()
                    print("Stopped recording")
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, processed_frame)
                print(f"Screenshot saved: {screenshot_path}")
        
        # Cleanup
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        if self.video_processor.recording:
            self.video_processor.stop_recording()
        cv2.destroyAllWindows()
        print("System shutdown complete")

if __name__ == "__main__":
    system = HumanAnalysisSystem()
    system.run()