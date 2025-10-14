import cv2
import numpy as np
from collections import deque
from config.settings import EMOTIONS

class EmotionDetector:
    def __init__(self):
        self.emotion_labels = EMOTIONS
        self.emotion_history = deque(maxlen=10)
        print("Enhanced Emotion Detector Initialized")
        
    def detect_emotion(self, face_roi, face_mesh=None):
        """Enhanced emotion detection with better facial analysis"""
        if face_mesh and 'landmarks' in face_mesh and len(face_mesh['landmarks']) >= 468:
            emotion, confidence = self._detect_emotion_enhanced(face_mesh)
        else:
            # Fallback to basic detection
            emotion, confidence = self._detect_from_roi(face_roi)
        
        # Add to history for smoothing
        self.emotion_history.append((emotion, confidence))
        
        # Return smoothed emotion
        return self._get_smoothed_emotion()
    
    def _detect_emotion_enhanced(self, face_mesh):
        """Enhanced emotion detection using multiple facial features"""
        landmarks = face_mesh['landmarks']
        
        try:
            # Calculate multiple facial features
            smile_ratio = self._get_smile_ratio(landmarks)
            brow_raise = self._get_brow_raise(landmarks)
            eye_openness = self._get_eye_openness_enhanced(landmarks)
            mouth_openness = self._get_mouth_openness_enhanced(landmarks)
            jaw_drop = self._get_jaw_drop_enhanced(landmarks)
            
            # Debug print (uncomment to see feature values)
            # print(f"Smile: {smile_ratio:.2f}, Brow: {brow_raise:.2f}, Eye: {eye_openness:.2f}, Mouth: {mouth_openness:.2f}")
            
            # Enhanced emotion decision logic
            if smile_ratio > 0.15:  # Smiling
                if mouth_openness > 0.25:  # Big smile with open mouth
                    return "happy", min(0.8 + smile_ratio, 0.95)
                else:  # Gentle smile
                    return "happy", min(0.7 + smile_ratio, 0.90)
                    
            elif mouth_openness > 0.35 and eye_openness > 0.8:  # Surprised
                return "surprise", 0.85
                
            elif brow_raise > 0.6 and smile_ratio < 0.1:  # Angry
                return "angry", 0.80
                
            elif brow_raise > 0.3 and smile_ratio < 0.05 and eye_openness < 0.6:  # Sad
                return "sad", 0.75
                
            elif jaw_drop > 0.4 and brow_raise > 0.4:  # Fear
                return "fear", 0.70
                
            elif mouth_openness < 0.15 and 0.4 < eye_openness < 0.8:  # Neutral
                return "neutral", 0.85
                
            else:
                return "neutral", 0.6
                
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return "neutral", 0.5
    
    def _get_smile_ratio(self, landmarks):
        """Calculate smile intensity using mouth corner movements"""
        try:
            # Mouth corners
            left_corner = landmarks[61]
            right_corner = landmarks[291]
            
            # Mouth center top and bottom
            mouth_center_top = landmarks[13]
            mouth_center_bottom = landmarks[14]
            
            # Calculate mouth width and height
            mouth_width = abs(right_corner['x'] - left_corner['x'])
            mouth_height = abs(mouth_center_bottom['y'] - mouth_center_top['y'])
            
            # In a smile, width increases and corners move upward
            face_width = self._get_face_width(landmarks)
            normal_mouth_ratio = 0.4  # Typical mouth width to face width ratio
            
            current_ratio = mouth_width / face_width
            smile_intensity = max(0, (current_ratio - normal_mouth_ratio) / (0.6 - normal_mouth_ratio))
            
            return min(smile_intensity, 1.0)
        except:
            return 0.0
    
    def _get_brow_raise(self, landmarks):
        """Calculate brow tension/raising"""
        try:
            # Left brow points
            left_brow_inner = landmarks[65]
            left_brow_outer = landmarks[55]
            
            # Right brow points  
            right_brow_inner = landmarks[295]
            right_brow_outer = landmarks[285]
            
            # Eye reference points
            left_eye_top = landmarks[159]
            right_eye_top = landmarks[386]
            
            # Calculate average brow position relative to eyes
            left_brow_height = abs(left_brow_inner['y'] - left_eye_top['y'])
            right_brow_height = abs(right_brow_inner['y'] - right_eye_top['y'])
            
            face_height = self._get_face_height(landmarks)
            brow_ratio = (left_brow_height + right_brow_height) / (face_height * 0.1)
            
            return min(brow_ratio, 1.0)
        except:
            return 0.0
    
    def _get_eye_openness_enhanced(self, landmarks):
        """Enhanced eye openness calculation"""
        try:
            # Left eye vertical points
            left_eye_top = landmarks[159]
            left_eye_bottom = landmarks[145]
            left_eye_open = abs(left_eye_top['y'] - left_eye_bottom['y'])
            
            # Right eye vertical points
            right_eye_top = landmarks[386]
            right_eye_bottom = landmarks[374]
            right_eye_open = abs(right_eye_top['y'] - right_eye_bottom['y'])
            
            # Eye horizontal reference for normalization
            left_eye_left = landmarks[33]
            left_eye_right = landmarks[133]
            left_eye_width = abs(left_eye_right['x'] - left_eye_left['x'])
            
            right_eye_left = landmarks[362]
            right_eye_right = landmarks[263]
            right_eye_width = abs(right_eye_right['x'] - right_eye_left['x'])
            
            # Calculate openness ratios
            left_ratio = left_eye_open / left_eye_width
            right_ratio = right_eye_open / right_eye_width
            
            avg_ratio = (left_ratio + right_ratio) / 2
            
            return min(avg_ratio * 3, 1.0)  # Scale for better sensitivity
        except:
            return 0.5
    
    def _get_mouth_openness_enhanced(self, landmarks):
        """Enhanced mouth openness calculation"""
        try:
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            
            openness = abs(upper_lip['y'] - lower_lip['y'])
            face_height = self._get_face_height(landmarks)
            
            return min(openness / (face_height * 0.08), 1.0)
        except:
            return 0.0
    
    def _get_jaw_drop_enhanced(self, landmarks):
        """Enhanced jaw drop detection"""
        try:
            chin = landmarks[152]
            nose_tip = landmarks[1]
            
            jaw_drop = abs(chin['y'] - nose_tip['y'])
            face_height = self._get_face_height(landmarks)
            
            return min(jaw_drop / (face_height * 0.5), 1.0)
        except:
            return 0.0
    
    def _get_face_width(self, landmarks):
        """Get face width for normalization"""
        try:
            left_face = landmarks[234]
            right_face = landmarks[454]
            return max(abs(right_face['x'] - left_face['x']), 50)
        except:
            return 100
    
    def _get_face_height(self, landmarks):
        """Get face height for normalization"""
        try:
            chin = landmarks[152]
            forehead = landmarks[10]
            return max(abs(chin['y'] - forehead['y']), 50)
        except:
            return 100
    
    def _detect_from_roi(self, face_roi):
        """Basic emotion detection from face ROI (fallback)"""
        # Simple brightness-based detection as fallback
        if face_roi.size > 0:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            if avg_brightness > 150:  # Bright face - likely smiling
                return "happy", 0.7
            else:
                return "neutral", 0.6
        return "neutral", 0.5
    
    def _get_smoothed_emotion(self):
        """Get smoothed emotion from history"""
        if not self.emotion_history:
            return "neutral", 0.5
            
        # Use weighted average of recent emotions
        recent = list(self.emotion_history)[-5:]  # Last 5 frames
        emotions = [e for e, c in recent]
        confidences = [c for e, c in recent]
        
        if emotions:
            most_common = max(set(emotions), key=emotions.count)
            avg_confidence = np.mean([c for e, c in recent if e == most_common])
            return most_common, min(avg_confidence, 0.95)
        
        return "neutral", 0.5
    
    def draw_emotion_info(self, image, face_bbox, emotion, confidence, facial_regions=None):
        """Draw emotion information on the image"""
        x, y, w, h = face_bbox
        
        color_map = {
            "happy": (0, 255, 0),      # Green
            "sad": (255, 0, 0),        # Blue  
            "angry": (0, 0, 255),      # Red
            "surprise": (0, 255, 255), # Yellow
            "neutral": (255, 255, 255),# White
            "fear": (255, 0, 255),     # Purple
            "disgust": (0, 128, 128)   # Teal
        }
        
        color = color_map.get(emotion, (255, 255, 255))
        
        # Draw emotion text
        text = f"{emotion.upper()} ({confidence:.2f})"
        cv2.putText(image, text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw confidence bar
        bar_width = 120
        bar_height = 8
        bar_x = x
        bar_y = y - 25
        
        cv2.rectangle(image, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        conf_width = int(bar_width * confidence)
        if confidence > 0.7:
            bar_color = (0, 255, 0)
        elif confidence > 0.5:
            bar_color = (0, 165, 255)
        else:
            bar_color = (0, 0, 255)
        
        cv2.rectangle(image, (bar_x, bar_y), 
                     (bar_x + conf_width, bar_y + bar_height), bar_color, -1)
        
        return image