import numpy as np
from config.settings import GESTURES

class GestureRecognizer:
    def __init__(self):
        self.gestures = GESTURES
        
    def recognize_gesture(self, hand_data):
        """Improved gesture recognition with better logic"""
        if not hand_data or 'landmarks' not in hand_data:
            return "unknown", 0.0
            
        hand_landmarks = hand_data['landmarks']
        
        if len(hand_landmarks) < 21:
            return "unknown", 0.0
            
        # Get finger states
        finger_states = self._analyze_finger_states(hand_landmarks)
        
        # Analyze gestures
        gesture, confidence = self._analyze_gestures(hand_landmarks, finger_states)
        
        return gesture, confidence
    
    def _analyze_finger_states(self, landmarks):
        """Analyze which fingers are extended - SIMPLIFIED AND CORRECTED"""
        fingers = {}
        
        # Finger tip indices
        thumb_tip = 4
        index_tip = 8
        middle_tip = 12
        ring_tip = 16
        pinky_tip = 20
        
        # Finger MCP joints (base)
        index_mcp = 5
        middle_mcp = 9
        ring_mcp = 13
        pinky_mcp = 17
        
        # SIMPLIFIED FINGER DETECTION
        # A finger is extended if tip is above MCP (lower y value)
        fingers['thumb'] = self._is_thumb_extended(landmarks)
        fingers['index'] = landmarks[index_tip]['y'] < landmarks[index_mcp]['y'] - 10
        fingers['middle'] = landmarks[middle_tip]['y'] < landmarks[middle_mcp]['y'] - 10
        fingers['ring'] = landmarks[ring_tip]['y'] < landmarks[ring_mcp]['y'] - 10
        fingers['pinky'] = landmarks[pinky_tip]['y'] < landmarks[pinky_mcp]['y'] - 10
            
        return fingers
    
    def _is_thumb_extended(self, landmarks):
        """Check if thumb is extended"""
        thumb_tip = 4
        thumb_ip = 3
        thumb_mcp = 2
        
        # Thumb is extended if tip is to the left of IP joint (for right hand)
        return landmarks[thumb_tip]['x'] < landmarks[thumb_ip]['x'] - 5
    
    def _analyze_gestures(self, landmarks, finger_states):
        """CLEANED UP gesture analysis with non-conflicting conditions"""
        thumb = finger_states['thumb']
        index = finger_states['index']
        middle = finger_states['middle']
        ring = finger_states['ring']
        pinky = finger_states['pinky']
        
        extended_fingers = [index, middle, ring, pinky]
        count_extended = sum(extended_fingers)
        
        # REORDERED CONDITIONS - Most specific first
        
        # Fist - NO fingers extended (including thumb)
        if not thumb and count_extended == 0:
            return "fist", 0.95
        
        # Open hand - ALL fingers extended
        elif thumb and count_extended == 4:
            return "open_hand", 0.92
        
        # Victory - ONLY index and middle extended
        elif index and middle and not ring and not pinky:
            # Additional check: ensure fingers are separated
            if self._calculate_distance(landmarks[8], landmarks[12]) > 20:
                return "victory", 0.93
        
        # Pointing - ONLY index extended
        elif index and not middle and not ring and not pinky:
            return "pointing", 0.90
        
        # Thumbs up - ONLY thumb extended
        elif thumb and count_extended == 0:
            return "thumbs_up", 0.94
        
        # Thumbs down - thumb extended with fist (special case)
        elif thumb and count_extended == 0:
            # Check orientation for thumbs down
            wrist = landmarks[0]
            if landmarks[4]['y'] > wrist['y'] + 50:
                return "thumbs_down", 0.94
        
        # Three fingers - index, middle, ring
        elif index and middle and ring and not pinky:
            return "three_fingers", 0.88
        
        # Four fingers - all except thumb
        elif not thumb and count_extended == 4:
            return "four_fingers", 0.86
        
        # OK gesture
        elif self._is_ok_gesture(landmarks):
            return "ok", 0.89
        
        # Rock - thumb and pinky
        elif thumb and pinky and not index and not middle and not ring:
            return "rock", 0.87
        
        # Pinch gesture
        elif self._is_pinching(landmarks):
            return "pinch", 0.85
        
        else:
            return "unknown", 0.3
    
    def _is_ok_gesture(self, landmarks):
        """Check for OK gesture"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        distance = self._calculate_distance(thumb_tip, index_tip)
        return distance < 25
    
    def _is_pinching(self, landmarks):
        """Check for pinching gesture"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        distance = self._calculate_distance(thumb_tip, index_tip)
        return 10 < distance < 35
    
    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1['x'] - point2['x'])**2 + 
                      (point1['y'] - point2['y'])**2)