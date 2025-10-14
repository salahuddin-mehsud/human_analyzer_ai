# Enhanced configuration settings for the AI Human Analysis System

# Camera settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FPS = 30

# Detection confidence thresholds
FACE_CONFIDENCE = 0.6  # Reduced for better detection
POSE_CONFIDENCE = 0.6
HAND_CONFIDENCE = 0.6
FACE_MESH_CONFIDENCE = 0.6

# Emotion labels
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Enhanced Gesture labels
GESTURES = {
    "thumbs_up": "👍 Thumbs Up",
    "thumbs_down": "👎 Thumbs Down", 
    "victory": "✌️ Victory",
    "pointing": "👆 Pointing",
    "fist": "✊ Fist",
    "open_hand": "🖐️ Open Hand",
    "ok": "👌 OK",
    "rock": "🤘 Rock",
    "three_fingers": "3 Fingers",
    "four_fingers": "4 Fingers",
    "pinch": "🤏 Pinch",
    "unknown": "❓ Unknown"
}

# Facial regions for detailed analysis
FACIAL_REGIONS = {
    "left_eye": "Left Eye",
    "right_eye": "Right Eye", 
    "left_eyebrow": "Left Eyebrow",
    "right_eyebrow": "Right Eyebrow",
    "mouth_outer": "Mouth",
    "mouth_inner": "Lips",
    "nose_tip": "Nose Tip",
    "nose_bridge": "Nose Bridge",
    "face_oval": "Face Outline"
}

# Display colors
COLORS = {
    "face": (0, 255, 0),
    "body": (255, 0, 0),
    "hands": (0, 0, 255),
    "emotion": (255, 255, 0),
    "gesture": (0, 255, 255),
    "face_mesh": (0, 255, 0),
    "facial_regions": (255, 200, 0)
}