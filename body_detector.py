import cv2
import mediapipe as mp
import numpy as np

class BodyDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,  # Reduced for better detection
            min_tracking_confidence=0.5
        )
        
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,  # Reduced threshold
            min_tracking_confidence=0.5
        )
        
        self.mp_draw = mp.solutions.drawing_utils
        self.face_mesh_connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
        self.hand_connections = mp.solutions.hands.HAND_CONNECTIONS
        
        # Custom drawing specs
        self.face_mesh_drawing_spec = self.mp_draw.DrawingSpec(
            color=(0, 255, 0), thickness=1, circle_radius=1
        )
        self.hand_drawing_spec = self.mp_draw.DrawingSpec(
            color=(0, 0, 255), thickness=2, circle_radius=2
        )
           
    def detect_face(self, image):
        """Detect faces in the image"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(rgb_image)
            
            faces = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, c = image.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    faces.append({
                        'bbox': (x, y, width, height),
                        'confidence': detection.score[0],
                        'keypoints': detection.location_data.relative_keypoints
                    })
                    
            return faces
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []
    
    def detect_face_mesh(self, image):
        """Detect detailed face mesh with 468 landmarks"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            face_meshes = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = []
                    h, w, c = image.shape
                    
                    for idx, lm in enumerate(face_landmarks.landmark):
                        landmarks.append({
                            'id': idx,
                            'x': int(lm.x * w),
                            'y': int(lm.y * h),
                            'z': lm.z,
                            'visibility': lm.visibility if hasattr(lm, 'visibility') else 1.0
                        })
                    
                    # Calculate bounding box from mesh points
                    if landmarks:
                        xs = [lm['x'] for lm in landmarks]
                        ys = [lm['y'] for lm in landmarks]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        
                        face_meshes.append({
                            'landmarks': landmarks,
                            'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                            'mesh': face_landmarks
                        })
                    
            return face_meshes
        except Exception as e:
            print(f"Error in face mesh detection: {e}")
            return []
    
    def detect_pose(self, image):
        """Detect body pose landmarks"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_image)
            
            landmarks = []
            pose_data = None
            if results.pose_landmarks:
                pose_data = results.pose_landmarks
                for lm in results.pose_landmarks.landmark:
                    h, w, c = image.shape
                    landmarks.append({
                        'x': int(lm.x * w),
                        'y': int(lm.y * h),
                        'z': lm.z,
                        'visibility': lm.visibility
                    })
                    
            return landmarks, pose_data
        except Exception as e:
            print(f"Error in pose detection: {e}")
            return [], None
    
    def detect_hands(self, image):
        """Detect detailed hand landmarks with 21 points per hand - FIXED"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            
            hands_data = []
            hand_landmarks_list = []
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    h, w, c = image.shape
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        landmarks.append({
                            'id': idx,
                            'x': int(lm.x * w),
                            'y': int(lm.y * h),
                            'z': lm.z
                        })
                    hands_data.append({
                        'landmarks': landmarks,
                        'mesh': hand_landmarks
                    })
                    hand_landmarks_list.append(hand_landmarks)
                    
            # ALWAYS return a tuple with two values to prevent unpacking errors
            return hands_data, hand_landmarks_list if hand_landmarks_list else None
        except Exception as e:
            print(f"Error in hand detection: {e}")
            return [], None  # Always return a tuple
    
    def get_facial_regions(self, face_mesh):
        """Extract specific facial regions for detailed analysis with CORRECT indices"""
        if not face_mesh or 'landmarks' not in face_mesh:
            return {}
            
        landmarks = face_mesh['landmarks']
        
        # Correct MediaPipe Face Mesh indices
        regions = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'right_eyebrow': [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
            'mouth_outer': [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
            'mouth_inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191],
            'nose_tip': [1, 2, 98, 327],
            'nose_bridge': [168, 6, 197, 195, 5, 4],
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        }
        
        facial_regions = {}
        for region_name, indices in regions.items():
            points = []
            for idx in indices:
                if idx < len(landmarks):
                    points.append((landmarks[idx]['x'], landmarks[idx]['y']))
            facial_regions[region_name] = points
            
        return facial_regions
    
    def draw_detections(self, image, faces, pose_landmarks, hand_landmarks_list, face_meshes=None):
        """Draw all detections on the image with enhanced visualization"""
        # Draw face meshes with detailed landmarks
        if face_meshes:
            for face_mesh in face_meshes:
                if 'mesh' in face_mesh:
                    self.mp_draw.draw_landmarks(
                        image=image,
                        landmark_list=face_mesh['mesh'],
                        connections=self.face_mesh_connections,
                        landmark_drawing_spec=self.face_mesh_drawing_spec,
                        connection_drawing_spec=self.face_mesh_drawing_spec
                    )
                    
                    # Draw facial regions
                    facial_regions = self.get_facial_regions(face_mesh)
                    for region_name, points in facial_regions.items():
                        if len(points) > 2:
                            pts = np.array(points, np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            cv2.polylines(image, [pts], isClosed=True, 
                                        color=(255, 255, 0), thickness=1)
        
        # Draw basic face bounding boxes
        for face in faces:
            x, y, w, h = face['bbox']
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        # Draw pose with enhanced visualization
        if pose_landmarks:
            self.mp_draw.draw_landmarks(
                image, pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3)
            )
            
        # Draw hands with detailed landmarks
        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list:
                self.mp_draw.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3)
                )
                
        return image