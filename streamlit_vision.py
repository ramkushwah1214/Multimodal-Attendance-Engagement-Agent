import os
import sys
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.vision.detector import PersonDetector
from app.vision.face_recognizer import LocalFaceRecognizer
from app.vision.head_pose import HeadPoseEstimator
from streamlit_utils import get_db_connection

class StreamlitVisionEngine:
    def __init__(self):
        self.detector = PersonDetector(confidence_threshold=0.45)
        self.face_recognizer = LocalFaceRecognizer(similarity_threshold=0.65)
        self.head_pose = HeadPoseEstimator()

    def load_enrolled_faces(self) -> List[Tuple[int, str, np.ndarray]]:
        """Loads all consented students with valid embedding files."""
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""
        SELECT id, full_name, embedding_path 
        FROM students 
        WHERE has_consented = 1 AND embedding_path IS NOT NULL
        """)
        rows = c.fetchall()
        conn.close()

        cache = []
        for s_id, s_name, path in rows:
            if path and os.path.exists(path):
                try:
                    vec = np.load(path)
                    cache.append((s_id, s_name, vec))
                except Exception:
                    pass
        return cache

    def process_snapshot(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Processes a single BGR image:
        - Detects persons
        - Detects faces within person crops
        - Estimates 3D head pose Euler angles
        - Matches against consented student database
        - Annotates frame with HUD telemetry
        """
        annotated = image_bgr.copy()
        h, w = annotated.shape[:2]

        person_boxes = self.detector.detect(image_bgr)
        enrolled = self.load_enrolled_faces()

        detections = []

        # If no full persons detected, check whole frame for faces directly
        if len(person_boxes) == 0:
            person_boxes = [(0, 0, w, h, 0.5)]

        for p_idx, (px1, py1, px2, py2, pconf) in enumerate(person_boxes):
            person_crop = image_bgr[py1:py2, px1:px2]
            if person_crop.size == 0:
                continue

            # Draw person box (thin cyan)
            if (px2 - px1) < w or (py2 - py1) < h:
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 180, 0), 1)

            # Detect face
            face_crop = self.face_recognizer.detect_face(person_crop)
            is_forward = True
            yaw, pitch, roll = 0.0, 0.0, 0.0

            matched_id = None
            matched_name = None
            match_score = 0.0

            if face_crop is not None and face_crop.size > 0:
                # 3D Head pose
                is_forward, yaw, pitch, roll = self.head_pose.estimate_orientation(face_crop)

                # Face embedding & match
                emb = self.face_recognizer.extract_embedding(face_crop)
                if emb is not None and enrolled:
                    matched_id, matched_name, match_score = self.face_recognizer.match_face(emb, enrolled)

            # Display name and status
            status_text = matched_name if matched_name else "Unknown / Unverified"
            color = (0, 220, 100) if matched_name else (0, 165, 255)
            pose_status = "Forward Focus" if is_forward else "Distracted / Turned"
            pose_color = (0, 220, 100) if is_forward else (0, 140, 255)

            # Draw badge background
            label_y = max(24, py1 - 8)
            cv2.putText(annotated, f"{status_text} ({match_score:.0%})", (px1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(annotated, f"Pose: {pose_status} | Yaw: {yaw:.0f} deg", (px1, label_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, pose_color, 1)

            detections.append({
                "student_id": matched_id,
                "student_name": matched_name or "Unverified",
                "confidence": match_score,
                "is_forward": is_forward,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "box": (px1, py1, px2, py2)
            })

        # Return RGB for Streamlit rendering
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return annotated_rgb, detections
