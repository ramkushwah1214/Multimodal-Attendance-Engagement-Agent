import os
import sys
import sqlite3
import datetime
import uuid
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

# Add backend directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Directories
DATA_DIR = os.path.join(BACKEND_DIR, "data")
DB_DIR = os.path.join(DATA_DIR, "db")
DB_PATH = os.path.join(DB_DIR, "attendance.db")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")

for d in [DATA_DIR, DB_DIR, EMBEDDINGS_DIR, UPLOADS_DIR]:
    os.makedirs(d, exist_ok=True)


def get_db_connection():
    """Returns a standard sqlite3 connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_streamlit_db():
    """Initializes SQLite database tables synchronously."""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.executescript("""
    CREATE TABLE IF NOT EXISTS courses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        instructor_name TEXT NOT NULL,
        room_number TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        course_name TEXT NOT NULL,
        email TEXT,
        profile_image_path TEXT,
        is_archived BOOLEAN DEFAULT 0,
        has_consented BOOLEAN DEFAULT 0,
        consent_date TIMESTAMP,
        embedding_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS consent_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        consent_action TEXT NOT NULL,
        notes TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_uid TEXT UNIQUE NOT NULL,
        course_id INTEGER,
        course_title TEXT NOT NULL,
        instructor_name TEXT NOT NULL,
        room_number TEXT,
        target_class TEXT NOT NULL,
        expected_duration_minutes INTEGER DEFAULT 60,
        status TEXT DEFAULT 'SCHEDULED',
        start_time TIMESTAMP,
        end_time TIMESTAMP,
        actual_duration_seconds REAL DEFAULT 0.0,
        source_type TEXT DEFAULT 'LIVE',
        video_filename TEXT,
        video_duration_seconds REAL DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(course_id) REFERENCES courses(id)
    );

    CREATE TABLE IF NOT EXISTS attendance_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        student_id INTEGER,
        tracking_track_id INTEGER,
        detected_name TEXT,
        status TEXT DEFAULT 'PENDING',
        first_seen_seconds REAL,
        last_seen_seconds REAL,
        sustained_duration_seconds REAL DEFAULT 0.0,
        observation_count INTEGER DEFAULT 0,
        avg_confidence REAL DEFAULT 0.0,
        is_manual_override BOOLEAN DEFAULT 0,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
        FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS engagement_windows (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        window_index INTEGER NOT NULL,
        start_seconds REAL NOT NULL,
        end_seconds REAL NOT NULL,
        person_count_avg REAL DEFAULT 0.0,
        forward_facing_ratio REAL DEFAULT 0.0,
        movement_intensity REAL DEFAULT 0.0,
        prolonged_inactivity_ratio REAL DEFAULT 0.0,
        speech_active_ratio REAL DEFAULT 0.0,
        interaction_activity_score REAL DEFAULT 0.0,
        is_audio_available BOOLEAN DEFAULT 1,
        is_video_available BOOLEAN DEFAULT 1,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS teaching_suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        suggestion_type TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        evidence TEXT,
        urgency TEXT DEFAULT 'MEDIUM',
        generated_by TEXT DEFAULT 'RULE_BASED',
        acknowledged BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
    );
    """)
    conn.commit()
    conn.close()


def seed_demo_classroom(force: bool = False):
    """Seeds realistic demo students, past sessions, attendance, and analytics."""
    init_streamlit_db()
    conn = get_db_connection()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM students")
    student_count = c.fetchone()[0]

    if student_count > 0 and not force:
        conn.close()
        return False

    if force:
        c.executescript("""
        DELETE FROM teaching_suggestions;
        DELETE FROM engagement_windows;
        DELETE FROM attendance_records;
        DELETE FROM consent_logs;
        DELETE FROM sessions;
        DELETE FROM students;
        DELETE FROM courses;
        """)

    # 1. Insert Course
    c.execute("""
    INSERT INTO courses (course_code, name, instructor_name, room_number)
    VALUES ('CS-101', 'Intro to Algorithms & Data Structures', 'Dr. Ada Lovelace', 'Hall 302')
    """)
    course_id = c.lastrowid

    # 2. Insert Students
    students_data = [
        ("CS-2024-001", "Emma Watson", "Intro to Algorithms & Data Structures", "emma.w@university.edu"),
        ("CS-2024-002", "Liam Smith", "Intro to Algorithms & Data Structures", "liam.s@university.edu"),
        ("CS-2024-003", "Sophia Garcia", "Intro to Algorithms & Data Structures", "sophia.g@university.edu"),
        ("CS-2024-004", "Noah Patel", "Intro to Algorithms & Data Structures", "noah.p@university.edu"),
        ("CS-2024-005", "Ava Chen", "Intro to Algorithms & Data Structures", "ava.c@university.edu"),
        ("CS-2024-006", "Marcus Aurelius", "Intro to Algorithms & Data Structures", "marcus.a@university.edu"),
    ]

    student_ids = []
    now = datetime.datetime.utcnow()
    for sid, name, course, email in students_data:
        c.execute("""
        INSERT INTO students (student_id, full_name, course_name, email, has_consented, consent_date, is_archived)
        VALUES (?, ?, ?, ?, 1, ?, 0)
        """, (sid, name, course, email, now))
        s_db_id = c.lastrowid
        student_ids.append((s_db_id, sid, name))

        # Log consent
        c.execute("""
        INSERT INTO consent_logs (student_id, consent_action, notes)
        VALUES (?, 'GRANTED', 'Biometric consent documented on record for automated classroom attendance.')
        """, (s_db_id,))

        # Save synthetic embedding
        vec = np.random.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        emb_path = os.path.join(EMBEDDINGS_DIR, f"student_{s_db_id}_{sid}.npy")
        np.save(emb_path, vec)

        c.execute("UPDATE students SET embedding_path = ? WHERE id = ?", (emb_path, s_db_id))

    # 3. Insert Sessions
    sessions_data = [
        ("SESS-ALGO-01", "Lecture 01: Asymptotic Complexity & Big-O", "COMPLETED", 3, 50, 2980.0),
        ("SESS-ALGO-02", "Lecture 02: Divide & Conquer Strategies", "COMPLETED", 2, 50, 3050.0),
        ("SESS-ALGO-03", "Lecture 03: Dynamic Programming Core Patterns", "COMPLETED", 1, 55, 3300.0),
        ("SESS-ALGO-04", "Lecture 04: Graph Traversal & Shortest Paths", "LIVE", 0, 60, 1850.0),
    ]

    status_cycle = ["PRESENT", "PRESENT", "LATE", "PRESENT", "PRESENT", "LEFT_EARLY"]

    for sess_uid, title, status, days_ago, exp_dur, act_dur in sessions_data:
        sess_start = now - datetime.timedelta(days=days_ago, hours=2)
        sess_end = sess_start + datetime.timedelta(seconds=act_dur) if status == "COMPLETED" else None

        c.execute("""
        INSERT INTO sessions (
            session_uid, course_id, course_title, instructor_name, room_number,
            target_class, expected_duration_minutes, status, start_time, end_time,
            actual_duration_seconds, source_type
        ) VALUES (?, ?, ?, 'Dr. Ada Lovelace', 'Hall 302', 'CS-101', ?, ?, ?, ?, ?, 'LIVE')
        """, (sess_uid, course_id, title, exp_dur, status, sess_start, sess_end, act_dur))
        sess_id = c.lastrowid

        # Insert Attendance
        for idx, (s_db_id, sid, s_name) in enumerate(student_ids):
            st = status_cycle[(idx + days_ago) % len(status_cycle)]
            first_seen = 25.0 if st != "LATE" else 750.0
            last_seen = act_dur - 40.0 if st != "LEFT_EARLY" else act_dur * 0.45
            sustained = max(100.0, last_seen - first_seen)
            obs = 42 if st == "PRESENT" else (22 if st in ["LATE", "LEFT_EARLY"] else 4)

            c.execute("""
            INSERT INTO attendance_records (
                session_id, student_id, detected_name, status,
                first_seen_seconds, last_seen_seconds, sustained_duration_seconds,
                observation_count, avg_confidence, is_manual_override, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0.91, 0, 'Biometric facial verification sustained across session.')
            """, (sess_id, s_db_id, s_name, st, first_seen, last_seen, sustained, obs))

        # Insert Engagement Windows (30s intervals)
        num_windows = min(20, max(6, int(act_dur // 30)))
        base_score = 72.0 + (days_ago * 4.0)
        for w in range(num_windows):
            start_s = w * 30.0
            end_s = start_s + 30.0
            
            # Simulated realistic classroom fluctuations
            variation = np.sin(w / 2.5) * 12.0 - (8.0 if w in [6, 7, 8] else 0.0)
            score = float(np.clip(base_score + variation, 45.0, 96.0))
            fwd_ratio = float(np.clip(0.75 + np.sin(w / 3.0) * 0.18 - (0.25 if w in [6, 7] else 0.0), 0.20, 0.98))
            motion = float(np.clip(0.22 + np.cos(w / 2.0) * 0.15, 0.05, 0.85))
            speech = float(np.clip(0.35 + np.sin(w / 1.5) * 0.25, 0.02, 0.88))

            c.execute("""
            INSERT INTO engagement_windows (
                session_id, window_index, start_seconds, end_seconds,
                person_count_avg, forward_facing_ratio, movement_intensity,
                speech_active_ratio, interaction_activity_score, is_audio_available, is_video_available
            ) VALUES (?, ?, ?, ?, 5.8, ?, ?, ?, ?, 1, 1)
            """, (sess_id, w, start_s, end_s, fwd_ratio, motion, speech, score))

        # Insert Teaching Suggestions
        c.execute("""
        INSERT INTO teaching_suggestions (
            session_id, suggestion_type, title, description, evidence, urgency, generated_by
        ) VALUES
        (?, 'ENGAGEMENT_DROP', 'Attention Dip Detected in Mid-Lecture Interval',
         'Forward-facing orientation decreased by 34% between minutes 15-20. Recommended: Initiate a 2-minute interactive think-pair-share or quick concept poll.',
         'Forward pose fell to 32% during window index 6-8 while motion spiked.', 'HIGH', 'RULE_BASED'),
        (?, 'LOW_AUDIO', 'Extended Monologue Period / Low Speech Activity',
         'Continuous lecture delivery without student verbal interactions recorded for 18 consecutive minutes. Recommended: Pause for open classroom Q&A or prompt a student to explain the last code snippet.',
         'Speech active ratio remained under 8% over 12 windows.', 'MEDIUM', 'RULE_BASED')
        """, (sess_id, sess_id))

    conn.commit()
    conn.close()
    return True


def get_dashboard_metrics() -> Dict:
    """Retrieves top-level aggregated KPI telemetry."""
    init_streamlit_db()
    conn = get_db_connection()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM students WHERE COALESCE(is_archived, 0) = 0")
    total_students = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM sessions")
    total_sessions = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM sessions WHERE status = 'LIVE'")
    live_sessions = c.fetchone()[0]

    # Attendance percentage
    c.execute("""
    SELECT 
        COUNT(*),
        SUM(CASE WHEN status IN ('PRESENT', 'LATE') THEN 1 ELSE 0 END)
    FROM attendance_records
    """)
    row = c.fetchone()
    total_att = row[0] or 0
    present_att = row[1] or 0
    att_rate = round((present_att / total_att * 100), 1) if total_att > 0 else 0.0

    # Average interaction score
    c.execute("SELECT AVG(interaction_activity_score) FROM engagement_windows")
    avg_score_row = c.fetchone()[0]
    avg_interaction = round(avg_score_row, 1) if avg_score_row else 0.0

    # Status distribution
    c.execute("""
    SELECT status, COUNT(*) 
    FROM attendance_records 
    GROUP BY status
    """)
    status_dist = {r[0]: r[1] for r in c.fetchall()}

    conn.close()
    return {
        "total_students": total_students,
        "total_sessions": total_sessions,
        "live_sessions": live_sessions,
        "attendance_rate": att_rate,
        "avg_interaction": avg_interaction,
        "status_distribution": status_dist
    }


def get_sessions_df() -> pd.DataFrame:
    """Returns a DataFrame of all sessions with summarized stats."""
    init_streamlit_db()
    conn = get_db_connection()
    query = """
    SELECT 
        s.id,
        s.session_uid,
        s.course_title,
        s.instructor_name,
        s.room_number,
        s.status,
        s.expected_duration_minutes,
        s.actual_duration_seconds,
        s.created_at,
        COUNT(a.id) as total_attendance,
        SUM(CASE WHEN a.status IN ('PRESENT', 'LATE') THEN 1 ELSE 0 END) as present_count,
        ROUND(AVG(w.interaction_activity_score), 1) as avg_score
    FROM sessions s
    LEFT JOIN attendance_records a ON s.id = a.session_id
    LEFT JOIN engagement_windows w ON s.id = w.session_id
    GROUP BY s.id
    ORDER BY s.id DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_students_df() -> pd.DataFrame:
    """Returns a DataFrame of all enrolled students."""
    init_streamlit_db()
    conn = get_db_connection()
    query = """
    SELECT 
        id,
        student_id,
        full_name,
        course_name,
        email,
        has_consented,
        consent_date,
        CASE WHEN embedding_path IS NOT NULL THEN 'Verified Vector' ELSE 'Missing' END as embedding_status,
        created_at
    FROM students
    WHERE COALESCE(is_archived, 0) = 0
    ORDER BY id ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def add_student_record(student_id: str, full_name: str, course_name: str, email: str, consent: bool) -> int:
    """Registers a new student and logs consent."""
    init_streamlit_db()
    conn = get_db_connection()
    c = conn.cursor()
    now = datetime.datetime.utcnow()
    c.execute("""
    INSERT INTO students (student_id, full_name, course_name, email, has_consented, consent_date)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (student_id, full_name, course_name, email, 1 if consent else 0, now if consent else None))
    new_id = c.lastrowid

    c.execute("""
    INSERT INTO consent_logs (student_id, consent_action, notes)
    VALUES (?, ?, ?)
    """, (new_id, 'GRANTED' if consent else 'REVOKED', 'Student registration consent recorded.'))

    conn.commit()
    conn.close()
    return new_id


def update_student_consent(student_db_id: int, consent: bool):
    """Updates biometric consent status for a student."""
    conn = get_db_connection()
    c = conn.cursor()
    now = datetime.datetime.utcnow()
    c.execute("""
    UPDATE students 
    SET has_consented = ?, consent_date = ?, updated_at = ?
    WHERE id = ?
    """, (1 if consent else 0, now if consent else None, now, student_db_id))

    c.execute("""
    INSERT INTO consent_logs (student_id, consent_action, notes)
    VALUES (?, ?, ?)
    """, (student_db_id, 'GRANTED' if consent else 'REVOKED', 'Manual consent status updated via dashboard.'))

    conn.commit()
    conn.close()


def save_student_embedding(student_db_id: int, student_id_code: str, vector: np.ndarray) -> str:
    """Saves biometric 512-dim embedding securely to local .npy file."""
    emb_path = os.path.join(EMBEDDINGS_DIR, f"student_{student_db_id}_{student_id_code}.npy")
    np.save(emb_path, vector)

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE students SET embedding_path = ? WHERE id = ?", (emb_path, student_db_id))
    conn.commit()
    conn.close()
    return emb_path


def get_session_attendance_df(session_id: int) -> pd.DataFrame:
    """Returns attendance records for a specific session."""
    init_streamlit_db()
    conn = get_db_connection()
    query = """
    SELECT 
        a.id as record_id,
        s.student_id,
        COALESCE(s.full_name, a.detected_name, 'Unidentified Person') as full_name,
        a.status,
        ROUND(a.first_seen_seconds, 1) as first_seen_s,
        ROUND(a.last_seen_seconds, 1) as last_seen_s,
        ROUND(a.sustained_duration_seconds, 1) as sustained_s,
        a.observation_count,
        ROUND(a.avg_confidence, 2) as confidence,
        a.is_manual_override,
        a.notes
    FROM attendance_records a
    LEFT JOIN students s ON a.student_id = s.id
    WHERE a.session_id = ?
    ORDER BY a.id ASC
    """
    df = pd.read_sql_query(query, conn, params=(session_id,))
    conn.close()
    return df


def update_attendance_override(record_id: int, new_status: str, audit_notes: str):
    """Applies instructor manual override with audit justification."""
    conn = get_db_connection()
    c = conn.cursor()
    now = datetime.datetime.utcnow()
    c.execute("""
    UPDATE attendance_records
    SET status = ?, is_manual_override = 1, notes = ?, updated_at = ?
    WHERE id = ?
    """, (new_status, audit_notes, now, record_id))
    conn.commit()
    conn.close()


def get_session_windows_df(session_id: int) -> pd.DataFrame:
    """Returns 30-sec telemetry window metrics for a session."""
    init_streamlit_db()
    conn = get_db_connection()
    query = """
    SELECT 
        window_index,
        start_seconds,
        end_seconds,
        person_count_avg,
        ROUND(forward_facing_ratio * 100, 1) as forward_facing_pct,
        ROUND(movement_intensity * 100, 1) as movement_intensity_pct,
        ROUND(speech_active_ratio * 100, 1) as speech_active_pct,
        ROUND(interaction_activity_score, 1) as interaction_score
    FROM engagement_windows
    WHERE session_id = ?
    ORDER BY window_index ASC
    """
    df = pd.read_sql_query(query, conn, params=(session_id,))
    conn.close()
    return df


def get_session_suggestions_df(session_id: int) -> pd.DataFrame:
    """Returns pedagogical suggestions for a session."""
    init_streamlit_db()
    conn = get_db_connection()
    query = """
    SELECT 
        id,
        suggestion_type,
        title,
        description,
        evidence,
        urgency,
        generated_by,
        created_at
    FROM teaching_suggestions
    WHERE session_id = ?
    ORDER BY id ASC
    """
    df = pd.read_sql_query(query, conn, params=(session_id,))
    conn.close()
    return df


def generate_live_suggestions(session_id: int):
    """Generates context-aware suggestions from current session telemetry."""
    windows_df = get_session_windows_df(session_id)
    if windows_df.empty:
        return

    conn = get_db_connection()
    c = conn.cursor()

    # Clear existing
    c.execute("DELETE FROM teaching_suggestions WHERE session_id = ?", (session_id,))

    scores = windows_df["interaction_score"].tolist()
    fwds = windows_df["forward_facing_pct"].tolist()
    speeches = windows_df["speech_active_pct"].tolist()
    motions = windows_df["movement_intensity_pct"].tolist()

    avg_score = float(np.mean(scores))
    avg_fwd = float(np.mean(fwds))
    avg_speech = float(np.mean(speeches))
    avg_motion = float(np.mean(motions))

    suggestions = []

    # 1. Forward pose drop check
    if avg_fwd < 50.0:
        suggestions.append((
            session_id, 'ENGAGEMENT_DROP',
            'Forward-Facing Orientation Below Norm',
            f'Average forward-facing ratio is {avg_fwd:.1f}%. Action: Switch lecture format: Pose a direct question, run an active poll, or transition to a board demonstration.',
            f'Measured average forward head pose Euler alignment: {avg_fwd:.1f}% across {len(fwds)} windows.',
            'HIGH', 'RULE_BASED'
        ))

    # 2. Low speech / extended monologue check
    if avg_speech < 15.0:
        suggestions.append((
            session_id, 'LOW_AUDIO',
            'Sustained Low Classroom Vocal Engagement',
            f'Classroom speech activity ratio is only {avg_speech:.1f}%. Action: Conduct a 2-minute Peer Discussion (Think-Pair-Share) on the topic just covered.',
            f'Aggregate audio VAD threshold was active in only {avg_speech:.1f}% of recorded windows.',
            'MEDIUM', 'RULE_BASED'
        ))

    # 3. High motion / restlessness check
    if avg_motion > 45.0:
        suggestions.append((
            session_id, 'RESTLESSNESS',
            'Elevated Physical Restlessness Detected',
            f'Motion differencing index averaged {avg_motion:.1f}%. Action: Allow a brief 60-second stretch break or prompt a student to summarize key findings.',
            f'Frame differencing movement intensity index exceeded 45% threshold.',
            'LOW', 'RULE_BASED'
        ))

    # 4. Interaction peak positive reinforcement
    if avg_score >= 80.0:
        suggestions.append((
            session_id, 'INTERACTION_PEAK',
            'High Observable Interaction Level Sustained',
            f'Mean interaction index is {avg_score:.1f}%, demonstrating strong alignment with content.',
            f'Combined score of head orientation, moderate motion, and balanced speech reached {avg_score:.1f}/100.',
            'LOW', 'RULE_BASED'
        ))

    # Default if none triggered
    if not suggestions:
        suggestions.append((
            session_id, 'ROUTINE_CHECK',
            'Standard Classroom Dynamic Observed',
            'Behavioral engagement metrics and attendance verifications are within balanced thresholds.',
            f'Mean interaction score: {avg_score:.1f} / 100.',
            'LOW', 'RULE_BASED'
        ))

    for s in suggestions:
        c.execute("""
        INSERT INTO teaching_suggestions (
            session_id, suggestion_type, title, description, evidence, urgency, generated_by
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, s)

    conn.commit()
    conn.close()
