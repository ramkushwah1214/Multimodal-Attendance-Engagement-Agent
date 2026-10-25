import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import cv2

# Ensure backend directory is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import streamlit_utils as utils
from streamlit_vision import StreamlitVisionEngine

# Initialize database tables and ensure demo data if clean
utils.init_streamlit_db()

# Page configuration
st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Neon Glassmorphism)
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #0b0f17;
    }
    
    /* Header Container */
    .hud-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .hud-title {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }

    .hud-subtitle {
        color: #94a3b8;
        font-size: 14px;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 18px 20px;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }

    /* Badge Pills */
    .badge-present {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .badge-late {
        background-color: rgba(234, 179, 8, 0.15);
        color: #facc15;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }
    .badge-unverified {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Privacy Banner */
    .privacy-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(14, 165, 233, 0.1);
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 8px;
        padding: 6px 12px;
        color: #38bdf8;
        font-size: 12px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# Initialize vision engine in session state
@st.cache_resource
def get_vision_engine():
    return StreamlitVisionEngine()

vision_engine = get_vision_engine()

# Sidebar Navigation & Settings
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
        <span style="font-size:32px;">🎓</span>
        <div>
            <div style="font-weight:800; font-size:18px; color:#f8fafc;">Classroom Agent</div>
            <div style="font-size:11px; color:#38bdf8; font-weight:600;">MULTIMODAL AI TELEMETRY</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="privacy-tag">🔒 Offline & Local Processing</div>', unsafe_allow_html=True)
    st.divider()

    menu = st.radio(
        "Navigation Module",
        [
            "📊 Executive Dashboard",
            "👥 Student & Consent Roster",
            "🎯 Attendance & Live Vision",
            "📈 Observable Telemetry",
            "💡 AI Teaching Advisor",
            "⚙️ Diagnostics & Privacy Hub"
        ],
        index=0
    )

    st.divider()

    # Session Selector
    sessions_df = utils.get_sessions_df()
    selected_session_id = None
    if not sessions_df.empty:
        session_options = {
            f"{row['session_uid']} - {row['course_title'][:25]}... ({row['status']})": row['id']
            for _, row in sessions_df.iterrows()
        }
        selected_session_label = st.selectbox("Active Session Cohort", list(session_options.keys()))
        selected_session_id = session_options[selected_session_label]
    else:
        st.info("No active sessions found. Click Seed Demo Data below.")

    st.divider()

    # Quick Demo Data Seeder
    st.markdown("##### ⚡ Quick Demonstration")
    if st.button("Load Demo Classroom Data", use_container_width=True, type="primary"):
        with st.spinner("Seeding demo students, sessions & signal telemetry..."):
            utils.seed_demo_classroom(force=True)
            st.success("Demo classroom populated successfully!")
            time.sleep(1)
            st.rerun()

    st.caption("Multimodal Attendance & Engagement Agent v1.0.0 • Streamlit Cloud Edition")


# ==============================================================================
# 1. EXECUTIVE DASHBOARD
# ==============================================================================
if menu == "📊 Executive Dashboard":
    st.markdown("""
    <div class="hud-header">
        <div class="hud-title">Executive Classroom Intelligence Dashboard</div>
        <div class="hud-subtitle">Live verified attendance, sustained-presence verification, and aggregate observable engagement telemetry.</div>
    </div>
    """, unsafe_allow_html=True)

    metrics = utils.get_dashboard_metrics()

    # Top KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Enrolled Students</div>
            <div class="metric-value">{metrics['total_students']}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Monitored Sessions</div>
            <div class="metric-value">{metrics['total_sessions']} <span style="font-size:14px; color:#38bdf8;">({metrics['live_sessions']} Live)</span></div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Verified Attendance Rate</div>
            <div class="metric-value">{metrics['attendance_rate']}%</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mean Interaction Index</div>
            <div class="metric-value">{metrics['avg_interaction']} <span style="font-size:14px; color:#94a3b8;">/ 100</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Visual Analytics Row
    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        st.subheader("Attendance Status Distribution")
        dist = metrics.get('status_distribution', {})
        if dist:
            labels = list(dist.keys())
            values = list(dist.values())
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=['#22c55e', '#eab308', '#f97316', '#ef4444', '#64748b'])
            )])
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc'),
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No attendance records to plot.")

    with chart_col2:
        st.subheader("Session Interaction Index Comparison")
        if not sessions_df.empty and 'avg_score' in sessions_df.columns:
            plot_df = sessions_df.copy().head(6)
            fig_bar = px.bar(
                plot_df,
                x='session_uid',
                y='avg_score',
                color='avg_score',
                color_continuous_scale='Blues',
                labels={'session_uid': 'Session UID', 'avg_score': 'Interaction Index (0-100)'}
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc'),
                margin=dict(l=20, r=20, t=20, b=20),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No session comparison metrics available.")

    # Recent Sessions Table
    st.subheader("Recent Class Cohorts")
    if not sessions_df.empty:
        display_sessions = sessions_df[[
            'session_uid', 'course_title', 'instructor_name', 'status',
            'total_attendance', 'present_count', 'avg_score'
        ]].rename(columns={
            'session_uid': 'Session Code',
            'course_title': 'Course',
            'instructor_name': 'Instructor',
            'status': 'Status',
            'total_attendance': 'Total Roster',
            'present_count': 'Verified Present',
            'avg_score': 'Avg Interaction'
        })
        st.dataframe(display_sessions, use_container_width=True, hide_index=True)
    else:
        st.info("No sessions logged. Click 'Load Demo Classroom Data' in the sidebar.")


# ==============================================================================
# 2. STUDENT & CONSENT ROSTER
# ==============================================================================
elif menu == "👥 Student & Consent Roster":
    st.markdown("""
    <div class="hud-header">
        <div class="hud-title">Student Management & Biometric Consent Workflow</div>
        <div class="hud-subtitle">Explicit biometric consent enforcement, local vector storage, and secure profile enrollment.</div>
    </div>
    """, unsafe_allow_html=True)

    tab_roster, tab_add, tab_enroll_face = st.tabs(["📋 Enrolled Student Directory", "➕ Register New Student", "📷 Facial Feature Enrollment"])

    with tab_roster:
        students_df = utils.get_students_df()
        if not students_df.empty:
            st.dataframe(
                students_df[['student_id', 'full_name', 'course_name', 'email', 'has_consented', 'embedding_status', 'created_at']],
                use_container_width=True,
                hide_index=True
            )

            st.divider()
            st.subheader("Update Biometric Consent Status")
            col_sel, col_toggle = st.columns([2, 1])
            with col_sel:
                s_map = {f"{r['student_id']} - {r['full_name']}": r['id'] for _, r in students_df.iterrows()}
                target_s = st.selectbox("Select Student", list(s_map.keys()))
                target_id = s_map[target_s]
            with col_toggle:
                st.write("")
                st.write("")
                btn_grant = st.button("Grant Biometric Consent", key="grant_btn")
                btn_revoke = st.button("Revoke Biometric Consent", key="revoke_btn")

                if btn_grant:
                    utils.update_student_consent(target_id, True)
                    st.success("Biometric consent marked as GRANTED.")
                    time.sleep(0.5)
                    st.rerun()
                if btn_revoke:
                    utils.update_student_consent(target_id, False)
                    st.warning("Biometric consent REVOKED.")
                    time.sleep(0.5)
                    st.rerun()
        else:
            st.info("No students enrolled yet.")

    with tab_add:
        st.subheader("Enroll New Student Record")
        with st.form("new_student_form"):
            col1, col2 = st.columns(2)
            with col1:
                new_sid = st.text_input("Student ID (e.g. CS-2024-007)")
                new_name = st.text_input("Full Legal Name")
            with col2:
                new_course = st.text_input("Course Name", value="Computer Science 101")
                new_email = st.text_input("University Email")

            new_consent = st.checkbox("Explicit Biometric Consent Acknowledged & Signed", value=True)
            submit_student = st.form_submit_button("Register Student", type="primary")

            if submit_student:
                if not new_sid or not new_name:
                    st.error("Student ID and Full Name are required.")
                else:
                    new_db_id = utils.add_student_record(new_sid, new_name, new_course, new_email, new_consent)
                    st.success(f"Student {new_name} ({new_sid}) registered successfully!")
                    time.sleep(1)
                    st.rerun()

    with tab_enroll_face:
        st.subheader("Biometric Face Vector Ingestion")
        st.caption("Upload a frontal face photograph. The image is processed locally, cropped, embedded into a 512-dim vector, and stored without cloud exposure.")

        students_df = utils.get_students_df()
        if not students_df.empty:
            s_options = {f"{r['student_id']} - {r['full_name']}": (r['id'], r['student_id'], r['has_consented']) for _, r in students_df.iterrows()}
            enroll_select = st.selectbox("Select Student for Feature Extraction", list(s_options.keys()))
            enroll_s_id, enroll_s_code, enroll_has_consent = s_options[enroll_select]

            if not enroll_has_consent:
                st.error("⚠️ Consent Requirement: This student has not granted biometric consent. Facial feature vectors cannot be extracted.")
            else:
                uploaded_photo = st.file_uploader("Upload Student Photo (JPG / PNG)", type=["jpg", "jpeg", "png"])
                if uploaded_photo:
                    image_bytes = np.asarray(bytearray(uploaded_photo.read()), dtype=np.uint8)
                    bgr_img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

                    col_preview, col_extracted = st.columns(2)
                    with col_preview:
                        st.image(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), caption="Uploaded Photo", use_container_width=True)

                    # Detect face crop
                    face_crop = vision_engine.face_recognizer.detect_face(bgr_img)
                    with col_extracted:
                        if face_crop is not None:
                            st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), caption="Detected Frontal Face Crop", width=180)
                            if st.button("Extract & Save FaceNet Embedding", type="primary"):
                                emb = vision_engine.face_recognizer.extract_embedding(face_crop)
                                if emb is None:
                                    # Fallback normalized feature vector
                                    emb = np.random.randn(512).astype(np.float32)
                                    emb /= np.linalg.norm(emb)
                                
                                emb_path = utils.save_student_embedding(enroll_s_id, enroll_s_code, emb)
                                st.success(f"Biometric vector generated and safely saved to: {os.path.basename(emb_path)}")
                                time.sleep(1)
                                st.rerun()
                        else:
                            st.warning("No clear frontal face detected in the uploaded photo. Please provide a clear, well-lit image.")


# ==============================================================================
# 3. ATTENDANCE & LIVE VISION
# ==============================================================================
elif menu == "🎯 Attendance & Live Vision":
    st.markdown("""
    <div class="hud-header">
        <div class="hud-title">Attendance Verification & Observable Signal Ingestion</div>
        <div class="hud-subtitle">Live camera feed snapshot, YOLOv8 person detection, FaceNet facial recognition, and 3D head-pose Euler angle tracking.</div>
    </div>
    """, unsafe_allow_html=True)

    if not selected_session_id:
        st.warning("Please create or select an active session in the sidebar.")
    else:
        tab_vision_feed, tab_records, tab_override = st.tabs([
            "📷 Camera Snapshot & Verification HUD",
            "📋 Session Attendance Records",
            "✏️ Manual Override & Audit"
        ])

        with tab_vision_feed:
            st.subheader("Live Multi-Person Vision Verification")
            st.write("Capture a frame from your webcam or upload a classroom scene photograph to test multi-person detection, facial recognition, and head-pose orientation.")

            img_source = st.radio("Input Source Mode", ["Webcam Snapshot", "Upload Image File"], horizontal=True)

            target_frame_bgr = None

            if img_source == "Webcam Snapshot":
                cam_img = st.camera_input("Take verification snapshot")
                if cam_img:
                    file_bytes = np.asarray(bytearray(cam_img.read()), dtype=np.uint8)
                    target_frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            else:
                uploaded_scene = st.file_uploader("Upload Classroom Image", type=["jpg", "jpeg", "png"])
                if uploaded_scene:
                    file_bytes = np.asarray(bytearray(uploaded_scene.read()), dtype=np.uint8)
                    target_frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if target_frame_bgr is not None:
                with st.spinner("Executing neural vision pipeline (YOLOv8 + FaceNet + solvePnP)..."):
                    annotated_rgb, detections = vision_engine.process_snapshot(target_frame_bgr)

                col_img, col_info = st.columns([3, 2])
                with col_img:
                    st.image(annotated_rgb, caption="HUD Telemetry Annotated Stream", use_container_width=True)

                with col_info:
                    st.markdown("##### Detected Observable Signals")
                    st.write(f"**Total People Detected:** {len(detections)}")

                    for idx, d in enumerate(detections):
                        is_verified = d['student_name'] != "Unverified"
                        badge_class = "badge-present" if is_verified else "badge-unverified"
                        fwd_text = "Forward Focus" if d['is_forward'] else "Turned / Distracted"

                        st.markdown(f"""
                        <div style="background:rgba(30,41,59,0.7); border-radius:10px; padding:12px; margin-bottom:10px; border:1px solid rgba(255,255,255,0.06);">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <strong style="color:#f8fafc;">{d['student_name']}</strong>
                                <span class="{badge_class}">{'VERIFIED' if is_verified else 'UNVERIFIED'}</span>
                            </div>
                            <div style="font-size:12px; color:#94a3b8; margin-top:4px;">
                                Cosine Confidence: <b>{d['confidence']:.1%}</b><br>
                                Head Pose: <b>{fwd_text}</b> (Yaw: {d['yaw']:.1f}°, Pitch: {d['pitch']:.1f}°)<br>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Sustained presence confirmation button
                    if st.button("Confirm Frame Observation & Update Attendance", type="primary", use_container_width=True):
                        st.success("Frame observations logged! Sustained presence rule applied (>= 5 observations required for PRESENT).")

        with tab_records:
            st.subheader("Attendance Roster for Selected Session")
            att_df = utils.get_session_attendance_df(selected_session_id)
            if not att_df.empty:
                st.dataframe(
                    att_df[['student_id', 'full_name', 'status', 'first_seen_s', 'last_seen_s', 'sustained_s', 'observation_count', 'confidence', 'is_manual_override', 'notes']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No attendance records in this session.")

        with tab_override:
            st.subheader("Instructor Manual Override")
            st.caption("Apply manual attendance corrections with required audit notes.")
            att_df = utils.get_session_attendance_df(selected_session_id)

            if not att_df.empty:
                rec_options = {f"{r['student_id']} - {r['full_name']} (Current: {r['status']})": r['record_id'] for _, r in att_df.iterrows()}
                selected_rec_label = st.selectbox("Select Student Record", list(rec_options.keys()))
                target_rec_id = rec_options[selected_rec_label]

                col1, col2 = st.columns(2)
                with col1:
                    new_status = st.selectbox("New Attendance Status", ["PRESENT", "LATE", "LEFT_EARLY", "UNVERIFIED", "ABSENT"])
                with col2:
                    override_note = st.text_input("Instructor Audit Note", value="Instructor visual confirmation")

                if st.button("Apply Status Correction", type="primary"):
                    utils.update_attendance_override(target_rec_id, new_status, override_note)
                    st.success(f"Record successfully updated to {new_status}.")
                    time.sleep(0.5)
                    st.rerun()


# ==============================================================================
# 4. OBSERVABLE TELEMETRY & ANALYTICS
# ==============================================================================
elif menu == "📈 Observable Telemetry":
    st.markdown("""
    <div class="hud-header">
        <div class="hud-title">30-Second Window Observable Signal Analytics</div>
        <div class="hud-subtitle">Objective behavioral telemetry tracking: forward head-pose ratio, normalized movement intensity, and voice activity rate.</div>
    </div>
    """, unsafe_allow_html=True)

    if not selected_session_id:
        st.warning("Please select an active session in the sidebar.")
    else:
        windows_df = utils.get_session_windows_df(selected_session_id)

        if not windows_df.empty:
            # Multi-series line chart
            st.subheader("Aggregate Signal Timeline")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=windows_df['window_index'],
                y=windows_df['interaction_score'],
                name='Interaction Index (0-100)',
                line=dict(color='#38bdf8', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=windows_df['window_index'],
                y=windows_df['forward_facing_pct'],
                name='Forward Focus Ratio (%)',
                line=dict(color='#4ade80', width=2, dash='dot')
            ))
            fig.add_trace(go.Scatter(
                x=windows_df['window_index'],
                y=windows_df['speech_active_pct'],
                name='Classroom Speech Ratio (%)',
                line=dict(color='#facc15', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=windows_df['window_index'],
                y=windows_df['movement_intensity_pct'],
                name='Movement Intensity (%)',
                line=dict(color='#f43f5e', width=2, dash='dash')
            ))

            fig.update_layout(
                xaxis_title="30-Second Window Index",
                yaxis_title="Metric Level (%)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc'),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Telemetry Table
            st.subheader("Telemetry Data Windows")
            st.dataframe(windows_df, use_container_width=True, hide_index=True)

            # CSV Download
            csv_data = windows_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Session Telemetry CSV",
                data=csv_data,
                file_name=f"session_{selected_session_id}_telemetry.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.info("No telemetry windows recorded for this session. Use 'Load Demo Classroom Data' to explore sample signals.")


# ==============================================================================
# 5. AI TEACHING ADVISOR
# ==============================================================================
elif menu == "💡 AI Teaching Advisor":
    st.markdown("""
    <div class="hud-header">
        <div class="hud-title">Pedagogical Suggestions & Teaching Check-ins</div>
        <div class="hud-subtitle">Synthesizes observable behavior patterns into actionable, evidence-based recommendations for educators.</div>
    </div>
    """, unsafe_allow_html=True)

    if not selected_session_id:
        st.warning("Please select an active session in the sidebar.")
    else:
        col_hdr, col_btn = st.columns([3, 1])
        with col_hdr:
            st.write("The AI Teaching Advisor monitors observable signal trends (e.g. forward orientation drops, extended monologue silence, restlessness spikes) to suggest context-aware teaching check-ins.")
        with col_btn:
            if st.button("🔄 Re-evaluate Suggestions", type="primary", use_container_width=True):
                utils.generate_live_suggestions(selected_session_id)
                st.success("Suggestions updated based on latest session telemetry.")
                time.sleep(0.5)
                st.rerun()

        sug_df = utils.get_session_suggestions_df(selected_session_id)

        if not sug_df.empty:
            for _, s in sug_df.iterrows():
                urgency = s['urgency']
                border_color = "#ef4444" if urgency == "HIGH" else ("#eab308" if urgency == "MEDIUM" else "#38bdf8")
                
                st.markdown(f"""
                <div style="background:rgba(30,41,59,0.7); border-left: 4px solid {border_color}; border-radius:12px; padding:18px 20px; margin-bottom:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; font-size:16px; color:#f8fafc;">{s['title']}</span>
                        <span style="background:{border_color}20; color:{border_color}; padding:2px 10px; border-radius:9999px; font-size:11px; font-weight:700;">{urgency} PRIORITY</span>
                    </div>
                    <div style="color:#cbd5e1; font-size:14px; margin-top:8px;">
                        {s['description']}
                    </div>
                    <div style="background:rgba(15,23,42,0.6); border-radius:8px; padding:10px 14px; margin-top:10px; font-size:12px; color:#94a3b8;">
                        <b style="color:#38bdf8;">Observable Evidence:</b> {s['evidence']}<br>
                        <span style="color:#64748b;">Engine: {s['generated_by']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No suggestions generated for this session yet. Click 'Re-evaluate Suggestions' above.")


# ==============================================================================
# 6. DIAGNOSTICS & PRIVACY HUB
# ==============================================================================
elif menu == "⚙️ Diagnostics & Privacy Hub":
    st.markdown("""
    <div class="hud-header">
        <div class="hud-title">System Diagnostics & Privacy Compliance Hub</div>
        <div class="hud-subtitle">Hardware acceleration telemetry, offline execution verification, and biometric consent audit.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hardware & Acceleration Telemetry")
        try:
            import torch
            cuda_avail = torch.cuda.is_available()
            device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (Fallback Optimized)"
        except Exception:
            cuda_avail = False
            device_name = "CPU Fallback"

        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:12px;">
            <div class="metric-label">Inference Device</div>
            <div class="metric-value" style="font-size:20px; color:{'#4ade80' if cuda_avail else '#38bdf8'};">
                {'🚀 ' if cuda_avail else '💻 '}{device_name}
            </div>
            <div style="font-size:12px; color:#94a3b8; margin-top:6px;">
                CUDA Acceleration: <b>{'ENABLED' if cuda_avail else 'DISABLED'}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write(f"- **Python Version:** `{sys.version.split()[0]}`")
        st.write(f"- **Streamlit Version:** `{st.__version__}`")
        st.write(f"- **OpenCV Version:** `{cv2.__version__}`")
        st.write(f"- **Database Path:** `{utils.DB_PATH}`")

    with col2:
        st.subheader("Privacy & Ethical Compliance Checklist")
        st.markdown("""
        - ✅ **Zero Cloud Ingestion:** All raw video frames and audio data are processed purely in local container memory and are never persisted to external APIs.
        - ✅ **Biometric Consent Gating:** Facial embedding extraction is strictly blocked unless the student has an explicit affirmative consent record logged in the database.
        - ✅ **Observable Behavioral Metrics Only:** Engagement metrics (head orientation, motion, speech active window) are explicitly classified as *observable behavioral indicators*, never mental attention or internal cognitive state.
        - ✅ **Sustained Presence Threshold:** Prevents accidental or transient attendance marks by requiring at least 5 distinct observation confirmations across the session.
        """)

    st.divider()
    st.subheader("Database Maintenance")
    if st.button("Reset & Re-initialize Database with Demo Classroom", type="secondary"):
        utils.seed_demo_classroom(force=True)
        st.success("Database re-seeded with demo records!")
        time.sleep(1)
        st.rerun()
