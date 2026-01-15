import json
import time
from collections import defaultdict
from datetime import datetime, timedelta

import cv2
import mediapipe as mp

from django.db.models import Count
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

from .models import Employee, ActivityLog

# MediaPipe globals for streaming
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ============================================
# RECEIVE LOGS FROM AGENT
# ============================================
@csrf_exempt
def receive_log(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    employee_code = data.get("employee_code")
    status = data.get("status")
    webcam_present = data.get("webcam_present", False)
    keyboard_active = data.get("keyboard_active", False)
    mouse_active = data.get("mouse_active", False)
    confidence = data.get("confidence", 0.0)
    extra_info = data.get("extra_info", {})
    activity_label = data.get("activity_label")

    if not employee_code or not status:
        return JsonResponse({"error": "employee_code and status required"}, status=400)

    employee, _ = Employee.objects.get_or_create(employee_code=employee_code)

    ActivityLog.objects.create(
        employee=employee,
        status=status,
        webcam_present=webcam_present,
        keyboard_active=keyboard_active,
        mouse_active=mouse_active,
        confidence=confidence,
        activity_label=activity_label,
        extra_info=extra_info,
    )

    return JsonResponse({"message": "log saved"}, status=201)


# ============================================
# DAILY DURATION CALCULATOR
# ============================================
def calculate_daily_durations(logs_qs):
    """
    Returns:
    {
        '2025-12-07': {'WORKING': 120, 'IDLE': 45, 'AWAY': 30},
        ...
    }
    """
    daily = {}
    logs = list(logs_qs.order_by("created_at"))

    if not logs:
        return daily

    for i in range(len(logs)):
        curr = logs[i]
        curr_day = curr.created_at.date().isoformat()

        if curr_day not in daily:
            daily[curr_day] = {"WORKING": 0, "IDLE": 0, "AWAY": 0}

        if i < len(logs) - 1:
            next_time = logs[i + 1].created_at
        else:
            midnight = datetime.combine(
                curr.created_at.date() + timedelta(days=1),
                datetime.min.time()
            ).astimezone(curr.created_at.tzinfo)

            next_time = min(midnight, timezone.now())

        duration_min = (next_time - curr.created_at).total_seconds() / 60.0
        daily[curr_day][curr.status] += round(duration_min)

    return daily


# ============================================
# MAIN DASHBOARD WITH ANALYTICS
# ============================================
def dashboard(request):
    # ---------- FILTERS ----------
    start_str = request.GET.get("start")
    end_str = request.GET.get("end")
    employee_code = request.GET.get("employee")

    today = timezone.localdate()
    default_start = today - timedelta(days=6)
    default_end = today

    try:
        start_date = datetime.strptime(start_str, "%Y-%m-%d").date() if start_str else default_start
    except ValueError:
        start_date = default_start

    try:
        end_date = datetime.strptime(end_str, "%Y-%m-%d").date() if end_str else default_end
    except ValueError:
        end_date = default_end

    logs_qs = ActivityLog.objects.filter(
        created_at__date__gte=start_date,
        created_at__date__lte=end_date,
    ).select_related("employee")

    if employee_code and employee_code != "all":
        logs_qs = logs_qs.filter(employee__employee_code=employee_code)

    # ---------- STATUS DISTRIBUTION ----------
    status_data = logs_qs.values("status").annotate(count=Count("id")).order_by("status")

    status_labels = [x["status"] for x in status_data]
    status_counts = [x["count"] for x in status_data]
    status_counts_map = {x["status"]: x["count"] for x in status_data}

    # ---------- DAILY DURATIONS ----------
    daily_stats = calculate_daily_durations(logs_qs)
    sorted_dates = sorted(daily_stats.keys())

    working_series = [daily_stats[d].get("WORKING", 0) for d in sorted_dates]
    idle_series = [daily_stats[d].get("IDLE", 0) for d in sorted_dates]
    away_series = [daily_stats[d].get("AWAY", 0) for d in sorted_dates]

    # ---------- ACTIVITY DISTRIBUTION ----------
    activity_counts = defaultdict(int)
    for log in logs_qs:
        label = log.activity_label or "UNKNOWN"
        activity_counts[label] += 1

    activity_labels_json = json.dumps(list(activity_counts.keys()))
    activity_counts_json = json.dumps(list(activity_counts.values()))

    # ---------- HOURLY TIMELINE ----------
    hourly_counts = defaultdict(int)
    for log in logs_qs:
        hr = log.created_at.hour
        hourly_counts[hr] += 1

    timeline_hours_json = json.dumps(list(range(24)))
    timeline_data_json = json.dumps([hourly_counts.get(h, 0) for h in range(24)])

    # ---------- MULTI-EMPLOYEE COMPARISON ----------
    employee_activity_map = defaultdict(int)
    for log in logs_qs:
        employee_activity_map[log.employee.employee_code] += 1

    employee_compare_labels = json.dumps(list(employee_activity_map.keys()))
    employee_compare_values = json.dumps(list(employee_activity_map.values()))

    # ---------- TABLE + EMPLOYEES ----------
    logs = logs_qs.order_by("-created_at")[:300]
    employees = Employee.objects.all().order_by("employee_code")

    context = {
        "start_date": start_date,
        "end_date": end_date,
        "selected_employee": employee_code or "all",
        "employees": employees,
        "logs": logs,

        # Status pie
        "status_labels_json": json.dumps(status_labels),
        "status_counts_json": json.dumps(status_counts),
        "status_counts_map": status_counts_map,

        # Daily line
        "dates_json": json.dumps(sorted_dates),
        "working_json": json.dumps(working_series),
        "idle_json": json.dumps(idle_series),
        "away_json": json.dumps(away_series),

        # Activity pie
        "activity_labels_json": activity_labels_json,
        "activity_counts_json": activity_counts_json,

        # Hourly timeline
        "timeline_hours_json": timeline_hours_json,
        "timeline_data_json": timeline_data_json,

        # Employee comparison
        "employee_compare_labels": employee_compare_labels,
        "employee_compare_values": employee_compare_values,
    }

    return render(request, "monitoring/dashboard.html", context)


# ============================================
# REAL-TIME VIDEO STREAM
# ============================================
def gen_frames():
    """
    Capture webcam frames, draw pose skeleton, overlay latest status/activity
    and stream as MJPEG.
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    last_status_fetch = 0
    current_status = "UNKNOWN"
    current_activity = "-"

    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            # Pose
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            # Fetch latest log roughly once per second
            now_ts = time.time()
            if now_ts - last_status_fetch > 1:
                latest = ActivityLog.objects.order_by("-created_at").first()
                if latest:
                    current_status = latest.status
                    current_activity = latest.activity_label or "-"
                last_status_fetch = now_ts

            # Overlay text box
            cv2.rectangle(frame, (5, 5), (420, 70), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Status: {current_status}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Activity: {current_activity}",
                (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Encode as JPEG
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()
        pose.close()


def video_feed(request):
    return StreamingHttpResponse(
        gen_frames(),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )
