from django.contrib import admin
from .models import Employee, ActivityLog


@admin.register(ActivityLog)
class ActivityLogAdmin(admin.ModelAdmin):
    list_display = (
        "employee",
        "status",
        "activity_label",          # 👈 NEW
        "webcam_present",
        "keyboard_active",
        "mouse_active",
        "confidence",
        "created_at",
    )
    list_filter = (
        "status",
        "activity_label",          # 👈 NEW
        "webcam_present",
        "keyboard_active",
        "mouse_active",
        "created_at",
    )

    list_filter = ("status", "webcam_present", "keyboard_active", "mouse_active", "created_at")
    search_fields = ("employee__employee_code", "employee__name")
