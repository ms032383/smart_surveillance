from django.db import models


class Employee(models.Model):
    employee_code = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return self.name or self.employee_code


class ActivityLog(models.Model):
    STATUS_CHOICES = [
        ("WORKING", "Working"),
        ("IDLE", "Idle"),
        ("AWAY", "Away from desk"),
    ]

    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    webcam_present = models.BooleanField(default=False)
    keyboard_active = models.BooleanField(default=False)
    mouse_active = models.BooleanField(default=False)
    confidence = models.FloatField(default=0.0)
    activity_label = models.CharField(max_length=50, blank=True, null=True)  # 👈 NEW
    created_at = models.DateTimeField(auto_now_add=True)
    extra_info = models.JSONField(blank=True, null=True)

    def __str__(self):
        return f"{self.employee} - {self.status} @ {self.created_at}"

