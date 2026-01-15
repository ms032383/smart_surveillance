from django.urls import path
from .views import receive_log
from .views import receive_log, dashboard, video_feed

urlpatterns = [
    path("api/logs/", receive_log, name="receive_log"),
    path("dashboard/", dashboard, name="dashboard"),
    path("video-feed/", video_feed, name="video_feed"),
]
