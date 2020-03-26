from django.urls import path
from drugs_uri.views import IndexView, Conflict, Sidefx


urlpatterns = [
    path('index', IndexView.as_view(), name="index"),
    path('conflict', Conflict.as_view(), name="conflict"),
    path('sidefx', Sidefx.as_view(), name="sidefx"),
]
