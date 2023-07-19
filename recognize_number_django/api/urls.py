from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("recognize", views.recognize, name="recognize"),
    path("create", views.create, name="create"),
    path("list", views.list_model, name="list"),
]