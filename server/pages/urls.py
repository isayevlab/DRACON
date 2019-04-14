import sys

from django.conf.urls import url
from pages import views
from django.conf.urls.static import static
from django.conf import settings


page_view = views.HomePageView



urlpatterns = [
    url(r'^$', page_view.as_view()),
   url(r'^main/$', page_view.as_view()),
    # url(r'^$', views.HomePageView.my_view),
    #url(r'^upload/', views.upload, name="upload"),
    #url(r'^main/upload/', views.upload, name="upload"),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)