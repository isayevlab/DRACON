from django.conf.urls import url
from pages import views
from django.conf.urls.static import static
from django.conf import settings


menu_view = views.MenuPageView.as_view()
demo_view = views.DemoPageView.as_view()
tsne_view = views.TSNEPageView.as_view()


urlpatterns = [
    url(r'^$', menu_view),
    url(r'^demo$', demo_view),
    url(r'^tsne$', tsne_view)
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
