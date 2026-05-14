from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# API Router
router = DefaultRouter()
router.register(r'records', views.AddictionRecordViewSet)

app_name = 'prediction'

urlpatterns = [
    # Page views
    path('', views.index, name='index'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('predict/', views.predict_page, name='predict'),
    path('records/', views.records_page, name='records'),
    
    # API endpoints
    path('api/', include(router.urls)),
]
