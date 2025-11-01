from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    # Template views
    path('', views.home_view, name='home'),
    path('predict/', views.predict_view, name='predict'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('history/', views.history_view, name='history'),
]

# ============================================
# predictor/api_urls.py - REST API URLs
# ============================================

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'predictions', views.PredictionViewSet, basename='prediction')
router.register(r'feedback', views.FeedbackViewSet, basename='feedback')

urlpatterns = [
    path('', include(router.urls)),
    path('model-info/', views.model_info, name='model-info'),
    path('batch-predict/', views.batch_predict, name='batch-predict'),
]
