from django.contrib import admin
from .models import PredictionRequest, UserFeedback, ModelMetrics

@admin.register(PredictionRequest)
class PredictionRequestAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'property_type', 'location', 'bedrooms', 'bathrooms',
        'predicted_price', 'confidence_score', 'created_at'
    ]
    list_filter = ['property_type', 'location', 'created_at']
    search_fields = ['location', 'property_type']
    readonly_fields = [
        'predicted_price', 'confidence_score', 'price_range_min',
        'price_range_max', 'created_at', 'updated_at'
    ]
    date_hierarchy = 'created_at'
    ordering = ['-created_at']

@admin.register(UserFeedback)
class UserFeedbackAdmin(admin.ModelAdmin):
    list_display = ['id', 'prediction', 'rating', 'created_at']
    list_filter = ['rating', 'created_at']
    search_fields = ['comment']
    readonly_fields = ['created_at']
    date_hierarchy = 'created_at'

@admin.register(ModelMetrics)
class ModelMetricsAdmin(admin.ModelAdmin):
    list_display = [
        'model_version', 'r2_score', 'mae', 'rmse', 'mape',
        'training_date', 'is_active'
    ]
    list_filter = ['is_active', 'training_date']
    readonly_fields = ['training_date']
    date_hierarchy = 'training_date'

