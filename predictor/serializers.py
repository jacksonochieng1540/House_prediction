"""
REST Framework serializers for property prediction
"""

from rest_framework import serializers
from .models import PredictionRequest, UserFeedback, ModelMetrics

class PredictionInputSerializer(serializers.Serializer):
    """Serializer for prediction input validation"""
    property_type = serializers.ChoiceField(
        choices=PredictionRequest._meta.get_field('property_type').choices
    )
    location = serializers.ChoiceField(
        choices=PredictionRequest._meta.get_field('location').choices
    )
    bedrooms = serializers.IntegerField(min_value=1, max_value=20)
    bathrooms = serializers.IntegerField(min_value=1, max_value=20)
    house_size = serializers.FloatField(
        required=False, 
        allow_null=True, 
        min_value=0
    )
    land_size = serializers.FloatField(
        required=False, 
        allow_null=True, 
        min_value=0
    )

class PredictionRequestSerializer(serializers.ModelSerializer):
    """Serializer for prediction request model"""
    price_formatted = serializers.ReadOnlyField()
    features_dict = serializers.ReadOnlyField()
    
    class Meta:
        model = PredictionRequest
        fields = [
            'id',
            'property_type',
            'location',
            'bedrooms',
            'bathrooms',
            'house_size',
            'land_size',
            'predicted_price',
            'price_formatted',
            'confidence_score',
            'price_range_min',
            'price_range_max',
            'created_at',
            'features_dict',
        ]
        read_only_fields = [
            'id',
            'predicted_price',
            'confidence_score',
            'price_range_min',
            'price_range_max',
            'created_at',
        ]

class UserFeedbackSerializer(serializers.ModelSerializer):
    """Serializer for user feedback"""
    
    class Meta:
        model = UserFeedback
        fields = [
            'id',
            'prediction',
            'rating',
            'comment',
            'actual_price',
            'created_at',
        ]
        read_only_fields = ['id', 'created_at']
    
    def validate_rating(self, value):
        """Validate rating is between 1 and 5"""
        if value < 1 or value > 5:
            raise serializers.ValidationError("Rating must be between 1 and 5")
        return value

class ModelMetricsSerializer(serializers.ModelSerializer):
    """Serializer for model metrics"""
    
    class Meta:
        model = ModelMetrics
        fields = [
            'id',
            'model_version',
            'mae',
            'rmse',
            'r2_score',
            'mape',
            'training_date',
            'is_active',
        ]
        read_only_fields = ['id', 'training_date']