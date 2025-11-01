"""
Django views for property price prediction
"""

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.db.models import Avg, Count, Q
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
from .models import PredictionRequest, UserFeedback, ModelMetrics
from .serializers import (
    PredictionRequestSerializer, 
    UserFeedbackSerializer,
    PredictionInputSerializer
)
from .ml_utils import predictor
import logging
import json

logger = logging.getLogger(__name__)

# Template Views
def home_view(request):
    """Main landing page"""
    context = {
        'property_types': dict(PredictionRequest._meta.get_field('property_type').choices),
        'locations': dict(PredictionRequest._meta.get_field('location').choices),
        'total_predictions': PredictionRequest.objects.count(),
        'avg_price': PredictionRequest.objects.aggregate(Avg('predicted_price'))['predicted_price__avg'],
    }
    return render(request, 'predictor/home.html', context)

def predict_view(request):
    """Prediction form page"""
    from .models import PropertyType, Location
    context = {
        'property_types': PropertyType.choices,
        'locations': Location.choices,
    }
    return render(request, 'predictor/predict.html', context)

def dashboard_view(request):
    """Analytics dashboard"""
    # Get statistics
    total_predictions = PredictionRequest.objects.count()
    
    # Property type distribution
    property_dist = PredictionRequest.objects.values('property_type').annotate(
        count=Count('id')
    ).order_by('-count')
    
    # Location distribution
    location_dist = PredictionRequest.objects.values('location').annotate(
        count=Count('id'),
        avg_price=Avg('predicted_price')
    ).order_by('-avg_price')[:10]
    
    # Recent predictions
    recent = PredictionRequest.objects.all()[:10]
    
    # Model metrics
    try:
        latest_metrics = ModelMetrics.objects.filter(is_active=True).first()
    except:
        latest_metrics = None
    
    context = {
        'total_predictions': total_predictions,
        'property_dist': list(property_dist),
        'location_dist': list(location_dist),
        'recent_predictions': recent,
        'model_metrics': latest_metrics,
    }
    return render(request, 'predictor/dashboard.html', context)

def history_view(request):
    """Prediction history page"""
    predictions = PredictionRequest.objects.all()[:50]
    context = {'predictions': predictions}
    return render(request, 'predictor/history.html', context)

# API Views
class PredictionViewSet(viewsets.ModelViewSet):
    """ViewSet for prediction requests"""
    queryset = PredictionRequest.objects.all()
    serializer_class = PredictionRequestSerializer
    
    def get_client_ip(self, request):
        """Get client IP address"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
    
    @action(detail=False, methods=['post'])
    def predict(self, request):
        """Make a price prediction"""
        try:
            # Validate input
            serializer = PredictionInputSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(
                    {'error': 'Invalid input', 'details': serializer.errors},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Prepare data for prediction
            property_data = {
                'property_type': serializer.validated_data['property_type'],
                'location': serializer.validated_data['location'],
                'bedrooms': serializer.validated_data['bedrooms'],
                'bathrooms': serializer.validated_data['bathrooms'],
                'house_size': serializer.validated_data.get('house_size'),
                'land_size': serializer.validated_data.get('land_size'),
            }
            
            # Make prediction
            result = predictor.predict_with_confidence(property_data)
            
            # Get explanation
            explanations = predictor.explain_prediction(property_data, result)
            
            # Save prediction to database
            prediction_obj = PredictionRequest.objects.create(
                property_type=property_data['property_type'],
                location=property_data['location'],
                bedrooms=property_data['bedrooms'],
                bathrooms=property_data['bathrooms'],
                house_size=property_data.get('house_size'),
                land_size=property_data.get('land_size'),
                predicted_price=result['predicted_price'],
                confidence_score=result['confidence_score'],
                price_range_min=result['price_range_min'],
                price_range_max=result['price_range_max'],
                ip_address=self.get_client_ip(request),
                session_id=request.session.session_key,
            )
            
            # Prepare response
            response_data = {
                'prediction_id': prediction_obj.id,
                'predicted_price': result['predicted_price'],
                'predicted_price_formatted': f"KSh {result['predicted_price']:,.0f}",
                'confidence_score': result['confidence_score'],
                'confidence_percentage': f"{result['confidence_score'] * 100:.1f}%",
                'price_range': {
                    'min': result['price_range_min'],
                    'max': result['price_range_max'],
                    'min_formatted': f"KSh {result['price_range_min']:,.0f}",
                    'max_formatted': f"KSh {result['price_range_max']:,.0f}",
                },
                'explanations': explanations,
                'model_metrics': result['model_metrics'],
                'input_features': property_data,
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return Response(
                {'error': 'Prediction failed', 'details': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'])
    def statistics(self, request):
        """Get prediction statistics"""
        try:
            stats = {
                'total_predictions': PredictionRequest.objects.count(),
                'avg_price': PredictionRequest.objects.aggregate(
                    Avg('predicted_price')
                )['predicted_price__avg'],
                'by_property_type': list(
                    PredictionRequest.objects.values('property_type').annotate(
                        count=Count('id'),
                        avg_price=Avg('predicted_price')
                    )
                ),
                'by_location': list(
                    PredictionRequest.objects.values('location').annotate(
                        count=Count('id'),
                        avg_price=Avg('predicted_price')
                    ).order_by('-avg_price')[:10]
                ),
                'recent_predictions': PredictionRequestSerializer(
                    PredictionRequest.objects.all()[:10], many=True
                ).data,
            }
            return Response(stats, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Statistics error: {str(e)}")
            return Response(
                {'error': 'Failed to fetch statistics'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class FeedbackViewSet(viewsets.ModelViewSet):
    """ViewSet for user feedback"""
    queryset = UserFeedback.objects.all()
    serializer_class = UserFeedbackSerializer
    
    def create(self, request, *args, **kwargs):
        """Submit feedback for a prediction"""
        try:
            serializer = self.get_serializer(data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response(
                    {'message': 'Feedback submitted successfully', 'data': serializer.data},
                    status=status.HTTP_201_CREATED
                )
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Feedback error: {str(e)}")
            return Response(
                {'error': 'Failed to submit feedback'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

@api_view(['GET'])
def model_info(request):
    """Get information about the ML model"""
    try:
        metrics = ModelMetrics.objects.filter(is_active=True).first()
        
        info = {
            'model_version': metrics.model_version if metrics else 'v1.0',
            'metrics': {
                'mae': metrics.mae if metrics else None,
                'rmse': metrics.rmse if metrics else None,
                'r2_score': metrics.r2_score if metrics else None,
                'mape': metrics.mape if metrics else None,
            } if metrics else predictor.stats.get('metrics', {}),
            'training_date': metrics.training_date if metrics else None,
            'feature_importance': predictor.stats.get('feature_importance', {}),
            'supported_property_types': [choice[0] for choice in PredictionRequest._meta.get_field('property_type').choices],
            'supported_locations': [choice[0] for choice in PredictionRequest._meta.get_field('location').choices],
        }
        
        return Response(info, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return Response(
            {'error': 'Failed to fetch model info'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
def batch_predict(request):
    """Make multiple predictions at once"""
    try:
        predictions_data = request.data.get('predictions', [])
        results = []
        
        for prop_data in predictions_data:
            serializer = PredictionInputSerializer(data=prop_data)
            if serializer.is_valid():
                property_data = {
                    'property_type': serializer.validated_data['property_type'],
                    'location': serializer.validated_data['location'],
                    'bedrooms': serializer.validated_data['bedrooms'],
                    'bathrooms': serializer.validated_data['bathrooms'],
                    'house_size': serializer.validated_data.get('house_size'),
                    'land_size': serializer.validated_data.get('land_size'),
                }
                
                result = predictor.predict_with_confidence(property_data)
                results.append({
                    'input': property_data,
                    'prediction': result['predicted_price'],
                    'confidence': result['confidence_score'],
                })
        
        return Response({'results': results}, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return Response(
            {'error': 'Batch prediction failed'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )