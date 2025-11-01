from django.apps import AppConfig

class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'
    verbose_name = 'Property Price Predictor'
    
    def ready(self):
        # Load ML models when Django starts
        try:
            from .ml_utils import predictor
            print("ML models loaded successfully")
        except Exception as e:
            print(f"Error loading ML models: {e}")