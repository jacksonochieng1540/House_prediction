from django.core.management.base import BaseCommand
from predictor.models import ModelMetrics

class Command(BaseCommand):
    help = 'Populate initial model metrics'

    def handle(self, *args, **kwargs):
        # Create initial model metrics
        ModelMetrics.objects.get_or_create(
            model_version='v1.0',
            defaults={
                'mae': 15000000,
                'rmse': 25000000,
                'r2_score': 0.95,
                'mape': 12.5,
                'is_active': True
            }
        )
        self.stdout.write(self.style.SUCCESS('Initial data populated'))