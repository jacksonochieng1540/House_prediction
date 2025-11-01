from django.core.management.base import BaseCommand
from predictor.models import ModelMetrics
import subprocess
import os

class Command(BaseCommand):
    help = 'Retrain the ML model with latest data'

    def handle(self, *args, **kwargs):
        self.stdout.write('Starting model retraining...')
        
        # Run training script
        result = subprocess.run(['python', 'train_model.py'], 
                                capture_output=True, text=True)
        
        if result.returncode == 0:
            self.stdout.write(self.style.SUCCESS('Model retrained successfully'))
            
            # Update database with new metrics
            # (Add logic to parse metrics and save to ModelMetrics)
        else:
            self.stdout.write(self.style.ERROR(f'Training failed: {result.stderr}'))