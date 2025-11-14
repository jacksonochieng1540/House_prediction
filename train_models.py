import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Run the training script
exec(open('property_ml_model.py').read())
