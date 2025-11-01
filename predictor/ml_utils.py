"""
ML utilities for property price prediction in Django
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class PropertyPricePredictor:
    """Wrapper class for ML model predictions"""
    
    def __init__(self):
        self.models = None
        self.le_property = None
        self.le_location = None
        self.feature_names = None
        self.stats = None
        self.model_dir = settings.ML_MODEL_DIR
        self.load_models()
    
    def load_models(self):
        """Load all trained models and encoders"""
        try:
            self.models = joblib.load(self.model_dir / 'property_models.pkl')
            self.le_property = joblib.load(self.model_dir / 'property_type_encoder.pkl')
            self.le_location = joblib.load(self.model_dir / 'location_encoder.pkl')
            self.feature_names = joblib.load(self.model_dir / 'feature_names.pkl')
            self.stats = joblib.load(self.model_dir / 'model_stats.pkl')
            logger.info("ML models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def engineer_features(self, data):
        """Engineer features for prediction"""
        # Calculate derived features
        data['bath_bed_ratio'] = data['bathroom'] / data['Bedroom']
        data['total_area'] = data.get('House size', 0) + data.get('Land size', 0)
        
        # Get premium values from training stats
        location_premiums = {loc: self.stats.get('location_premium', {}).get(loc, 0) 
                            for loc in self.le_location.classes_}
        property_premiums = {prop: self.stats.get('property_premium', {}).get(prop, 0) 
                            for prop in self.le_property.classes_}
        
        data['location_premium'] = location_premiums.get(data['Location'], 0)
        data['property_premium'] = property_premiums.get(data['propertyType'], 0)
        
        return data
    
    def prepare_input(self, property_data):
        """
        Prepare input data for prediction
        
        Args:
            property_data: dict with keys:
                - property_type: str
                - location: str
                - bedrooms: int
                - bathrooms: int
                - house_size: float (optional)
                - land_size: float (optional)
        
        Returns:
            pandas DataFrame ready for prediction
        """
        # Map input to feature format
        data = {
            'propertyType': property_data['property_type'],
            'Location': property_data['location'],
            'Bedroom': property_data['bedrooms'],
            'bathroom': property_data['bathrooms'],
            'House size': property_data.get('house_size', self.stats['median_values']['House size']),
            'Land size': property_data.get('land_size', self.stats['median_values']['Land size'])
        }
        
        # Engineer features
        data = self.engineer_features(data)
        
        # Create DataFrame with all required features
        df = pd.DataFrame([data])
        
        # Encode categorical variables
        try:
            df['propertyType'] = self.le_property.transform([data['propertyType']])[0]
            df['Location'] = self.le_location.transform([data['Location']])[0]
        except ValueError as e:
            logger.error(f"Encoding error: {str(e)}")
            raise ValueError(f"Invalid property type or location: {str(e)}")
        
        # Ensure all features are present in correct order
        df = df[self.feature_names]
        
        return df
    
    def ensemble_predict(self, X):
        """Make ensemble prediction"""
        scaler = self.models['scaler']
        X_scaled = scaler.transform(X)
        
        # Get predictions from all models
        rf_pred = self.models['rf'].predict(X)
        gb_pred = self.models['gb'].predict(X)
        ridge_pred = self.models['ridge'].predict(X_scaled)
        
        # Weighted ensemble (RF: 50%, GB: 30%, Ridge: 20%)
        ensemble_pred = 0.5 * rf_pred + 0.3 * gb_pred + 0.2 * ridge_pred
        
        return ensemble_pred[0]
    
    def predict_with_confidence(self, property_data):
        """
        Make prediction with confidence interval
        
        Returns:
            dict with:
                - predicted_price: float
                - confidence_score: float (0-1)
                - price_range_min: float
                - price_range_max: float
                - feature_importance: dict
        """
        try:
            # Prepare input
            X = self.prepare_input(property_data)
            
            # Get ensemble prediction
            predicted_price = self.ensemble_predict(X)
            
            # Get predictions from individual models for confidence calculation
            rf_pred = self.models['rf'].predict(X)[0]
            gb_pred = self.models['gb'].predict(X)[0]
            
            # Calculate confidence based on model agreement
            predictions = [rf_pred, gb_pred, predicted_price]
            std_dev = np.std(predictions)
            mean_pred = np.mean(predictions)
            
            # Confidence score (inverse of coefficient of variation)
            cv = std_dev / mean_pred if mean_pred > 0 else 1
            confidence_score = max(0, min(1, 1 - cv))
            
            # Calculate prediction interval (using RF's std from estimators)
            tree_predictions = np.array([
                tree.predict(X)[0] for tree in self.models['rf'].estimators_
            ])
            std_error = np.std(tree_predictions)
            
            # 95% confidence interval
            margin = 1.96 * std_error
            price_range_min = max(0, predicted_price - margin)
            price_range_max = predicted_price + margin
            
            # Get feature importance
            feature_importance = {
                name: float(importance) 
                for name, importance in zip(
                    self.feature_names, 
                    self.models['rf'].feature_importances_
                )
            }
            
            return {
                'predicted_price': float(predicted_price),
                'confidence_score': float(confidence_score),
                'price_range_min': float(price_range_min),
                'price_range_max': float(price_range_max),
                'feature_importance': feature_importance,
                'model_metrics': self.stats['metrics']
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def get_similar_properties(self, property_data, n=5):
        """
        Get similar properties from training data
        (This would require storing training data or recent predictions)
        """
        # This is a placeholder - implement based on your needs
        return []
    
    def explain_prediction(self, property_data, prediction_result):
        """Generate human-readable explanation of prediction"""
        explanations = []
        
        # Location impact
        location = property_data['location']
        explanations.append(
            f"Location ({location}): Premium area with high property values"
        )
        
        # Property type impact
        prop_type = property_data['property_type']
        explanations.append(
            f"Property Type ({prop_type}): Influences base price significantly"
        )
        
        # Size impact
        if property_data.get('house_size'):
            explanations.append(
                f"House Size: {property_data['house_size']:.0f} m² contributes to overall value"
            )
        
        # Bedrooms/Bathrooms
        explanations.append(
            f"Rooms: {property_data['bedrooms']} bedrooms and "
            f"{property_data['bathrooms']} bathrooms affect pricing"
        )
        
        # Top contributing features
        feature_importance = prediction_result['feature_importance']
        top_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        explanations.append(
            f"Key factors: {', '.join([f[0] for f in top_features])}"
        )
        
        return explanations

# Create singleton instance
predictor = PropertyPricePredictor()