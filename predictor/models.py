"""
Django models for property price prediction
"""

from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone

class PropertyType(models.TextChoices):
    """Property type choices"""
    APARTMENT = 'Apartment', 'Apartment'
    TOWNHOUSE = 'Townhouse', 'Townhouse'
    VACANT_LAND = 'Vacant Land', 'Vacant Land'
    COMMERCIAL = 'Commercial Property', 'Commercial Property'
    INDUSTRIAL = 'Industrial Property', 'Industrial Property'

class Location(models.TextChoices):
    """Location choices based on dataset"""
    KAREN = 'Karen', 'Karen'
    KILIMANI = 'Kilimani', 'Kilimani'
    KILELESHWA = 'Kileleshwa', 'Kileleshwa'
    KITISURU = 'Kitisuru', 'Kitisuru'
    LAVINGTON = 'Lavington', 'Lavington'
    LORESHO = 'Loresho', 'Loresho'
    MUTHAIGA = 'Muthaiga', 'Muthaiga'
    MUTHAIGA_NORTH = 'Muthaiga North', 'Muthaiga North'
    NYARI = 'Nyari', 'Nyari'
    PARKLANDS = 'Parklands', 'Parklands'
    RIVERSIDE = 'Riverside', 'Riverside'
    ROSSLYN = 'Rosslyn', 'Rosslyn'
    RUNDA = 'Runda', 'Runda'
    THIGIRI = 'Thigiri', 'Thigiri'
    WESTLANDS = 'Westlands', 'Westlands'
    KYUNA = 'Kyuna', 'Kyuna'
    KABETE = 'Kabete', 'Kabete'
    LOWER_KABETE = 'Lower Kabete', 'Lower Kabete'
    KIAMBU_ROAD = 'Kiambu Road', 'Kiambu Road'
    ONGATA_RONGAI = 'Ongata Rongai', 'Ongata Rongai'
    NGONG_RD = 'Ngong Rd', 'Ngong Rd'
    NAIROBI_WEST = 'Nairobi West', 'Nairobi West'
    SYOKIMAU = 'Syokimau', 'Syokimau'
    THOME = 'Thome', 'Thome'
    WAITHAKA = 'Waithaka', 'Waithaka'
    MOMBASA_RD = 'Mombasa Rd', 'Mombasa Rd'

class PredictionRequest(models.Model):
    """Model to store prediction requests and results"""
    
    # Input features
    property_type = models.CharField(
        max_length=50,
        choices=PropertyType.choices,
        help_text="Type of property"
    )
    location = models.CharField(
        max_length=50,
        choices=Location.choices,
        help_text="Property location in Nairobi"
    )
    bedrooms = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(20)],
        help_text="Number of bedrooms"
    )
    bathrooms = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(20)],
        help_text="Number of bathrooms"
    )
    house_size = models.FloatField(
        validators=[MinValueValidator(0)],
        null=True,
        blank=True,
        help_text="House size in square meters"
    )
    land_size = models.FloatField(
        validators=[MinValueValidator(0)],
        null=True,
        blank=True,
        help_text="Land size in square meters"
    )
    
    # Prediction results
    predicted_price = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Predicted price in KSh"
    )
    confidence_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Model confidence score (0-1)"
    )
    price_range_min = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Minimum predicted price"
    )
    price_range_max = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Maximum predicted price"
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    session_id = models.CharField(max_length=255, null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['location']),
            models.Index(fields=['property_type']),
        ]
    
    def __str__(self):
        return f"{self.property_type} in {self.location} - KSh {self.predicted_price:,.0f}"
    
    @property
    def price_formatted(self):
        """Return formatted price"""
        if self.predicted_price:
            return f"KSh {self.predicted_price:,.0f}"
        return "N/A"
    
    @property
    def features_dict(self):
        """Return features as dictionary"""
        return {
            'property_type': self.property_type,
            'location': self.location,
            'bedrooms': self.bedrooms,
            'bathrooms': self.bathrooms,
            'house_size': self.house_size,
            'land_size': self.land_size,
        }

class UserFeedback(models.Model):
    """Model to collect user feedback on predictions"""
    
    prediction = models.ForeignKey(
        PredictionRequest,
        on_delete=models.CASCADE,
        related_name='feedback'
    )
    rating = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text="Rating from 1-5 stars"
    )
    comment = models.TextField(blank=True, help_text="Optional feedback comment")
    actual_price = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Actual price if known"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Feedback for {self.prediction} - {self.rating} stars"

class ModelMetrics(models.Model):
    """Store model performance metrics"""
    
    model_version = models.CharField(max_length=50)
    mae = models.FloatField(help_text="Mean Absolute Error")
    rmse = models.FloatField(help_text="Root Mean Squared Error")
    r2_score = models.FloatField(help_text="R² Score")
    mape = models.FloatField(help_text="Mean Absolute Percentage Error")
    training_date = models.DateTimeField(default=timezone.now)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-training_date']
        verbose_name_plural = "Model Metrics"
    
    def __str__(self):
        return f"Model {self.model_version} - R²: {self.r2_score:.4f}"