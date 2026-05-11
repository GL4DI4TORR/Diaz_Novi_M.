from django.db import models

class AddictionRecord(models.Model):
    """Model to store user addiction prediction records"""
    
    # User Info
    age = models.IntegerField()
    gender = models.CharField(max_length=20, choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')])
    
    # Screen Time Features
    daily_screen_time_hours = models.FloatField()
    social_media_hours = models.FloatField()
    gaming_hours = models.FloatField()
    work_study_hours = models.FloatField()
    
    # Sleep & Well-being
    sleep_hours = models.FloatField()
    stress_level = models.CharField(max_length=20, choices=[('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High')])
    
    # App Usage
    notifications_per_day = models.IntegerField()
    app_opens_per_day = models.IntegerField()
    weekend_screen_time = models.FloatField()
    
    # Academic/Work
    academic_work_impact = models.CharField(max_length=20, choices=[('Yes', 'Yes'), ('No', 'No')])
    
    # Prediction Results
    predicted_addiction_risk = models.CharField(max_length=50, choices=[
        ('None', 'None'),
        ('Mild', 'Mild'),
        ('Moderate', 'Moderate'),
        ('Severe', 'Severe')
    ])
    addiction_probability = models.FloatField(default=0.0)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        
    def __str__(self):
        return f"User {self.id} - {self.predicted_addiction_risk} ({self.created_at.strftime('%Y-%m-%d')})"
