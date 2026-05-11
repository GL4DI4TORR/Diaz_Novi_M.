from rest_framework import serializers
from .models import AddictionRecord


class AddictionRecordSerializer(serializers.ModelSerializer):
    """Serializer for AddictionRecord model"""
    
    class Meta:
        model = AddictionRecord
        fields = '__all__'
        read_only_fields = ['id', 'predicted_addiction_risk', 'addiction_probability', 
                           'created_at', 'updated_at']


class PredictionInputSerializer(serializers.Serializer):
    """Serializer for prediction input validation"""
    
    age = serializers.IntegerField(min_value=5, max_value=100)
    gender = serializers.ChoiceField(choices=['Male', 'Female', 'Other'])
    daily_screen_time_hours = serializers.FloatField(min_value=0, max_value=24)
    social_media_hours = serializers.FloatField(min_value=0, max_value=24)
    gaming_hours = serializers.FloatField(min_value=0, max_value=24)
    work_study_hours = serializers.FloatField(min_value=0, max_value=24)
    sleep_hours = serializers.FloatField(min_value=0, max_value=24)
    stress_level = serializers.ChoiceField(choices=['Low', 'Medium', 'High'])
    notifications_per_day = serializers.IntegerField(min_value=0, max_value=10000)
    app_opens_per_day = serializers.IntegerField(min_value=0, max_value=10000)
    weekend_screen_time = serializers.FloatField(min_value=0, max_value=24)
    academic_work_impact = serializers.ChoiceField(choices=['Yes', 'No'])


class PredictionResponseSerializer(serializers.Serializer):
    """Serializer for prediction response"""
    
    prediction = serializers.CharField()
    probability = serializers.FloatField()
    record_id = serializers.IntegerField()
    message = serializers.CharField()
    recommendations = serializers.ListField()
