from django.contrib import admin
from .models import AddictionRecord


@admin.register(AddictionRecord)
class AddictionRecordAdmin(admin.ModelAdmin):
    """Admin interface for AddictionRecord model"""
    
    list_display = ['id', 'age', 'gender', 'predicted_addiction_risk', 
                    'addiction_probability', 'created_at']
    list_filter = ['predicted_addiction_risk', 'stress_level', 'gender', 'created_at']
    search_fields = ['id', 'age']
    readonly_fields = ['created_at', 'updated_at', 'addiction_probability', 
                      'predicted_addiction_risk']
    
    fieldsets = (
        ('User Information', {
            'fields': ('age', 'gender')
        }),
        ('Screen Time', {
            'fields': ('daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
                      'work_study_hours', 'weekend_screen_time')
        }),
        ('Well-being', {
            'fields': ('sleep_hours', 'stress_level')
        }),
        ('App Usage', {
            'fields': ('notifications_per_day', 'app_opens_per_day')
        }),
        ('Impact', {
            'fields': ('academic_work_impact',)
        }),
        ('Prediction Results', {
            'fields': ('predicted_addiction_risk', 'addiction_probability')
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at')
        })
    )
    
    def has_add_permission(self, request):
        # Records are created through predictions only, not manually
        return False
