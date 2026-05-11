from django.shortcuts import render
from django.http import JsonResponse
from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Q
import os
import pandas as pd
from .models import AddictionRecord
from .serializers import AddictionRecordSerializer
from .ml_service import ml_service


# ==================== Dataset Loading ====================

def load_dataset_records(limit=None):
    """Load dataset from Smartphone_Dataset.csv"""
    try:
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'Smartphone_Dataset.csv')
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            
            records = []
            count = 0
            
            for _, row in df.iterrows():
                if limit and count >= limit:
                    break
                
                # Prepare data for ML prediction
                prediction_data = {
                    'age': int(row['age']) if pd.notna(row['age']) else 25,
                    'gender': str(row['gender']).strip() if pd.notna(row['gender']) else 'Other',
                    'daily_screen_time_hours': float(row['daily_screen_time_hours']) if pd.notna(row['daily_screen_time_hours']) else 5.0,
                    'social_media_hours': float(row['social_media_hours']) if pd.notna(row['social_media_hours']) else 2.0,
                    'gaming_hours': float(row['gaming_hours']) if pd.notna(row['gaming_hours']) else 1.0,
                    'work_study_hours': float(row['work_study_hours']) if pd.notna(row['work_study_hours']) else 3.0,
                    'sleep_hours': float(row['sleep_hours']) if pd.notna(row['sleep_hours']) else 7.0,
                    'notifications_per_day': int(row['notifications_per_day']) if pd.notna(row['notifications_per_day']) else 100,
                    'app_opens_per_day': int(row['app_opens_per_day']) if pd.notna(row['app_opens_per_day']) else 50,
                    'weekend_screen_time': float(row['weekend_screen_time']) if pd.notna(row['weekend_screen_time']) else 6.0,
                    'stress_level': str(row['stress_level']).strip() if pd.notna(row['stress_level']) else 'Medium',
                    'academic_work_impact': str(row['academic_work_impact']).strip() if pd.notna(row['academic_work_impact']) else 'No'
                }
                
                # Get ML prediction for accurate risk assessment
                ml_result = ml_service.predict(prediction_data)
                
                # Map ML risk levels to display format
                risk_mapping = {
                    'LOW': 'No Addiction',
                    'MEDIUM': 'Mild', 
                    'HIGH': 'Moderate',
                    'CRITICAL': 'Severe'
                }
                addiction_risk = risk_mapping.get(ml_result['risk_level'], 'Mild')
                addiction_prob = ml_result['probability']
                
                try:
                    # Handle NaN values by using defaults
                    record = AddictionRecord(
                        age=int(row['age']) if pd.notna(row['age']) else 25,
                        gender=str(row['gender']).strip() if pd.notna(row['gender']) else 'Other',
                        daily_screen_time_hours=float(row['daily_screen_time_hours']) if pd.notna(row['daily_screen_time_hours']) else 5.0,
                        social_media_hours=float(row['social_media_hours']) if pd.notna(row['social_media_hours']) else 2.0,
                        gaming_hours=float(row['gaming_hours']) if pd.notna(row['gaming_hours']) else 1.0,
                        work_study_hours=float(row['work_study_hours']) if pd.notna(row['work_study_hours']) else 3.0,
                        sleep_hours=float(row['sleep_hours']) if pd.notna(row['sleep_hours']) else 7.0,
                        stress_level=str(row['stress_level']).strip() if pd.notna(row['stress_level']) else 'Medium',
                        notifications_per_day=int(row['notifications_per_day']) if pd.notna(row['notifications_per_day']) else 100,
                        app_opens_per_day=int(row['app_opens_per_day']) if pd.notna(row['app_opens_per_day']) else 50,
                        weekend_screen_time=float(row['weekend_screen_time']) if pd.notna(row['weekend_screen_time']) else 6.0,
                        academic_work_impact=str(row['academic_work_impact']).strip() if pd.notna(row['academic_work_impact']) else 'No',
                        predicted_addiction_risk=addiction_risk,
                        addiction_probability=addiction_prob
                    )
                    records.append(record)
                    count += 1
                except Exception as e:
                    print(f"Error creating record: {e}")
                    continue
            
            # Check record count limit before loading dataset
            current_count = AddictionRecord.objects.count()
            if current_count >= 10000:
                return {
                    'success': False,
                    'message': 'Maximum record limit reached (10,000 records). Please delete some records before loading dataset.'
                }
            
            # Find existing IDs to reuse
            existing_ids = set(AddictionRecord.objects.values_list('id', flat=True))
            next_id = 1
            
            # Assign IDs to records, reusing deleted IDs
            for record in records:
                while next_id in existing_ids:
                    next_id += 1
                record.id = next_id
                existing_ids.add(next_id)
                next_id += 1
            
            # Bulk create all records at once
            if records:
                AddictionRecord.objects.bulk_create(records, batch_size=1000)
            
            return {
                'success': True,
                'records_loaded': count,
                'message': f'Loaded {count} records from dataset'
            }
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {
            'success': False,
            'message': str(e)
        }


# ==================== Page Views ====================

def index(request):
    """Home page"""
    model_info = ml_service.get_model_info()
    context = {
        'page_title': 'Smartphone Addiction Detection System',
        'model_loaded': model_info['model_loaded'],
        'model_info': model_info
    }
    return render(request, 'prediction/index.html', context)


def dashboard(request):
    """Dashboard with statistics"""
    import pandas as pd
    
    # CSV Dataset Statistics
    csv_addiction_dist = {'None': 0, 'Mild': 0, 'Moderate': 0, 'Severe': 0}
    csv_total = 0
    
    try:
        df = pd.read_csv('Smartphone_Dataset.csv')
        csv_total = len(df)
        
        # Normalize addiction levels and count them
        df['addiction_level'] = df['addiction_level'].fillna('None')
        df['addiction_level'] = df['addiction_level'].str.title().str.strip()
        
        # Map variations to standard levels
        level_mapping = {
            'None': 'None',
            'Mild': 'Mild', 
            'Moderate': 'Moderate',
            'Severe': 'Severe',
            'High': 'Severe',
            'Extreme': 'Severe',
            'Sev': 'Severe'
        }
        
        df['normalized_level'] = df['addiction_level'].map(level_mapping).fillna('None')
        csv_addiction_dist = df['normalized_level'].value_counts().to_dict()
        
        # Ensure all levels are present
        for level in ['None', 'Mild', 'Moderate', 'Severe']:
            if level not in csv_addiction_dist:
                csv_addiction_dist[level] = 0
                
    except Exception as e:
        print(f"Error loading CSV for dashboard: {e}")
    
    # Database Records Statistics (New Predictions)
    db_addiction_dist = {}
    db_total = AddictionRecord.objects.count()
    
    for level in ['None', 'Mild', 'Moderate', 'Severe']:
        if level == 'None':
            db_count = AddictionRecord.objects.filter(
                Q(predicted_addiction_risk='None') | Q(predicted_addiction_risk='No Addiction')
            ).count()
        else:
            db_count = AddictionRecord.objects.filter(predicted_addiction_risk=level).count()
        db_addiction_dist[level] = db_count
    
    # Calculate CSV percentages
    csv_none_percentage = (csv_addiction_dist['None'] / csv_total * 100) if csv_total > 0 else 0
    csv_mild_percentage = (csv_addiction_dist['Mild'] / csv_total * 100) if csv_total > 0 else 0
    csv_moderate_percentage = (csv_addiction_dist['Moderate'] / csv_total * 100) if csv_total > 0 else 0
    csv_severe_percentage = (csv_addiction_dist['Severe'] / csv_total * 100) if csv_total > 0 else 0
    
    # Calculate Database percentages
    db_none_percentage = (db_addiction_dist['None'] / db_total * 100) if db_total > 0 else 0
    db_mild_percentage = (db_addiction_dist['Mild'] / db_total * 100) if db_total > 0 else 0
    db_moderate_percentage = (db_addiction_dist['Moderate'] / db_total * 100) if db_total > 0 else 0
    db_severe_percentage = (db_addiction_dist['Severe'] / db_total * 100) if db_total > 0 else 0
    
    avg_prob = 0
    if db_total > 0:
        records = AddictionRecord.objects.all()
        avg_prob = sum(r.addiction_probability for r in records) / db_total
    
    context = {
        'page_title': 'Dashboard',
        # CSV Dataset Data
        'csv_total_records': csv_total,
        'csv_addiction_distribution': csv_addiction_dist,
        'csv_none_percentage': csv_none_percentage,
        'csv_mild_percentage': csv_mild_percentage,
        'csv_moderate_percentage': csv_moderate_percentage,
        'csv_severe_percentage': csv_severe_percentage,
        # Database Records Data
        'db_total_records': db_total,
        'db_addiction_distribution': db_addiction_dist,
        'db_none_percentage': db_none_percentage,
        'db_mild_percentage': db_mild_percentage,
        'db_moderate_percentage': db_moderate_percentage,
        'db_severe_percentage': db_severe_percentage,
        'average_probability': round(avg_prob * 100, 2),
        'model_info': ml_service.get_model_info()
    }
    return render(request, 'prediction/dashboard.html', context)


def predict_page(request):
    """Prediction form page"""
    context = {'page_title': 'Predict Addiction Risk'}
    return render(request, 'prediction/predict.html', context)


def add_record_page(request):
    """Add record page - similar to predict but saves to database"""
    context = {'page_title': 'Add New Record'}
    return render(request, 'prediction/add_record.html', context)


def records_page(request):
    """View all records with pagination"""
    all_records = AddictionRecord.objects.all().order_by('id')
    
    # Pagination
    page = request.GET.get('page', 1)
    paginator = Paginator(all_records, 50)  # 50 records per page
    
    try:
        records = paginator.page(page)
    except PageNotAnInteger:
        records = paginator.page(1)
    except EmptyPage:
        records = paginator.page(paginator.num_pages)
    
    # Add calculated percentage to each record
    for record in records:
        record.addiction_probability_percent = record.addiction_probability * 100
    
    context = {
        'page_title': 'Prediction Records',
        'records': records,
        'paginator': paginator,
        'page_obj': records,
        'is_paginated': paginator.num_pages > 1,
        'page_range': paginator.page_range,
    }
    return render(request, 'prediction/records.html', context)


# ==================== API Views ====================

class AddictionRecordViewSet(viewsets.ModelViewSet):
    """ViewSet for addiction records with prediction endpoint"""
    queryset = AddictionRecord.objects.all()
    serializer_class = AddictionRecordSerializer
    
    @action(detail=False, methods=['post'])
    def predict(self, request):
        """Predict addiction risk from user input"""
        
        try:
            data = request.data
            
            # Use ML service for prediction
            result = ml_service.predict(data)
            
            if not result.get('success', False):
                return Response(
                    {'error': result.get('error', 'Prediction failed')},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )
            
            # Map risk level to addiction level
            risk_mapping = {
                'LOW': 'None',
                'MEDIUM': 'Mild', 
                'HIGH': 'Moderate',
                'CRITICAL': 'Severe'
            }
            addiction_level = risk_mapping.get(result['risk_level'], 'Mild')
            display_prediction = 'No Addiction' if addiction_level == 'None' else addiction_level
            
            # Check record count limit and find the lowest available ID to reuse
            total_records = AddictionRecord.objects.count()
            if total_records >= 10000:
                return Response({
                    'success': False,
                    'error': 'Maximum record limit reached (10,000 records). Please delete some records to add new ones.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            existing_ids = set(AddictionRecord.objects.values_list('id', flat=True))
            next_id = 1
            while next_id in existing_ids:
                next_id += 1
            
            # Additional ID restriction - don't allow IDs above 10000
            if next_id > 10000:
                return Response({
                    'success': False,
                    'error': 'ID limit reached. Please delete some records to reuse IDs.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Save record with reused ID
            record = AddictionRecord.objects.create(
                id=next_id,  # Explicitly set the ID to reuse deleted IDs
                age=int(data['age']),
                gender=data['gender'],
                daily_screen_time_hours=float(data['daily_screen_time_hours']),
                social_media_hours=float(data['social_media_hours']),
                gaming_hours=float(data['gaming_hours']),
                work_study_hours=float(data['work_study_hours']),
                sleep_hours=float(data['sleep_hours']),
                stress_level=data['stress_level'],
                notifications_per_day=int(data['notifications_per_day']),
                app_opens_per_day=int(data['app_opens_per_day']),
                weekend_screen_time=float(data['weekend_screen_time']),
                academic_work_impact=data['academic_work_impact'],
                predicted_addiction_risk=addiction_level,
                addiction_probability=result['probability']
            )
            
            recommendations = generate_recommendations(data, addiction_level, result['probability'])
            
            return Response({
                'success': True,
                'prediction': display_prediction,
                'risk_level': result['risk_level'],
                'probability': round(result['probability'] * 100, 2),
                'confidence': result['confidence'],
                'record_id': record.id,
                'recommendations': recommendations,
                'model_used': result['model_used'],
                'message': f'Addiction risk: {display_prediction} ({result["risk_level"]})'
            }, status=status.HTTP_201_CREATED)
            
        except KeyError as e:
            return Response(
                {'error': f'Missing field: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    @action(detail=False, methods=['get'])
    def statistics(self, request):
        """Get prediction statistics"""
        total = AddictionRecord.objects.count()
        
        stats = {
            'total_records': total,
            'addiction_levels': {}
        }
        
        for level in ['None', 'Mild', 'Moderate', 'Severe']:
            count = AddictionRecord.objects.filter(predicted_addiction_risk=level).count()
            stats['addiction_levels'][level] = count
        
        if total > 0:
            records = AddictionRecord.objects.all()
            avg = sum(r.addiction_probability for r in records) / total
            stats['average_probability'] = round(avg * 100, 2)
        
        return Response(stats)
    
    @action(detail=True, methods=['get'])
    def recommendations(self, request, pk=None):
        """Get recommendations for a specific record"""
        try:
            record = self.get_object()
            recommendations = generate_recommendations({
                'daily_screen_time_hours': record.daily_screen_time_hours,
                'sleep_hours': record.sleep_hours,
                'stress_level': record.stress_level
            }, record.predicted_addiction_risk, record.addiction_probability)
            
            return Response({'recommendations': recommendations})
        except AddictionRecord.DoesNotExist:
            return Response(
                {'error': 'Record not found'},
                status=status.HTTP_404_NOT_FOUND
            )
    
    @action(detail=False, methods=['post'])
    def load_dataset(self, request):
        """Load sample dataset into records"""
        # Check if we already have dataset records
        existing_count = AddictionRecord.objects.count()
        
        if existing_count > 0:
            return Response({
                'success': True,
                'message': f'Database already has {existing_count} records. Clear records first to reload dataset.',
                'records_loaded': 0,
                'total_records': existing_count
            })
        
        # Load all dataset records (7500 total)
        result = load_dataset_records(limit=None)
        
        total_records = AddictionRecord.objects.count()
        result['total_records'] = total_records
        
        return Response(result)
    
    @action(detail=False, methods=['post'])
    def clear_records(self, request):
        """Clear all records and reset auto-increment counter"""
        count = AddictionRecord.objects.count()
        AddictionRecord.objects.all().delete()
        
        # Completely reset auto-increment by dropping and recreating the sequence
        from django.db import connection
        with connection.cursor() as cursor:
            # Delete all records from the sequence table for this table
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='prediction_addictionrecord'")
            # Reset the sequence to 0
            cursor.execute("INSERT INTO sqlite_sequence (name, seq) VALUES ('prediction_addictionrecord', 0)")
        
        return Response({
            'success': True,
            'message': f'Cleared {count} records from database and reset ID counter',
            'records_deleted': count
        })
    
    @action(detail=False, methods=['get'])
    def sample_data(self, request):
        """Get random sample user data from dataset"""
        try:
            dataset_path = os.path.join(os.path.dirname(__file__), '..', 'Smartphone_Dataset.csv')
            if os.path.exists(dataset_path):
                df = pd.read_csv(dataset_path)
                
                # Clean the data similar to load_dataset_records
                df = df.drop_duplicates()
                
                # Handle NaN values with defaults
                def clean_value(val, default):
                    return val if pd.notna(val) else default
                
                # Get a random row
                sample_row = df.sample(1).iloc[0]
                
                sample_data = {
                    'age': int(clean_value(sample_row['age'], 25)),
                    'gender': str(clean_value(sample_row['gender'], 'Other')).strip(),
                    'daily_screen_time_hours': float(clean_value(sample_row['daily_screen_time_hours'], 5.0)),
                    'social_media_hours': float(clean_value(sample_row['social_media_hours'], 2.0)),
                    'gaming_hours': float(clean_value(sample_row['gaming_hours'], 1.0)),
                    'work_study_hours': float(clean_value(sample_row['work_study_hours'], 3.0)),
                    'sleep_hours': float(clean_value(sample_row['sleep_hours'], 7.0)),
                    'notifications_per_day': int(clean_value(sample_row['notifications_per_day'], 100)),
                    'app_opens_per_day': int(clean_value(sample_row['app_opens_per_day'], 50)),
                    'weekend_screen_time': float(clean_value(sample_row['weekend_screen_time'], 6.0)),
                    'stress_level': str(clean_value(sample_row['stress_level'], 'Medium')).strip(),
                    'academic_work_impact': str(clean_value(sample_row['academic_work_impact'], 'No')).strip()
                }
                
                return Response({
                    'success': True,
                    'sample_data': sample_data,
                    'message': 'Random sample data loaded successfully'
                })
            else:
                return Response({
                    'success': False,
                    'error': 'Dataset file not found'
                }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def retrieve(self, request, pk=None):
        """Get a single record"""
        try:
            record = AddictionRecord.objects.get(pk=pk)
            serializer = AddictionRecordSerializer(record)
            return Response({
                'success': True,
                'record': serializer.data
            })
        except AddictionRecord.DoesNotExist:
            return Response({
                'success': False,
                'error': 'Record not found'
            }, status=status.HTTP_404_NOT_FOUND)
    
    def update(self, request, pk=None):
        """Update a single record"""
        try:
            record = AddictionRecord.objects.get(pk=pk)
            data = request.data
            
            # Update record fields
            record.age = data.get('age', record.age)
            record.gender = data.get('gender', record.gender)
            record.daily_screen_time_hours = data.get('daily_screen_time_hours', record.daily_screen_time_hours)
            record.social_media_hours = data.get('social_media_hours', record.social_media_hours)
            record.gaming_hours = data.get('gaming_hours', record.gaming_hours)
            record.work_study_hours = data.get('work_study_hours', record.work_study_hours)
            record.sleep_hours = data.get('sleep_hours', record.sleep_hours)
            record.notifications_per_day = data.get('notifications_per_day', record.notifications_per_day)
            record.app_opens_per_day = data.get('app_opens_per_day', record.app_opens_per_day)
            record.weekend_screen_time = data.get('weekend_screen_time', record.weekend_screen_time)
            record.stress_level = data.get('stress_level', record.stress_level)
            record.academic_work_impact = data.get('academic_work_impact', record.academic_work_impact)
            
            # Get new prediction without creating records
            try:
                # Use the same prediction logic as the main predict method
                result = ml_service.predict(data)
                if result.get('success', False):
                    # Map risk level to addiction level (same as predict method)
                    risk_mapping = {
                        'LOW': 'No Addiction',
                        'MEDIUM': 'Mild', 
                        'HIGH': 'Moderate',
                        'CRITICAL': 'Severe'
                    }
                    addiction_level = risk_mapping.get(result['risk_level'], 'Mild')
                    
                    record.predicted_addiction_risk = addiction_level
                    record.addiction_probability = result.get('probability', 0.0)
                else:
                    # If prediction fails, keep existing values
                    print(f"Prediction failed during update: {result.get('error', 'Unknown error')}")
                    pass
                
            except Exception as e:
                # If prediction fails, keep existing values
                print(f"Prediction error during update: {e}")
                pass
            
            record.save()
            
            serializer = AddictionRecordSerializer(record)
            return Response({
                'success': True,
                'record': serializer.data,
                'message': 'Record updated successfully'
            })
        except AddictionRecord.DoesNotExist:
            return Response({
                'success': False,
                'error': 'Record not found'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def destroy(self, request, pk=None):
        """Delete a single record"""
        try:
            record = AddictionRecord.objects.get(pk=pk)
            record.delete()
            return Response({
                'success': True,
                'message': 'Record deleted successfully'
            })
        except AddictionRecord.DoesNotExist:
            return Response({
                'success': False,
                'error': 'Record not found'
            }, status=status.HTTP_404_NOT_FOUND)
    
    @action(detail=False, methods=['get'])
    def model_info(self, request):
        """Get model information and feature importance"""
        model_info = ml_service.get_model_info()
        feature_importance = ml_service.get_feature_importance()
        
        return Response({
            'success': True,
            'model_info': model_info,
            'feature_importance': feature_importance
        })


def generate_recommendations(data, risk_level, probability):
    """Generate personalized recommendations"""
    
    recommendations = []
    
    screen_time = float(data.get('daily_screen_time_hours', 0))
    sleep = float(data.get('sleep_hours', 0))
    stress = data.get('stress_level', 'Medium')
    
    if risk_level in ['Moderate', 'Severe']:
        recommendations.append({
            'icon': '⚠️',
            'type': 'warning',
            'text': f'HIGH RISK: Your addiction level is {risk_level}. Consider professional support.'
        })
    
    if screen_time > 8:
        recommendations.append({
            'icon': '📱',
            'type': 'action',
            'text': 'Reduce screen time gradually. Target: 6-7 hours per day'
        })
    
    if sleep < 7:
        recommendations.append({
            'icon': '😴',
            'type': 'action',
            'text': 'Improve sleep: aim for 7-8 hours per night'
        })
    
    if stress == 'High':
        recommendations.append({
            'icon': '🧘',
            'type': 'action',
            'text': 'Manage stress: meditation, exercise, or counseling'
        })
    
    if not recommendations:
        recommendations.append({
            'icon': '✅',
            'type': 'success',
            'text': 'Excellent! Your smartphone usage is healthy.'
        })
    
    return recommendations


# ==================== API Endpoints ====================

@api_view(['GET'])
def model_info(request):
    """Get model information"""
    if MODEL_ARTIFACTS is None:
        return Response(
            {'error': 'Model not loaded'},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    
    metrics = MODEL_ARTIFACTS['metrics']
    return Response({
        'model_loaded': True,
        'metrics': {
            'accuracy': metrics.get('accuracy'),
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'f1_score': metrics.get('f1'),
            'roc_auc': metrics.get('roc_auc')
        }
    })
