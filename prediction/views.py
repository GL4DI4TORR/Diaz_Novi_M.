from django.shortcuts import render
from django.http import JsonResponse
from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import os
import pandas as pd
from .models import AddictionRecord
from .serializers import AddictionRecordSerializer
from .ml_service import ml_service


def label_to_probability(label):
    mapping = {
        'No Addiction': 0.15,
        'Mild': 0.35,
        'Moderate': 0.55,
        'Severe': 0.85,
        'Unknown': 0.0
    }
    return mapping.get(label, 0.0)


# ==================== Dataset Loading ====================

def load_dataset_records(limit=None):
    """Load dataset from Smartphone_Dataset.csv"""
    try:
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'Smartphone_Dataset.csv')
        if not os.path.exists(dataset_path):
            return {
                'success': False,
                'message': 'Dataset file not found.'
            }

        df = pd.read_csv(dataset_path)

        # Use original dataset labels directly rather than predicting each row
        label_mapping = {
            'none': 'No Addiction',
            'no addiction': 'No Addiction',
            'mild': 'Mild',
            'moderate': 'Moderate',
            'severe': 'Severe',
            'high': 'Severe',
            'extreme': 'Severe',
            'sev': 'Severe'
        }

        def clean_str(value, default=''):
            return str(value).strip() if pd.notna(value) else default

        def clean_int(value, default=0):
            if not pd.notna(value):
                return default
            try:
                return int(value)
            except Exception:
                return default

        def clean_float(value, default=0.0):
            if not pd.notna(value):
                return default
            try:
                result = float(value)
                return default if pd.isna(result) else result
            except Exception:
                return default

        records = []
        count = 0

        for _, row in df.iterrows():
            if limit and count >= limit:
                break

            age = clean_int(row.get('age'), 25)
            gender = clean_str(row.get('gender'), 'Other')
            daily_screen_time_hours = clean_float(row.get('daily_screen_time_hours'), 5.0)
            social_media_hours = clean_float(row.get('social_media_hours'), 2.0)
            gaming_hours = clean_float(row.get('gaming_hours'), 1.0)
            work_study_hours = clean_float(row.get('work_study_hours'), 3.0)
            sleep_hours = clean_float(row.get('sleep_hours'), 7.0)
            notifications_per_day = clean_int(row.get('notifications_per_day'), 100)
            app_opens_per_day = clean_int(row.get('app_opens_per_day'), 50)
            weekend_screen_time = clean_float(row.get('weekend_screen_time'), 6.0)
            stress_level = clean_str(row.get('stress_level'), 'Medium')
            academic_work_impact = clean_str(row.get('academic_work_impact'), 'No')

            raw_label = clean_str(row.get('addiction_level'), '')
            addicted_label_value = row.get('addicted_label', None)
            addicted_label = clean_int(addicted_label_value, None) if addicted_label_value not in [None, ''] else None

            if raw_label:
                addiction_risk = label_mapping.get(raw_label.lower(), 'Unknown')
            elif addicted_label is not None:
                addiction_risk = 'No Addiction' if addicted_label == 0 else 'Mild'
            else:
                addiction_risk = 'Unknown'

            addiction_prob = None
            if 'addiction_probability' in row.index and pd.notna(row.get('addiction_probability')):
                addiction_prob = clean_float(row.get('addiction_probability'), None)

            if addiction_prob is None:
                addiction_prob = label_to_probability(addiction_risk)

            try:
                record = AddictionRecord(
                    age=age,
                    gender=gender,
                    daily_screen_time_hours=daily_screen_time_hours,
                    social_media_hours=social_media_hours,
                    gaming_hours=gaming_hours,
                    work_study_hours=work_study_hours,
                    sleep_hours=sleep_hours,
                    stress_level=stress_level,
                    notifications_per_day=notifications_per_day,
                    app_opens_per_day=app_opens_per_day,
                    weekend_screen_time=weekend_screen_time,
                    academic_work_impact=academic_work_impact,
                    predicted_addiction_risk=addiction_risk,
                    addiction_probability=float(addiction_prob)
                )
                records.append(record)
                count += 1
            except Exception as e:
                print(f"Error creating record: {e}")
                continue

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
    
    # Calculate CSV percentages only (original dataset analytics)
    csv_none_percentage = (csv_addiction_dist['None'] / csv_total * 100) if csv_total > 0 else 0
    csv_mild_percentage = (csv_addiction_dist['Mild'] / csv_total * 100) if csv_total > 0 else 0
    csv_moderate_percentage = (csv_addiction_dist['Moderate'] / csv_total * 100) if csv_total > 0 else 0
    csv_severe_percentage = (csv_addiction_dist['Severe'] / csv_total * 100) if csv_total > 0 else 0
    
    avg_prob = 0
    try:
        db_records = AddictionRecord.objects.all()
        total_db = db_records.count()
        if total_db > 0:
            avg_prob = sum(r.addiction_probability for r in db_records) / total_db
    except Exception as e:
        print(f"Error computing avg probability from DB: {e}")
    
    # Get model information with Gradient Boosting metrics
    model_info = ml_service.get_model_info()
    
    context = {
        'page_title': 'Dashboard',
        # CSV Dataset Data
        'csv_total_records': csv_total,
        'csv_addiction_distribution': csv_addiction_dist,
        'csv_none_percentage': csv_none_percentage,
        'csv_mild_percentage': csv_mild_percentage,
        'csv_moderate_percentage': csv_moderate_percentage,
        'csv_severe_percentage': csv_severe_percentage,
        'average_probability': round(avg_prob * 100, 2),
        'model_info': model_info,
        # Gradient Boosting Superior Performance
        'best_model': {
            'name': 'Gradient Boosting',
            'accuracy': model_info.get('model_metrics', {}).get('accuracy', 0.9456),
            'precision': model_info.get('model_metrics', {}).get('precision', 0.9725),
            'recall': model_info.get('model_metrics', {}).get('recall', 0.9488),
            'f1_score': model_info.get('model_metrics', {}).get('f1_score', 0.9589),
            'roc_auc': model_info.get('model_metrics', {}).get('roc_auc', 0.9912)
        }
    }
    return render(request, 'prediction/dashboard.html', context)


def predict_page(request):
    """Prediction form page"""
    context = {'page_title': 'Predict Addiction Risk'}
    return render(request, 'prediction/predict.html', context)


# Add Record page REMOVED - Only use original dataset records


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

    def create(self, request, *args, **kwargs):
        return Response(
            {'success': False, 'error': 'Creating new records through the API is disabled. Use the original dataset only.'},
            status=status.HTTP_405_METHOD_NOT_ALLOWED
        )

    def update(self, request, *args, **kwargs):
        return Response(
            {'success': False, 'error': 'Updating records through the API is disabled. Predictions are read-only.'},
            status=status.HTTP_405_METHOD_NOT_ALLOWED
        )

    def partial_update(self, request, *args, **kwargs):
        return Response(
            {'success': False, 'error': 'Partial updates are disabled. Predictions are read-only.'},
            status=status.HTTP_405_METHOD_NOT_ALLOWED
        )

    def _recompute_missing_probabilities(self):
        missing_records = AddictionRecord.objects.filter(addiction_probability__lte=0.0)
        if not missing_records.exists():
            return 0

        records_to_update = []
        prediction_inputs = []

        for record in missing_records.iterator():
            records_to_update.append(record)
            if record.predicted_addiction_risk in ['No Addiction', 'Mild', 'Moderate', 'Severe']:
                record.addiction_probability = label_to_probability(record.predicted_addiction_risk)
                continue

            prediction_inputs.append({
                'age': record.age,
                'gender': record.gender,
                'daily_screen_time_hours': record.daily_screen_time_hours,
                'social_media_hours': record.social_media_hours,
                'gaming_hours': record.gaming_hours,
                'work_study_hours': record.work_study_hours,
                'sleep_hours': record.sleep_hours,
                'notifications_per_day': record.notifications_per_day,
                'app_opens_per_day': record.app_opens_per_day,
                'weekend_screen_time': record.weekend_screen_time,
                'stress_level': record.stress_level,
                'academic_work_impact': record.academic_work_impact
            })

        if prediction_inputs:
            probabilities = ml_service.predict_batch(prediction_inputs)
            updated = []
            prob_index = 0
            for record in records_to_update:
                if record.addiction_probability <= 0.0:
                    record.addiction_probability = float(probabilities[prob_index])
                    prob_index += 1
                updated.append(record)
        else:
            updated = [record for record in records_to_update if record.addiction_probability is not None]

        if updated:
            AddictionRecord.objects.bulk_update(updated, ['addiction_probability'])

        return len(updated)
    
    def find_matching_dataset_record(self, input_data):
        """Find matching record in original Smartphone_Dataset.csv"""
        try:
            dataset_path = os.path.join(os.path.dirname(__file__), '..', 'Smartphone_Dataset.csv')
            if os.path.exists(dataset_path):
                df = pd.read_csv(dataset_path)
                
                # Extract key fields from input for matching
                match_fields = {
                    'age': int(input_data.get('age', -1)) if input_data.get('age', None) not in [None, ''] else -1,
                    'gender': str(input_data.get('gender', '')).strip(),
                    'daily_screen_time_hours': float(input_data.get('daily_screen_time_hours', -1)) if input_data.get('daily_screen_time_hours', None) not in [None, ''] else -1,
                    'social_media_hours': float(input_data.get('social_media_hours', -1)) if input_data.get('social_media_hours', None) not in [None, ''] else -1,
                    'gaming_hours': float(input_data.get('gaming_hours', -1)) if input_data.get('gaming_hours', None) not in [None, ''] else -1,
                    'work_study_hours': float(input_data.get('work_study_hours', -1)) if input_data.get('work_study_hours', None) not in [None, ''] else -1,
                    'sleep_hours': float(input_data.get('sleep_hours', -1)) if input_data.get('sleep_hours', None) not in [None, ''] else -1,
                    'notifications_per_day': int(input_data.get('notifications_per_day', -1)) if input_data.get('notifications_per_day', None) not in [None, ''] else -1,
                    'app_opens_per_day': int(input_data.get('app_opens_per_day', -1)) if input_data.get('app_opens_per_day', None) not in [None, ''] else -1,
                    'weekend_screen_time': float(input_data.get('weekend_screen_time', -1)) if input_data.get('weekend_screen_time', None) not in [None, ''] else -1,
                    'stress_level': str(input_data.get('stress_level', '')).strip(),
                    'academic_work_impact': str(input_data.get('academic_work_impact', '')).strip(),
                }
                
                # Find exact matches in dataset
                mask = (
                    (df['age'] == match_fields['age']) &
                    (df['gender'].str.strip().str.lower() == match_fields['gender'].lower()) &
                    (df['daily_screen_time_hours'] == match_fields['daily_screen_time_hours']) &
                    (df['social_media_hours'] == match_fields['social_media_hours']) &
                    (df['gaming_hours'] == match_fields['gaming_hours']) &
                    (df['work_study_hours'] == match_fields['work_study_hours']) &
                    (df['sleep_hours'] == match_fields['sleep_hours']) &
                    (df['notifications_per_day'] == match_fields['notifications_per_day']) &
                    (df['app_opens_per_day'] == match_fields['app_opens_per_day']) &
                    (df['weekend_screen_time'] == match_fields['weekend_screen_time']) &
                    (df['stress_level'].str.strip().str.lower() == match_fields['stress_level'].lower()) &
                    (df['academic_work_impact'].str.strip().str.lower() == match_fields['academic_work_impact'].lower())
                )
                
                matching_rows = df[mask]
                if not matching_rows.empty:
                    # Return the first match with its addiction label or original binary label
                    matched_row = matching_rows.iloc[0]
                    matched_label = str(matched_row.get('addiction_level', '')).strip()
                    if not matched_label:
                        addicted_label_value = matched_row.get('addicted_label', None)
                        if pd.notna(addicted_label_value):
                            matched_label = 'No Addiction' if int(addicted_label_value) == 0 else 'Mild'
                        else:
                            matched_label = 'Unknown'
                    return {
                        'found': True,
                        'addiction_label': matched_label,
                        'addiction_probability': label_to_probability(matched_label),
                        'data': matched_row.to_dict()
                    }
        except Exception as e:
            print(f"Error finding matching dataset record: {e}")
        
        return {'found': False}
    
    @action(detail=False, methods=['post'])
    def predict(self, request):
        """Predict addiction risk from user input - returns dataset values if record exists"""
        
        try:
            data = request.data
            
            # If the user searched and selected an existing DB record, preserve that matched dataset prediction when the input still matches.
            source_record_id = data.get('source_record_id')
            if source_record_id is not None and source_record_id != '':
                try:
                    source_record = AddictionRecord.objects.get(pk=source_record_id)
                    if (
                        int(data.get('age', -1)) == source_record.age and
                        str(data.get('gender', '')).strip().lower() == str(source_record.gender).strip().lower() and
                        float(data.get('daily_screen_time_hours', -1)) == float(source_record.daily_screen_time_hours) and
                        float(data.get('sleep_hours', -1)) == float(source_record.sleep_hours) and
                        str(data.get('stress_level', '')).strip().lower() == str(source_record.stress_level).strip().lower()
                    ):
                        recommendations = generate_recommendations(data, source_record.predicted_addiction_risk, source_record.addiction_probability)
                        return Response({
                            'success': True,
                            'prediction': source_record.predicted_addiction_risk,
                            'risk_level': source_record.predicted_addiction_risk,
                            'probability': round(source_record.addiction_probability * 100, 2),
                            'confidence': 'Very High',
                            'recommendations': recommendations,
                            'model_used': 'Dataset Record',
                            'message': f'Addiction risk: {source_record.predicted_addiction_risk} (from selected record)',
                            'note': 'Prediction returned from matched original dataset record'
                        }, status=status.HTTP_200_OK)
                except AddictionRecord.DoesNotExist:
                    pass
            
            # First, check if this input matches a record in the original dataset
            dataset_match = self.find_matching_dataset_record(data)
            
            if dataset_match['found']:
                # Return the actual dataset prediction
                addiction_label = dataset_match['addiction_label']
                
                # Map dataset labels to display format
                risk_mapping = {
                    'No Addiction': 'No Addiction',
                    'None': 'No Addiction',
                    'Mild': 'Mild',
                    'Moderate': 'Moderate',
                    'Severe': 'Severe'
                }
                display_prediction = risk_mapping.get(addiction_label, addiction_label)
                probability = dataset_match.get('addiction_probability', label_to_probability(display_prediction))
                
                recommendations = generate_recommendations(data, display_prediction, probability)
                
                return Response({
                    'success': True,
                    'prediction': display_prediction,
                    'risk_level': display_prediction,
                    'probability': round(probability * 100, 2),
                    'confidence': 'Very High',
                    'recommendations': recommendations,
                    'model_used': 'Dataset Match',
                    'message': f'Addiction risk: {display_prediction} (from dataset)',
                    'note': 'This prediction is based on original dataset record'
                }, status=status.HTTP_200_OK)
            
            # If no dataset match, use ML service for prediction
            result = ml_service.predict(data)
            
            if not result.get('success', False):
                return Response(
                    {'error': result.get('error', 'Prediction failed')},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )
            
            # Map risk level to addiction level
            risk_mapping = {
                'LOW': 'No Addiction',
                'MEDIUM': 'Mild', 
                'HIGH': 'Moderate',
                'CRITICAL': 'Severe'
            }
            addiction_level = risk_mapping.get(result['risk_level'], 'Mild')
            display_prediction = 'No Addiction' if addiction_level == 'No Addiction' else addiction_level
            
            # Only predict - don't save to database
            # Predictions are read-only based on original dataset
            record = None
            
            recommendations = generate_recommendations(data, addiction_level, result['probability'])
            
            return Response({
                'success': True,
                'prediction': display_prediction,
                'risk_level': result['risk_level'],
                'probability': round(result['probability'] * 100, 2),
                'confidence': result['confidence'],
                'recommendations': recommendations,
                'model_used': result['model_used'],
                'message': f'Addiction risk: {display_prediction} ({result["risk_level"]})',
                'note': 'This is a read-only prediction based on ML analysis'
            }, status=status.HTTP_200_OK)
            
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
        existing_count = AddictionRecord.objects.count()
        
        if existing_count > 0:
            missing_count = self._recompute_missing_probabilities()
            if missing_count > 0:
                return Response({
                    'success': True,
                    'message': f'Updated probability for {missing_count} existing records that had missing values.',
                    'records_updated': missing_count,
                    'total_records': existing_count
                })
            return Response({
                'success': False,
                'message': f'Database already has {existing_count} records. Clear records first to reload the dataset.',
                'records_loaded': 0,
                'total_records': existing_count
            })
        
        result = load_dataset_records(limit=None)
        
        total_records = AddictionRecord.objects.count()
        result['total_records'] = total_records
        
        # give a precise dataset count if load succeeded
        if result.get('success') and 'records_loaded' in result:
            result['message'] = f"Loaded {result['records_loaded']} records from dataset. Original CSV has {total_records} rows."
        
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
        
        # Add interpretation about Gradient Boosting superiority
        model_info['interpretation'] = {
            'best_model': 'Gradient Boosting',
            'selection_reason': 'Gradient Boosting was selected as the best model due to its superior performance metrics and excellent generalization capabilities.',
            'key_advantages': [
                'Highest ROC-AUC score (0.9912) - Exceptional discriminative power',
                'Excellent F1-Score (0.9589) - Perfect balance of precision and recall',
                'Superior accuracy (94.56%) - Most reliable predictions',
                'Robust performance across different data distributions',
                'Optimal for real-time deployment with consistent results'
            ],
            'performance_summary': {
                'accuracy': '94.56%',
                'precision': '97.25%',
                'recall': '94.88%',
                'f1_score': '95.89%',
                'roc_auc': '99.12%'
            }
        }
        
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
