"""
Flask REST API for Vehicle Health Monitoring
Main application file with endpoints for predictions, alerts, and service centers
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

from config import config
from services import PredictionService, AlertService, LocationService

# Initialize Flask app
app = Flask(__name__)

# Load configuration
env = os.getenv('FLASK_ENV', 'development')
app.config.from_object(config[env])

# Enable CORS
CORS(app, origins=app.config['CORS_ORIGINS'])

# Initialize services
print("Initializing services...")

# Prediction Service
prediction_service = PredictionService(
    lstm_model_path=app.config['LSTM_MODEL_PATH'],
    autoencoder_model_path=app.config['AUTOENCODER_MODEL_PATH']
)

# Alert Service (Twilio)
alert_service = AlertService(
    account_sid=app.config['TWILIO_ACCOUNT_SID'],
    auth_token=app.config['TWILIO_AUTH_TOKEN'],
    phone_number=app.config['TWILIO_PHONE_NUMBER'],
    rate_limit_seconds=app.config['ALERT_RATE_LIMIT_SECONDS']
)

# Location Service (Google Maps)
location_service = LocationService(
    api_key=app.config['GOOGLE_MAPS_API_KEY']
)

# Load ML models on startup
try:
    prediction_service.load_models()
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    print("⚠️ Application will start but predictions will fail until models are loaded.")

# ============================================================================
# ROUTES / ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Vehicle Health Monitoring API is running',
        'version': '1.0.0'
    }), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict vehicle health based on telemetry features
    
    Expected JSON body:
    {
        "features": [0.8, 0.9, 360, -25, 32, 65, 0.6, 2200, 0.25, 8, 45, 0.85]
    }
    
    Returns:
    {
        "kpis": {...},
        "component_health": {...},
        "degradation_contributors": [...],
        "maintenance_decision": {...},
        "should_alert": bool,
        "alert_severity": str
    }
    """
    try:
        # Validate request
        if not request.json:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request body must be JSON'
            }), 400
        
        features = request.json.get('features')
        
        if not features:
            return jsonify({
                'error': 'Missing features',
                'message': 'Request must include "features" array'
            }), 400
        
        if not isinstance(features, list):
            return jsonify({
                'error': 'Invalid features type',
                'message': 'Features must be an array of numbers'
            }), 400
        
        if len(features) != 12:
            return jsonify({
                'error': 'Invalid features length',
                'message': f'Expected 12 features, got {len(features)}'
            }), 400
        
        # Perform prediction
        result = prediction_service.predict(features)
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except ValueError as e:
        return jsonify({
            'error': 'Validation error',
            'message': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/api/service-centers', methods=['GET'])
def get_service_centers():
    """
    Get nearby BMW service centers
    
    Query parameters:
    - lat: latitude (required)
    - lng: longitude (required)
    - radius: search radius in meters (optional, default 10000)
    
    Returns:
    {
        "success": true,
        "count": int,
        "service_centers": [...]
    }
    """
    try:
        # Get query parameters
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        radius = request.args.get('radius', default=10000, type=int)
        
        # Validate parameters
        if lat is None or lng is None:
            return jsonify({
                'error': 'Missing parameters',
                'message': 'Both lat and lng parameters are required'
            }), 400
        
        if not (-90 <= lat <= 90):
            return jsonify({
                'error': 'Invalid latitude',
                'message': 'Latitude must be between -90 and 90'
            }), 400
        
        if not (-180 <= lng <= 180):
            return jsonify({
                'error': 'Invalid longitude',
                'message': 'Longitude must be between -180 and 180'
            }), 400
        
        # Get service centers
        service_centers = location_service.get_nearby_service_centers(
            latitude=lat,
            longitude=lng,
            radius=radius
        )
        
        return jsonify({
            'success': True,
            'count': len(service_centers),
            'service_centers': service_centers
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch service centers',
            'message': str(e)
        }), 500


@app.route('/api/alerts/send', methods=['POST'])
def send_alert():
    """
    Send SMS alert to user
    
    Expected JSON body:
    {
        "phone": "+1234567890",
        "failure_prob": 75.5,
        "rul": 25,
        "severity": "critical",
        "nearest_center": {...}  // optional
    }
    
    Returns:
    {
        "success": bool,
        "message": str
    }
    """
    try:
        # Validate request
        if not request.json:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request body must be JSON'
            }), 400
        
        phone = request.json.get('phone')
        failure_prob = request.json.get('failure_prob')
        rul = request.json.get('rul')
        severity = request.json.get('severity', 'normal')
        nearest_center = request.json.get('nearest_center')
        
        # Validate required fields
        if not phone:
            return jsonify({
                'error': 'Missing phone number',
                'message': 'Phone number is required'
            }), 400
        
        if failure_prob is None or rul is None:
            return jsonify({
                'error': 'Missing data',
                'message': 'failure_prob and rul are required'
            }), 400
        
        # Send alert
        result = alert_service.send_alert(
            to_phone=phone,
            failure_prob=failure_prob,
            rul=rul,
            severity=severity,
            nearest_center=nearest_center
        )
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to send alert',
            'message': str(e)
        }), 500


@app.route('/api/alerts/test', methods=['GET'])
def test_alert_service():
    """Test Twilio connection"""
    result = alert_service.test_connection()
    
    if result['success']:
        return jsonify(result), 200
    else:
        return jsonify(result), 500


@app.route('/api/place/<place_id>', methods=['GET'])
def get_place_details(place_id):
    """
    Get detailed information for a specific place
    
    Returns:
    {
        "success": bool,
        "details": {...}
    }
    """
    try:
        details = location_service.get_place_details(place_id)
        
        if details:
            return jsonify({
                'success': True,
                'details': details
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to fetch place details'
            }), 404
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch place details',
            'message': str(e)
        }), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


# Run application
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = app.config['DEBUG']
    
    print(f"\n{'='*60}")
    print(f"🚗 BMW Vehicle Health Monitoring API")
    print(f"{'='*60}")
    print(f"Environment: {env}")
    print(f"Port: {port}")
    print(f"Debug: {debug}")
    print(f"CORS Origins: {app.config['CORS_ORIGINS']}")
    print(f"{'='*60}\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
