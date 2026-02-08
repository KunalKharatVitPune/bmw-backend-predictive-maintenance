"""
Configuration settings for the Vehicle Health Monitoring API
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'False') == 'True'
    
    # Model paths
    MODEL_PATH = os.getenv('MODEL_PATH', './models')
    LSTM_MODEL_PATH = os.path.join(MODEL_PATH, 'lstm_attention.h5')
    AUTOENCODER_MODEL_PATH = os.path.join(MODEL_PATH, 'autoencoder.h5')
    
    # CORS
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:5173').split(',')
    
    # Twilio Configuration
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '')
    TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER', '')
    
    # Google Maps
    GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', '')
    
    # Alert Thresholds
    ALERT_THRESHOLD_FAILURE = float(os.getenv('ALERT_THRESHOLD_FAILURE', '0.7'))
    ALERT_THRESHOLD_RUL = int(os.getenv('ALERT_THRESHOLD_RUL', '30'))
    
    # Rate Limiting for SMS
    ALERT_RATE_LIMIT_SECONDS = int(os.getenv('ALERT_RATE_LIMIT_SECONDS', '3600'))

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

# Config dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
