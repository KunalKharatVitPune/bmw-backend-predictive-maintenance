# BMW Vehicle Health Monitoring - Backend API

Flask REST API backend for BMW Vehicle Health Monitoring system with ML-powered predictive maintenance.

## Features

- **Vehicle Health Prediction**: LSTM + Autoencoder models for failure prediction and anomaly detection
- **SMS Alerts**: Twilio integration for critical maintenance notifications
- **Service Center Locator**: Google Maps API integration for nearby BMW service centers
- **RESTful API**: Clean API design with comprehensive error handling

## Project Structure

```
backend/
├── app.py                          # Main Flask application
├── config.py                       # Configuration management
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── services/
│   ├── __init__.py
│   ├── prediction_service.py      # ML inference service
│   ├── alert_service.py           # Twilio SMS service
│   └── location_service.py        # Google Maps service
└── models/
    ├── lstm_attention.h5          # Pre-trained LSTM model
    └── autoencoder.h5             # Pre-trained Autoencoder model
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
cd backend
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the backend directory (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# Flask
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# Twilio (optional - for SMS alerts)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890

# Google Maps (optional - for service center locator)
GOOGLE_MAPS_API_KEY=your_maps_api_key
```

**Note:** The app will work without Twilio/Google Maps credentials. Mock data will be used for service centers, and SMS alerts will be disabled.

### 5. Run the Server

```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
```http
GET /health
```

### Predict Vehicle Health
```http
POST /api/predict
Content-Type: application/json

{
  "features": [0.8, 0.9, 360, -25, 32, 65, 0.6, 2200, 0.25, 8, 45, 0.85]
}
```

**Features (12 values):**
1. State of Charge (0-1)
2. State of Health (0-1)
3. Battery Voltage (V)
4. Battery Current (A)
5. Battery Temperature (°C)
6. Motor Temperature (°C)
7. Motor Vibration (mm/s)
8. Motor RPM
9. Brake Pad Wear (0-1)
10. Power Stress (kW)
11. Usage Intensity Index
12. Health Trend (0-1)

**Response:**
```json
{
  "success": true,
  "data": {
    "kpis": {
      "failure_probability": 35.2,
      "remaining_useful_life": 125.5,
      "anomaly_score": 0.0023,
      "overall_health": 87.3
    },
    "component_health": { ... },
    "degradation_contributors": [ ... ],
    "maintenance_decision": { ... }
  }
}
```

### Get Nearby Service Centers
```http
GET /api/service-centers?lat=40.7128&lng=-74.0060&radius=10000
```

### Send SMS Alert
```http
POST /api/alerts/send
Content-Type: application/json

{
  "phone": "+1234567890",
  "failure_prob": 75.5,
  "rul": 25,
  "severity": "critical",
  "nearest_center": { ... }
}
```

## Testing the API

### Using cURL

```bash
# Health check
curl http://localhost:5000/health

# Prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.8, 0.9, 360, -25, 32, 65, 0.6, 2200, 0.25, 8, 45, 0.85]}'

# Service centers
curl "http://localhost:5000/api/service-centers?lat=40.7128&lng=-74.0060"
```

### Using Postman

Import the following requests:
1. **GET** `http://localhost:5000/health`
2. **POST** `http://localhost:5000/api/predict` with JSON body
3. **GET** `http://localhost:5000/api/service-centers?lat=40.7128&lng=-74.0060`

## Configuration

### Alert Thresholds

Modify in `.env`:
```env
ALERT_THRESHOLD_FAILURE=0.7    # 70% failure probability
ALERT_THRESHOLD_RUL=30         # 30 cycles remaining
```

### CORS Origins

Add allowed frontend origins:
```env
CORS_ORIGINS=http://localhost:5173,http://localhost:3000,https://yourdomain.com
```

## Deployment

### Production Configuration

1. Set `FLASK_ENV=production` in `.env`
2. Use a production WSGI server (Gunicorn):
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

### Deployment Platforms

- **Render**: Easy deployment with automatic HTTPS
- **DigitalOcean**: App Platform or Droplet
- **AWS**: Elastic Beanstalk or EC2
- **Heroku**: Simple git-based deployment

## Development Notes

- Models are loaded on server startup for better performance
- Predictions are stateless - no session management required
- Mock data is provided for service centers if Google Maps API is not configured
- SMS alerts include rate limiting (1 message per hour per user by default)

## License

MIT
