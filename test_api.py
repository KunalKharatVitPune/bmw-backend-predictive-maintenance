"""
Test script for Vehicle Health Monitoring API
Run this after starting the Flask server to verify all endpoints work correctly
"""
import requests
import json

# Base URL
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing: Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✅ Health check passed!")


def test_prediction():
    """Test prediction endpoint"""
    print("\n" + "="*60)
    print("Testing: Vehicle Health Prediction")
    print("="*60)
    
    # Sample features (normalized values)
    test_data = {
        "features": [
            0.8,    # State of Charge
            0.9,    # State of Health
            360,    # Battery Voltage (V)
            -25,    # Battery Current (A)
            32,     # Battery Temperature (°C)
            65,     # Motor Temperature (°C)
            0.6,    # Motor Vibration (mm/s)
            2200,   # Motor RPM
            0.25,   # Brake Pad Wear
            8,      # Power Stress (kW)
            45,     # Usage Intensity
            0.85    # Health Trend
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/api/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "data" in data
    assert "kpis" in data["data"]
    print("✅ Prediction test passed!")


def test_service_centers():
    """Test service centers endpoint"""
    print("\n" + "="*60)
    print("Testing: Nearby Service Centers")
    print("="*60)
    
    # Test coordinates (New York City)
    params = {
        "lat": 40.7128,
        "lng": -74.0060,
        "radius": 10000
    }
    
    response = requests.get(f"{BASE_URL}/api/service-centers", params=params)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "service_centers" in data
    print("✅ Service centers test passed!")


def test_invalid_prediction():
    """Test prediction with invalid data"""
    print("\n" + "="*60)
    print("Testing: Invalid Prediction (Error Handling)")
    print("="*60)
    
    # Missing features
    test_data = {
        "features": [0.8, 0.9]  # Only 2 features instead of 12
    }
    
    response = requests.post(
        f"{BASE_URL}/api/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 400
    print("✅ Error handling test passed!")


def test_critical_condition():
    """Test prediction with critical vehicle condition"""
    print("\n" + "="*60)
    print("Testing: Critical Condition (Should Trigger Alert)")
    print("="*60)
    
    # Critical condition: low SoC, low SoH, high temperatures
    test_data = {
        "features": [
            0.15,   # Very low State of Charge
            0.45,   # Degraded State of Health
            320,    # Lower Battery Voltage
            -60,    # High negative current
            55,     # High Battery Temperature
            95,     # High Motor Temperature
            2.5,    # High Motor Vibration
            4500,   # High Motor RPM
            0.75,   # High Brake Wear
            35,     # High Power Stress
            85,     # High Usage Intensity
            0.40    # Low Health Trend
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/api/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    assert response.status_code == 200
    
    # Check if alert should be triggered
    if data["data"]["should_alert"]:
        print("⚠️ ALERT: Critical condition detected!")
        print(f"Failure Probability: {data['data']['kpis']['failure_probability']}%")
        print(f"Remaining Useful Life: {data['data']['kpis']['remaining_useful_life']} cycles")
        print(f"Maintenance: {data['data']['maintenance_decision']['message']}")
    
    print("✅ Critical condition test passed!")


if __name__ == "__main__":
    print("\n" + "🚗 BMW Vehicle Health Monitoring API - Test Suite")
    print("="*60)
    print("Make sure the Flask server is running on http://localhost:5000")
    print("="*60)
    
    try:
        # Run all tests
        test_health_check()
        test_prediction()
        test_service_centers()
        test_invalid_prediction()
        test_critical_condition()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the server.")
        print("Make sure the Flask server is running:")
        print("  cd backend")
        print("  python app.py")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
