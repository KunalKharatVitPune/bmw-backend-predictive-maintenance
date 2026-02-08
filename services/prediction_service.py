"""
Prediction Service - Handles ML model inference for vehicle health prediction
"""
import numpy as np
import tensorflow as tf
import os

# Try explicit imports to handle different TF versions
try:
    from tensorflow.keras.models import load_model
    # Workaround for TF 2.x/Keras 3.x compatibility
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
except ImportError:
    # Fallback or re-raise
    try:
        import keras
        from keras.models import load_model
    except ImportError:
        print("❌ Critical: Could not import Keras. Please check tensorflow installation.")
        load_model = None

from typing import Dict, List, Tuple

class PredictionService:
    """Service for handling vehicle health predictions using LSTM and Autoencoder models"""
    
    def __init__(self, lstm_model_path: str, autoencoder_model_path: str):
        """
        Initialize prediction service with pre-trained models
        
        Args:
            lstm_model_path: Path to LSTM attention model (.h5 file)
            autoencoder_model_path: Path to autoencoder model (.h5 file)
        """
        self.lstm_model = None
        self.autoencoder_model = None
        self.lstm_model_path = lstm_model_path
        self.autoencoder_model_path = autoencoder_model_path
        
        # Feature names for reference
        self.feature_names = [
            "State of Charge",
            "State of Health",
            "Battery Voltage",
            "Battery Current",
            "Battery Temperature",
            "Motor Temperature",
            "Motor Vibration",
            "Motor RPM",
            "Brake Pad Wear",
            "Power Stress",
            "Usage Intensity",
            "Health Trend"
        ]
        
    def load_models(self):
        """Load ML models into memory"""
        if load_model is None:
            print("❌ Cannot load models: Keras not imported.")
            return

        print(f"Loading LSTM model from {self.lstm_model_path}...")
        try:
            self.lstm_model = load_model(
                self.lstm_model_path, 
                compile=False
            )
        except Exception as e:
            print(f"❌ Failed to load LSTM model: {e}")
        
        print(f"Loading Autoencoder model from {self.autoencoder_model_path}...")
        try:
            self.autoencoder_model = load_model(
                self.autoencoder_model_path, 
                compile=False
            )
        except Exception as e:
            print(f"❌ Failed to load Autoencoder model: {e}")
        
        if self.lstm_model and self.autoencoder_model:
            print("✅ Models loaded successfully!")
        else:
            print("⚠️ Some models failed to load. Predictions may not work.")
        
    def predict(self, features: List[float]) -> Dict:
        """
        Perform vehicle health prediction
        
        Args:
            features: List of 12 normalized feature values
            
        Returns:
            Dictionary containing prediction results and health metrics
        """
        if self.lstm_model is None or self.autoencoder_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Validate input
        if len(features) != 12:
            raise ValueError(f"Expected 12 features, got {len(features)}")
        
        # Convert to numpy array
        features_array = np.array(features, dtype=np.float32)
        
        # Prepare LSTM input (sequence of 50 timesteps)
        sequence = np.repeat(features_array.reshape(1, 1, -1), 50, axis=1)
        
        # Get LSTM predictions (failure probability and RUL)
        try:
            lstm_output = self.lstm_model.predict(sequence, verbose=0)
            
            # Handle different model output formats
            if isinstance(lstm_output, list) and len(lstm_output) >= 2:
                # Model returns [failure_prob, rul] as separate outputs
                failure_prob = float(lstm_output[0][0][0])
                rul = float(lstm_output[1][0][0])
            else:
                # Model returns single output or unexpected format
                print(f"⚠️ Unexpected LSTM output format: {type(lstm_output)}")
                failure_prob = 0.3  # Default moderate risk
                rul = 100.0  # Default RUL
        except Exception as e:
            print(f"⚠️ LSTM prediction error: {e}")
            # Use safe defaults
            failure_prob = 0.3
            rul = 100.0
        
        # Get Autoencoder reconstruction for anomaly detection
        try:
            reconstruction = self.autoencoder_model.predict(
                features_array.reshape(1, -1), 
                verbose=0
            )
            anomaly_score = float(np.mean((features_array - reconstruction[0]) ** 2))
        except Exception as e:
            print(f"⚠️ Autoencoder prediction error: {e}")
            anomaly_score = 0.01  # Default low anomaly
        
        # Calculate overall vehicle health
        overall_health = self._calculate_overall_health(features_array)
        
        # Get component health breakdown
        component_health = self._calculate_component_health(features_array)
        
        # Get degradation contributors
        degradation_contributors = self._get_degradation_contributors(features_array)
        
        # Generate maintenance decision
        maintenance_decision = self._get_maintenance_decision(failure_prob, rul)
        
        # Check if alert should be triggered
        should_alert = self._should_trigger_alert(failure_prob, rul, anomaly_score)
        
        return {
            'kpis': {
                'failure_probability': round(failure_prob * 100, 2),  # Convert to percentage
                'remaining_useful_life': round(rul, 1),
                'anomaly_score': round(anomaly_score, 4),
                'overall_health': round(overall_health, 1)
            },
            'component_health': component_health,
            'degradation_contributors': degradation_contributors,
            'maintenance_decision': maintenance_decision,
            'should_alert': should_alert,
            'alert_severity': self._get_alert_severity(failure_prob, rul)
        }
    
    def _calculate_overall_health(self, features: np.ndarray) -> float:
        """
        Calculate overall vehicle health score
        
        Weighted combination of SoC, SoH, and Health Trend
        """
        soc = features[0]
        soh = features[1]
        health_trend = features[11]
        
        overall_health = (
            soc * 0.25 +
            soh * 0.35 +
            health_trend * 0.40
        ) * 100
        
        return float(overall_health)
    
    def _calculate_component_health(self, features: np.ndarray) -> Dict:
        """
        Calculate health scores for individual components
        
        Returns:
            Dictionary with component names and their health status
        """
        components = {
            'Battery System': {
                'score': round(float(np.mean([features[0], features[1]]) * 100), 1),
                'status': ''
            },
            'Thermal System': {
                'score': round(float(np.mean([1 - features[4]/70, 1 - features[5]/110]) * 100), 1),
                'status': ''
            },
            'Motor System': {
                'score': round(float(np.mean([1 - features[6]/3, features[7]/6000]) * 100), 1),
                'status': ''
            },
            'Braking System': {
                'score': round(float(1 - features[8]) * 100, 1),
                'status': ''
            },
            'Usage Stress': {
                'score': round(float(1 - features[10]/100) * 100, 1),
                'status': ''
            }
        }
        
        # Assign status based on score
        for component in components.values():
            score = component['score']
            if score >= 80:
                component['status'] = 'healthy'
                component['icon'] = '🟢'
            elif score >= 50:
                component['status'] = 'degrading'
                component['icon'] = '🟡'
            else:
                component['status'] = 'critical'
                component['icon'] = '🔴'
        
        return components
    
    def _get_degradation_contributors(self, features: np.ndarray) -> List[Dict]:
        """
        Identify top degradation contributors based on feature importance
        
        Returns:
            List of top 3 contributors with names and severity
        """
        # Calculate importance based on absolute values
        importance = np.abs(features)
        top_indices = np.argsort(importance)[-3:][::-1]
        
        contributors = []
        for idx in top_indices:
            contributors.append({
                'feature': self.feature_names[idx],
                'value': round(float(features[idx]), 3),
                'importance': round(float(importance[idx]), 3)
            })
        
        return contributors
    
    def _get_maintenance_decision(self, failure_prob: float, rul: float) -> Dict:
        """
        Generate maintenance recommendation based on predictions
        
        Returns:
            Dictionary with decision message and severity level
        """
        if failure_prob >= 0.7 or rul <= 30:
            return {
                'level': 'immediate',
                'message': '🚨 Immediate Maintenance Required',
                'description': 'Critical condition detected. Schedule service immediately.',
                'color': 'red'
            }
        elif failure_prob >= 0.4 or rul <= 60:
            return {
                'level': 'soon',
                'message': '⚠️ Schedule Maintenance Soon',
                'description': 'Degradation detected. Plan maintenance within the next week.',
                'color': 'yellow'
            }
        else:
            return {
                'level': 'normal',
                'message': '✅ Vehicle Operating Normally',
                'description': 'All systems functioning within normal parameters.',
                'color': 'green'
            }
    
    def _should_trigger_alert(self, failure_prob: float, rul: float, anomaly_score: float) -> bool:
        """
        Determine if SMS alert should be triggered
        
        Returns:
            Boolean indicating whether to send alert
        """
        return failure_prob >= 0.7 or rul <= 30 or anomaly_score > 0.5
    
    def _fallback_prediction(self, features: List[float]) -> Dict:
        """
        Fallback method when models fail. Uses rule-based logic.
        """
        features_array = np.array(features, dtype=np.float32)
        
        # Simple heuristic calculations
        soc = features[0]
        soh = features[1]
        temp = features[4]
        
        # Heuristic failure probability
        failure_prob = 0.1
        if soh < 80: failure_prob += 0.3
        if temp > 40: failure_prob += 0.2
        if soc < 20: failure_prob += 0.1
        
        # Heuristic RUL
        rul = soh * 2.5  # Rough estimate
        
        # Heuristic anomaly score
        anomaly_score = 0.01 + (100 - soh)/1000
        
        # Calculate derived metrics
        overall_health = self._calculate_overall_health(features_array)
        component_health = self._calculate_component_health(features_array)
        degradation_contributors = self._get_degradation_contributors(features_array)
        maintenance_decision = self._get_maintenance_decision(failure_prob, rul)
        should_alert = self._should_trigger_alert(failure_prob, rul, anomaly_score)
        
        return {
            'kpis': {
                'failure_probability': round(float(failure_prob) * 100, 2),
                'remaining_useful_life': round(float(rul), 1),
                'anomaly_score': round(float(anomaly_score), 4),
                'overall_health': round(float(overall_health), 1)
            },
            'component_health': component_health,
            'degradation_contributors': degradation_contributors,
            'maintenance_decision': maintenance_decision,
            'should_alert': should_alert,
            'alert_severity': self._get_alert_severity(failure_prob, rul)
        }

    def _get_alert_severity(self, failure_prob: float, rul: float) -> str:
        """
        Get alert severity level
        
        Returns:
            'critical', 'warning', or 'normal'
        """
        if failure_prob >= 0.7 or rul <= 30:
            return 'critical'
        elif failure_prob >= 0.4 or rul <= 60:
            return 'warning'
        else:
            return 'normal'
