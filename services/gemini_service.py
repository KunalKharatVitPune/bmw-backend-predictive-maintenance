"""
Gemini AI Service - Provides intelligent interpretations of vehicle telemetry data
"""
import os
from typing import Dict, List, Optional

# Make google-generativeai optional
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ google-generativeai not installed. AI insights will be disabled.")
    print("   To enable: pip install google-generativeai")


class GeminiService:
    """Service for generating AI insights using Google Gemini"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Gemini Service
        
        Args:
            api_key: Google Gemini API key
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY', '')
        self.model = None
        
        if not GEMINI_AVAILABLE:
            print("⚠️ Gemini module not installed. AI insights disabled.")
            return
            
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                print("✅ Gemini AI initialized successfully")
            except Exception as e:
                print(f"⚠️ Failed to initialize Gemini: {e}")
        else:
            print("⚠️ Gemini API key not provided. AI insights disabled.")
    
    def analyze_degradation(
        self, 
        contributors: List[Dict],
        kpis: Dict = None,
        component_health: Dict = None
    ) -> Dict:
        """
        Generate AI interpretation of degradation factors
        
        Args:
            contributors: List of degradation contributors with feature, value, importance
            kpis: KPI values (failure_probability, remaining_useful_life, etc.)
            component_health: Component health scores
            
        Returns:
            Dictionary with AI-generated insights
        """
        if not self.model:
            return {
                'success': False,
                'insights': None,
                'message': 'Gemini AI not configured'
            }
        
        try:
            # Build the prompt
            prompt = self._build_analysis_prompt(contributors, kpis, component_health)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Parse the response
            insights = response.text.strip()
            
            return {
                'success': True,
                'insights': insights,
                'message': 'Analysis generated successfully'
            }
            
        except Exception as e:
            print(f"Gemini analysis error: {e}")
            return {
                'success': False,
                'insights': None,
                'message': f'Failed to generate analysis: {str(e)}'
            }
    
    def _build_analysis_prompt(
        self, 
        contributors: List[Dict],
        kpis: Dict = None,
        component_health: Dict = None
    ) -> str:
        """Build the prompt for Gemini analysis"""
        
        prompt = """You are an expert vehicle diagnostics engineer. Analyze the following vehicle telemetry data and provide a concise, actionable interpretation.

## Degradation Factors (ranked by impact):
"""
        for i, c in enumerate(contributors[:5], 1):
            prompt += f"{i}. **{c.get('feature', 'Unknown')}**: Value = {c.get('value', 'N/A')}, Impact Score = {c.get('importance', 0):.3f}\n"
        
        if kpis:
            prompt += f"""
## Key Performance Indicators:
- Failure Probability: {kpis.get('failure_probability', 'N/A')}%
- Remaining Useful Life: {kpis.get('remaining_useful_life', 'N/A')} cycles
- Anomaly Score: {kpis.get('anomaly_score', 'N/A')}
- Health Score: {kpis.get('health_score', 'N/A')}%
"""
        
        if component_health:
            prompt += "\n## Component Health Scores:\n"
            for component, score in component_health.items():
                prompt += f"- {component}: {score}%\n"
        
        prompt += """
## Instructions:
Return your analysis as a JSON array. Each item should have:
- "title": A short 2-4 word title (e.g., "High Motor Stress", "Immediate Action Needed")
- "description": 1-2 sentence explanation
- "type": One of "warning", "critical", "info", or "tip"

Example format:
[
  {"title": "Primary Concern", "description": "Your motor RPM is running high which causes excessive wear.", "type": "warning"},
  {"title": "Recommended Action", "description": "Schedule a motor inspection within the next 2 weeks.", "type": "info"}
]

Return 3-4 JSON items. Keep descriptions concise and non-technical.
IMPORTANT: Return ONLY the JSON array, no other text or markdown formatting.
"""
        
        return prompt
    
    def get_quick_insight(self, feature: str, value: float, importance: float) -> str:
        """
        Get a quick one-line insight for a single degradation factor
        
        Args:
            feature: Feature name
            value: Current value
            importance: Impact score
            
        Returns:
            One-line insight string
        """
        if not self.model:
            return self._get_fallback_insight(feature, value, importance)
        
        try:
            prompt = f"""In ONE sentence (max 15 words), explain why '{feature}' with value {value} and impact {importance:.1f} is concerning for a vehicle. Be specific and practical."""
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Quick insight error: {e}")
            return self._get_fallback_insight(feature, value, importance)
    
    def _get_fallback_insight(self, feature: str, value: float, importance: float) -> str:
        """Fallback insights when Gemini is not available"""
        insights = {
            'motor_rpm': f"High motor RPM ({value}) indicates excessive strain on the motor.",
            'battery_voltage': f"Battery voltage at {value}V needs monitoring for optimal performance.",
            'motor_temperature': f"Motor temperature of {value}°C suggests thermal stress buildup.",
            'tire_pressure': f"Tire pressure of {value} PSI is outside optimal range.",
            'engine_load': f"Engine load at {value}% - consider reducing operational intensity.",
            'coolant_temp': f"Coolant temperature of {value}°C requires attention.",
            'throttle_position': f"Throttle position at {value}% indicates driving pattern stress.",
            'fuel_level': f"Fuel level at {value}% - monitor consumption patterns.",
        }
        
        # Normalize feature name for lookup
        feature_lower = feature.lower().replace(' ', '_')
        return insights.get(feature_lower, f"{feature} at {value} is contributing to system stress.")
    
    def test_connection(self) -> Dict:
        """Test Gemini API connection"""
        if not self.model:
            return {
                'success': False,
                'message': 'Gemini AI not initialized'
            }
        
        try:
            response = self.model.generate_content("Say 'Gemini is connected!' in exactly 3 words.")
            return {
                'success': True,
                'message': response.text.strip()
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Connection failed: {str(e)}'
            }
