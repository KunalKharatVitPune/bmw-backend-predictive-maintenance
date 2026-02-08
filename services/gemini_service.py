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
    
    # =========================================================================
    # CHATBOT METHODS
    # =========================================================================
    
    CHATBOT_NAME = "AutoCare AI"
    
    def initialize_chat_with_context(self, prediction_data: Dict, pdf_url: str = None) -> Dict:
        """
        Initialize a chat session with vehicle health context
        
        Args:
            prediction_data: The prediction results (KPIs, component health, etc.)
            pdf_url: Optional URL to the uploaded PDF report
            
        Returns:
            Dictionary with session info and greeting message
        """
        if not self.model:
            return {
                'success': False,
                'message': 'Gemini AI not configured',
                'greeting': None
            }
        
        try:
            # Build context from prediction data
            context = self._build_chat_context(prediction_data)
            
            # Create system prompt with context
            system_prompt = f"""You are {self.CHATBOT_NAME}, a friendly and knowledgeable vehicle health assistant.

You have access to the following vehicle health analysis data:

{context}

Your role:
- Answer questions about this specific vehicle's health status
- Explain KPIs, component scores, and degradation factors in simple terms
- Provide practical maintenance advice based on the data
- Be concise but helpful - aim for 2-3 sentences unless more detail is requested
- Use a friendly, professional tone
- If asked something outside your knowledge, politely explain you can only help with this vehicle's analysis

IMPORTANT: Always refer to the specific data provided. Don't make up values.
"""
            
            # Start chat session
            self.chat_session = self.model.start_chat(history=[])
            self.chat_context = system_prompt
            
            # Generate greeting
            greeting = f"Hello! I'm {self.CHATBOT_NAME}, your vehicle health assistant. How can I help you today?"
            
            return {
                'success': True,
                'message': 'Chat initialized successfully',
                'greeting': greeting,
                'chatbot_name': self.CHATBOT_NAME
            }
            
        except Exception as e:
            print(f"Chat initialization error: {e}")
            return {
                'success': False,
                'message': f'Failed to initialize chat: {str(e)}',
                'greeting': None
            }
    
    def chat(self, user_message: str) -> Dict:
        """
        Send a message and get a response
        
        Args:
            user_message: The user's question or message
            
        Returns:
            Dictionary with AI response
        """
        if not self.model:
            return {
                'success': False,
                'response': 'Chat is not available. Gemini AI not configured.'
            }
        
        if not hasattr(self, 'chat_session') or not self.chat_session:
            return {
                'success': False,
                'response': 'Chat session not initialized. Please start a new analysis first.'
            }
        
        try:
            # Prepend context to first message if not already done
            if not hasattr(self, 'context_sent') or not self.context_sent:
                full_message = f"{self.chat_context}\n\nUser question: {user_message}"
                self.context_sent = True
            else:
                full_message = user_message
            
            # Get response
            response = self.chat_session.send_message(full_message)
            
            return {
                'success': True,
                'response': response.text.strip()
            }
            
        except Exception as e:
            print(f"Chat error: {e}")
            return {
                'success': False,
                'response': f'Sorry, I encountered an error. Please try again.'
            }
    
    def reset_chat(self):
        """Reset the chat session"""
        self.chat_session = None
        self.chat_context = None
        self.context_sent = False
    
    def _build_chat_context(self, prediction_data: Dict) -> str:
        """Build context string from prediction data"""
        context = "## Vehicle Health Analysis Results\n\n"
        
        # KPIs
        kpis = prediction_data.get('kpis', {})
        if kpis:
            context += "### Key Performance Indicators:\n"
            context += f"- Failure Probability: {kpis.get('failure_probability', 'N/A')}%\n"
            context += f"- Remaining Useful Life: {kpis.get('remaining_useful_life', 'N/A')} cycles\n"
            context += f"- Anomaly Score: {kpis.get('anomaly_score', 'N/A')}\n"
            context += f"- Overall Health: {kpis.get('overall_health', 'N/A')}%\n\n"
        
        # Component Health
        component_health = prediction_data.get('component_health', {})
        if component_health:
            context += "### Component Health Scores:\n"
            for component, data in component_health.items():
                score = data.get('score', 'N/A') if isinstance(data, dict) else data
                context += f"- {component}: {score}%\n"
            context += "\n"
        
        # Degradation Contributors
        contributors = prediction_data.get('degradation_contributors', [])
        if contributors:
            context += "### Top Degradation Factors:\n"
            for i, contrib in enumerate(contributors[:5], 1):
                context += f"{i}. {contrib.get('feature', 'Unknown')}: Value={contrib.get('value', 'N/A')}, Impact={contrib.get('importance', 0):.3f}\n"
            context += "\n"
        
        # Maintenance Decision
        maintenance = prediction_data.get('maintenance_decision', {})
        if maintenance:
            context += "### Maintenance Recommendation:\n"
            context += f"- Level: {maintenance.get('level', 'N/A')}\n"
            context += f"- Message: {maintenance.get('message', 'N/A')}\n"
            context += f"- Description: {maintenance.get('description', 'N/A')}\n"
        
        return context
