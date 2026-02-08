"""
Alert Service - Handles SMS notifications via Twilio
"""
import time
from typing import Dict, Optional

# Make Twilio optional - app works without it
try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioRestException
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("⚠️ Twilio not installed. SMS alerts will be disabled.")
    print("   To enable: pip install twilio")

class AlertService:
    """Service for sending SMS alerts using Twilio"""
    
    def __init__(self, account_sid: str, auth_token: str, phone_number: str, rate_limit_seconds: int = 3600):
        """
        Initialize Alert Service
        
        Args:
            account_sid: Twilio account SID
            auth_token: Twilio authentication token
            phone_number: Twilio phone number (sender)
            rate_limit_seconds: Minimum seconds between alerts (default 1 hour)
        """
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.phone_number = phone_number
        self.rate_limit_seconds = rate_limit_seconds
        
        # In-memory store for rate limiting (use Redis in production)
        self.last_alert_time = {}
        
        # Initialize Twilio client only if credentials and module are available
        self.client = None
        if not TWILIO_AVAILABLE:
            print("⚠️ Twilio module not installed. SMS alerts disabled.")
        elif account_sid and auth_token:
            try:
                self.client = Client(account_sid, auth_token)
                print("✅ Twilio client initialized successfully")
            except Exception as e:
                print(f"⚠️ Failed to initialize Twilio client: {e}")
        else:
            print("⚠️ Twilio credentials not provided. SMS alerts disabled.")
    
    def send_alert(
        self, 
        to_phone: str, 
        failure_prob: float, 
        rul: float, 
        severity: str,
        nearest_center: Optional[Dict] = None
    ) -> Dict:
        """
        Send SMS alert to user
        
        Args:
            to_phone: Recipient phone number
            failure_prob: Failure probability percentage
            rul: Remaining useful life (cycles)
            severity: Alert severity ('critical', 'warning', 'normal')
            nearest_center: Optional dict with nearest service center info
            
        Returns:
            Dictionary with send status and message
        """
        # Check if Twilio is configured
        if not self.client:
            return {
                'success': False,
                'message': 'Twilio not configured. SMS alerts disabled.',
                'timestamp': time.time()
            }
        
        # Check rate limiting
        if not self._can_send_alert(to_phone):
            last_sent = self.last_alert_time.get(to_phone, 0)
            time_remaining = self.rate_limit_seconds - (time.time() - last_sent)
            return {
                'success': False,
                'message': f'Rate limit exceeded. Next alert allowed in {int(time_remaining/60)} minutes.',
                'timestamp': time.time()
            }
        
        # Compose message
        message_body = self._compose_message(failure_prob, rul, severity, nearest_center)
        
        try:
            # Send SMS via Twilio
            message = self.client.messages.create(
                body=message_body,
                from_=self.phone_number,
                to=to_phone
            )
            
            # Update rate limiting timestamp
            self.last_alert_time[to_phone] = time.time()
            
            return {
                'success': True,
                'message': 'Alert sent successfully',
                'sid': message.sid,
                'timestamp': time.time()
            }
            
        except Exception as e:
            # Check if it's a Twilio error (if Twilio is available)
            error_msg = f'Twilio error: {str(e)}' if TWILIO_AVAILABLE else f'Failed to send alert: {str(e)}'
            return {
                'success': False,
                'message': error_msg,
                'timestamp': time.time()
            }
    
    def _can_send_alert(self, phone_number: str) -> bool:
        """
        Check if alert can be sent based on rate limiting
        
        Args:
            phone_number: Recipient phone number
            
        Returns:
            Boolean indicating if alert can be sent
        """
        last_sent = self.last_alert_time.get(phone_number, 0)
        time_elapsed = time.time() - last_sent
        return time_elapsed >= self.rate_limit_seconds
    
    def _compose_message(
        self, 
        failure_prob: float, 
        rul: float, 
        severity: str,
        nearest_center: Optional[Dict] = None
    ) -> str:
        """
        Compose SMS message body
        
        Args:
            failure_prob: Failure probability percentage
            rul: Remaining useful life
            severity: Alert severity level
            nearest_center: Nearest service center info
            
        Returns:
            Formatted message string
        """
        # Alert emoji based on severity
        emoji = {
            'critical': '🚨',
            'warning': '⚠️',
            'normal': 'ℹ️'
        }.get(severity, 'ℹ️')
        
        # Compose base message
        message = f"{emoji} BMW Vehicle Health Alert\n\n"
        
        if severity == 'critical':
            message += "CRITICAL: Immediate maintenance required!\n\n"
        elif severity == 'warning':
            message += "WARNING: Schedule maintenance soon.\n\n"
        
        message += f"📊 Health Metrics:\n"
        message += f"• Failure Risk: {failure_prob:.1f}%\n"
        message += f"• Remaining Life: {rul:.0f} cycles\n\n"
        
        # Add nearest service center if available
        if nearest_center:
            message += f"📍 Nearest Service Center:\n"
            message += f"{nearest_center.get('name', 'N/A')}\n"
            message += f"Distance: {nearest_center.get('distance', 'N/A')} km\n\n"
        
        message += "Open your dashboard for detailed analysis."
        
        return message
    
    def test_connection(self) -> Dict:
        """
        Test Twilio connection
        
        Returns:
            Dictionary with connection status
        """
        if not self.client:
            return {
                'success': False,
                'message': 'Twilio client not initialized'
            }
        
        try:
            # Try to fetch account details
            account = self.client.api.accounts(self.account_sid).fetch()
            return {
                'success': True,
                'message': f'Connected to Twilio account: {account.friendly_name}'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Connection failed: {str(e)}'
            }
