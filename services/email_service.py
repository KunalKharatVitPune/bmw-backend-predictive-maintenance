"""
Email Service - Send PDF reports via email using SendGrid
"""
import os
from typing import Dict

# Make sendgrid optional
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False
    print("⚠️ sendgrid not installed. Email functionality will be disabled.")
    print("   To enable: pip install sendgrid")


class EmailService:
    """Service for sending emails via SendGrid"""
    
    def __init__(self, api_key: str = None, from_email: str = None):
        """
        Initialize Email Service
        
        Args:
            api_key: SendGrid API key
            from_email: Verified sender email address
        """
        self.api_key = api_key or os.getenv('SENDGRID_API_KEY', '')
        self.from_email = from_email or os.getenv('SENDGRID_FROM_EMAIL', '')
        self.client = None
        
        if not SENDGRID_AVAILABLE:
            print("⚠️ SendGrid module not installed. Email disabled.")
            return
            
        if self.api_key and self.from_email:
            try:
                self.client = SendGridAPIClient(self.api_key)
                print(f"✅ SendGrid configured with sender: {self.from_email}")
            except Exception as e:
                print(f"⚠️ Failed to initialize SendGrid: {e}")
        else:
            if not self.api_key:
                print("⚠️ SENDGRID_API_KEY not set. Email disabled.")
            if not self.from_email:
                print("⚠️ SENDGRID_FROM_EMAIL not set. Email disabled.")
    
    def send_pdf_report(self, to_email: str, pdf_url: str, report_date: str = None) -> Dict:
        """
        Send vehicle health report PDF link via email
        
        Args:
            to_email: Recipient email address
            pdf_url: Cloudinary URL of the PDF report
            report_date: Optional date string for the report
            
        Returns:
            Dictionary with success status and message
        """
        if not self.client:
            return {
                'success': False,
                'message': 'Email service not configured. Please set SENDGRID_API_KEY and SENDGRID_FROM_EMAIL.'
            }
        
        if not to_email or '@' not in to_email:
            return {
                'success': False,
                'message': 'Invalid email address provided.'
            }
        
        try:
            # Build email content
            subject = f"🚗 Your Vehicle Health Report{f' - {report_date}' if report_date else ''}"
            
            html_content = f"""
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background: linear-gradient(135deg, #3b82f6, #8b5cf6); padding: 30px; border-radius: 12px 12px 0 0; text-align: center;">
                    <h1 style="color: white; margin: 0; font-size: 24px;">🚗 Vehicle Health Report</h1>
                    <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0;">AutoCare AI Analysis</p>
                </div>
                
                <div style="background: #f8fafc; padding: 30px; border-radius: 0 0 12px 12px; border: 1px solid #e2e8f0; border-top: none;">
                    <p style="color: #334155; font-size: 16px; line-height: 1.6;">
                        Hello,
                    </p>
                    <p style="color: #334155; font-size: 16px; line-height: 1.6;">
                        Your vehicle health analysis report is ready. Click the button below to view or download your detailed PDF report.
                    </p>
                    
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="{pdf_url}" 
                           style="display: inline-block; background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; text-decoration: none; padding: 14px 32px; border-radius: 8px; font-weight: 600; font-size: 16px;">
                            📄 View Report
                        </a>
                    </div>
                    
                    <p style="color: #64748b; font-size: 14px; line-height: 1.6;">
                        This report includes:
                    </p>
                    <ul style="color: #64748b; font-size: 14px; line-height: 1.8;">
                        <li>Overall vehicle health score</li>
                        <li>Component-level analysis</li>
                        <li>Degradation factors and trends</li>
                        <li>Maintenance recommendations</li>
                        <li>AI-powered insights</li>
                    </ul>
                    
                    <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 24px 0;">
                    
                    <p style="color: #94a3b8; font-size: 12px; text-align: center;">
                        This report was generated by AutoCare AI - Predictive Maintenance System
                    </p>
                </div>
            </div>
            """
            
            plain_text = f"""
Vehicle Health Report

Your vehicle health analysis report is ready.

View your report: {pdf_url}

This report includes:
- Overall vehicle health score
- Component-level analysis
- Degradation factors and trends
- Maintenance recommendations
- AI-powered insights

Generated by AutoCare AI - Predictive Maintenance System
            """
            
            # Create email message
            message = Mail(
                from_email=self.from_email,
                to_emails=to_email,
                subject=subject,
                html_content=html_content,
                plain_text_content=plain_text
            )
            
            # Send email
            response = self.client.send(message)
            
            if response.status_code in [200, 201, 202]:
                return {
                    'success': True,
                    'message': f'Report sent successfully to {to_email}'
                }
            else:
                return {
                    'success': False,
                    'message': f'Failed to send email. Status: {response.status_code}'
                }
                
        except Exception as e:
            print(f"Email send error: {e}")
            return {
                'success': False,
                'message': f'Failed to send email: {str(e)}'
            }
    
    def test_connection(self) -> Dict:
        """Test SendGrid connection"""
        if not self.client:
            return {
                'success': False,
                'message': 'SendGrid not configured'
            }
        
        return {
            'success': True,
            'message': f'SendGrid configured with sender: {self.from_email}'
        }
