"""
Cloudinary Service
Handles PDF upload to Cloudinary cloud storage
"""
import cloudinary
import cloudinary.uploader
import os
import base64
from datetime import datetime


class CloudinaryService:
    """Service for uploading files to Cloudinary"""
    
    def __init__(self, cloud_name=None, api_key=None, api_secret=None):
        """Initialize Cloudinary with credentials"""
        self.cloud_name = cloud_name or os.getenv('CLOUDINARY_CLOUD_NAME')
        self.api_key = api_key or os.getenv('CLOUDINARY_API_KEY')
        self.api_secret = api_secret or os.getenv('CLOUDINARY_API_SECRET')
        
        if self.cloud_name and self.api_key and self.api_secret:
            cloudinary.config(
                cloud_name=self.cloud_name,
                api_key=self.api_key,
                api_secret=self.api_secret,
                secure=True
            )
            print(f"✅ Cloudinary configured: {self.cloud_name}")
        else:
            print("⚠️ Cloudinary credentials not configured")
    
    def upload_pdf(self, pdf_data, filename=None):
        """
        Upload PDF to Cloudinary
        
        Args:
            pdf_data: Base64 encoded PDF data or raw bytes
            filename: Optional custom filename
            
        Returns:
            dict with success status and URL or error
        """
        try:
            if not self.cloud_name:
                return {
                    'success': False,
                    'error': 'Cloudinary not configured'
                }
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"vehicle_health_report_{timestamp}"
            
            # Handle base64 data
            if isinstance(pdf_data, str):
                # Remove data URL prefix if present
                if pdf_data.startswith('data:'):
                    pdf_data = pdf_data.split(',')[1]
                # Decode base64
                pdf_bytes = base64.b64decode(pdf_data)
            else:
                pdf_bytes = pdf_data
            
            # Upload to Cloudinary
            result = cloudinary.uploader.upload(
                pdf_bytes,
                resource_type="raw",  # For non-image files like PDFs
                public_id=f"vehicle_reports/{filename}",
                format="pdf",
                overwrite=True,
                folder="vehicle_health_monitoring"
            )
            
            return {
                'success': True,
                'url': result.get('secure_url'),
                'public_id': result.get('public_id'),
                'created_at': result.get('created_at'),
                'bytes': result.get('bytes'),
                'format': result.get('format')
            }
            
        except Exception as e:
            print(f"❌ Cloudinary upload error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def delete_pdf(self, public_id):
        """Delete a PDF from Cloudinary"""
        try:
            result = cloudinary.uploader.destroy(public_id, resource_type="raw")
            return {
                'success': result.get('result') == 'ok',
                'result': result
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
