"""
Location Service - Handles Google Maps API integration for service center suggestions
"""
from typing import List, Dict, Optional
from math import radians, cos, sin, asin, sqrt

# Make Google Maps optional - app works without it
try:
    import googlemaps
    GOOGLEMAPS_AVAILABLE = True
except ImportError:
    GOOGLEMAPS_AVAILABLE = False
    print("⚠️ Google Maps not installed. Using mock service centers.")
    print("   To enable: pip install googlemaps")

class LocationService:
    """Service for finding nearby service centers using Google Maps API"""
    
    def __init__(self, api_key: str):
        """
        Initialize Location Service
        
        Args:
            api_key: Google Maps API key
        """
        self.api_key = api_key
        self.client = None
        
        if not GOOGLEMAPS_AVAILABLE:
            print("⚠️ Google Maps module not installed. Using mock data.")
        elif api_key:
            try:
                self.client = googlemaps.Client(key=api_key)
                print("✅ Google Maps client initialized successfully")
            except Exception as e:
                print(f"⚠️ Failed to initialize Google Maps client: {e}")
        else:
            print("⚠️ Google Maps API key not provided. Using mock data.")
    
    def get_nearby_service_centers(
        self, 
        latitude: float, 
        longitude: float, 
        radius: int = 10000,
        max_results: int = 10
    ) -> List[Dict]:
        """
        Find nearby BMW service centers
        
        Args:
            latitude: User's latitude
            longitude: User's longitude
            radius: Search radius in meters (default 10km)
            max_results: Maximum number of results to return
            
        Returns:
            List of service centers with details
        """
        if not self.client:
            # Return mock data if Google Maps not configured
            return self._get_mock_service_centers(latitude, longitude)
        
        try:
            # Search for BMW service centers
            location = (latitude, longitude)
            
            places_result = self.client.places_nearby(
                location=location,
                radius=radius,
                keyword='BMW service center',
                type='car_repair'
            )
            
            service_centers = []
            
            for place in places_result.get('results', [])[:max_results]:
                center = self._parse_place_result(place, latitude, longitude)
                service_centers.append(center)
            
            # Sort by distance
            service_centers.sort(key=lambda x: x['distance_km'])
            
            return service_centers
            
        except Exception as e:
            print(f"Error fetching service centers: {e}")
            return self._get_mock_service_centers(latitude, longitude)
    
    def _parse_place_result(self, place: Dict, user_lat: float, user_lng: float) -> Dict:
        """
        Parse Google Places API result into standardized format
        
        Args:
            place: Place result from Google Maps API
            user_lat: User's latitude for distance calculation
            user_lng: User's longitude for distance calculation
            
        Returns:
            Formatted service center dictionary
        """
        location = place['geometry']['location']
        place_lat = location['lat']
        place_lng = location['lng']
        
        # Calculate distance
        distance = self._calculate_distance(user_lat, user_lng, place_lat, place_lng)
        
        return {
            'id': place.get('place_id', ''),
            'name': place.get('name', 'Unknown Service Center'),
            'address': place.get('vicinity', 'Address not available'),
            'latitude': place_lat,
            'longitude': place_lng,
            'distance_km': round(distance, 2),
            'rating': place.get('rating', 0),
            'total_ratings': place.get('user_ratings_total', 0),
            'is_open': place.get('opening_hours', {}).get('open_now', None),
            'phone': None,  # Requires Place Details API call
            'website': None  # Requires Place Details API call
        }
    
    def get_place_details(self, place_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific place
        
        Args:
            place_id: Google Place ID
            
        Returns:
            Detailed place information or None
        """
        if not self.client:
            return None
        
        try:
            result = self.client.place(
                place_id=place_id,
                fields=['name', 'formatted_address', 'formatted_phone_number', 
                       'website', 'opening_hours', 'rating', 'review']
            )
            
            return result.get('result', {})
            
        except Exception as e:
            print(f"Error fetching place details: {e}")
            return None
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two coordinates using Haversine formula
        
        Args:
            lat1, lon1: First coordinate
            lat2, lon2: Second coordinate
            
        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        return c * r
    
    def _get_mock_service_centers(self, latitude: float, longitude: float) -> List[Dict]:
        """
        Return mock service center data for testing when API is not configured
        
        Returns:
            List of mock service centers
        """
        # Generate mock data near the provided location
        return [
            {
                'id': 'mock_1',
                'name': 'BMW Service Center - Downtown',
                'address': '123 Main Street, City Center',
                'latitude': latitude + 0.01,
                'longitude': longitude + 0.01,
                'distance_km': 1.2,
                'rating': 4.5,
                'total_ratings': 234,
                'is_open': True,
                'phone': '+1-555-0100',
                'website': 'https://example.com/bmw-downtown'
            },
            {
                'id': 'mock_2',
                'name': 'BMW Premium Service Hub',
                'address': '456 Oak Avenue, Business District',
                'latitude': latitude + 0.02,
                'longitude': longitude - 0.01,
                'distance_km': 2.5,
                'rating': 4.7,
                'total_ratings': 412,
                'is_open': True,
                'phone': '+1-555-0200',
                'website': 'https://example.com/bmw-premium'
            },
            {
                'id': 'mock_3',
                'name': 'BMW Authorized Service',
                'address': '789 Industrial Blvd, Tech Park',
                'latitude': latitude - 0.03,
                'longitude': longitude + 0.02,
                'distance_km': 4.8,
                'rating': 4.3,
                'total_ratings': 189,
                'is_open': False,
                'phone': '+1-555-0300',
                'website': 'https://example.com/bmw-authorized'
            },
            {
                'id': 'mock_4',
                'name': 'BMW Express Service Station',
                'address': '321 Highway Road, Suburb',
                'latitude': latitude + 0.05,
                'longitude': longitude + 0.03,
                'distance_km': 6.3,
                'rating': 4.6,
                'total_ratings': 321,
                'is_open': True,
                'phone': '+1-555-0400',
                'website': 'https://example.com/bmw-express'
            }
        ]
