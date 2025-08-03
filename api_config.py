"""
API Configuration for Prometheus Data Access

IMPORTANT: This file contains sensitive credentials. 
- Never commit this file to version control
- Add 'api_config.py' to .gitignore
- Use environment variables in production
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class APICredentials:
    """Secure storage for API credentials"""
    
    # NASA EarthData
    EARTHDATA_TOKEN: str = os.environ.get('EARTHDATA_TOKEN', 
        'eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6Im1yYWhtYW5pIiwiZXhwIjoxNzU5MzI5MjYwLCJpYXQiOjE3NTQxNDUyNjAsImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiIsImlkZW50aXR5X3Byb3ZpZGVyIjoiZWRsX29wcyIsImFjciI6ImVkbCIsImFzc3VyYW5jZV9sZXZlbCI6M30.5K-qMRMmkZbX8ChNhffqf5tkB5EJ-bjZ1JR0xTfMbEa3vr5GyZt-PML62XhduKEOZMt80x6DZePwAJFrMWIvL55EMJpwyBZZlDfT47dG6yBWGqPwrKjKNRWqAK_-6Ktxfn04IKNcRXYfnZI_b6HuUNEX0TbkVsvphRp9oestadGQGmWV80xi-Zh162qMCAFBAfNex85jSUwRqn7UMFsCGFi_PZ-IKsYCTrOR5yUKm-kLj-_Ax7ZpnEn4jz_hxOOH6np2ZlOGj-n7zDOa5G1T6ndWJyOLGrCswhkZUGZ6CnOrRlXCst_NNhaBUaUbCR158fQQXDJzlQ9CyhiIeYMHZw')
    
    # USGS
    USGS_API_KEY: str = os.environ.get('USGS_API_KEY', 
        '2U!e85rDbmalpwb7NcCvhypmNO3Nt8L2VhguU3Ry6MoLmMBo6v@btVeQWtbez3ib')
    
    # IRIS
    IRIS_EMAIL: str = os.environ.get('IRIS_EMAIL', 'time12@gmail.com')
    IRIS_PASSWORD: str = os.environ.get('IRIS_PASSWORD', 'ZNlyVEIuWrmlWmhd')
    
    # Copernicus/Sentinel Hub
    COPERNICUS_CLIENT_ID: str = os.environ.get('COPERNICUS_CLIENT_ID', 
        'sh-16df973e-9c24-4a16-ba4e-6c6307c34cdf')
    COPERNICUS_CLIENT_SECRET: str = os.environ.get('COPERNICUS_CLIENT_SECRET', 
        'Ll6DYmiiGTiHjvde6fTNIni6feLglNet')

class APIEndpoints:
    """API endpoints for various data sources"""
    
    # NASA EarthData
    EARTHDATA_BASE = "https://urs.earthdata.nasa.gov"
    EARTHDATA_CMR = "https://cmr.earthdata.nasa.gov"
    
    # ASF HyP3 (Alaska Satellite Facility)
    ASF_HYP3_BASE = "https://hyp3-api.asf.alaska.edu"
    ASF_SEARCH = "https://api.daac.asf.alaska.edu/services/search/param"
    
    # USGS Earthquake
    USGS_EARTHQUAKE_BASE = "https://earthquake.usgs.gov/fdsnws/event/1"
    USGS_EARTHQUAKE_QUERY = f"{USGS_EARTHQUAKE_BASE}/query"
    USGS_EARTHQUAKE_COUNT = f"{USGS_EARTHQUAKE_BASE}/count"
    
    # IRIS Seismic
    IRIS_BASE = "http://service.iris.edu/fdsnws"
    IRIS_STATION = f"{IRIS_BASE}/station/1"
    IRIS_DATASELECT = f"{IRIS_BASE}/dataselect/1"
    IRIS_EVENT = f"{IRIS_BASE}/event/1"
    
    # Copernicus/Sentinel Hub
    SENTINEL_HUB_AUTH = "https://services.sentinel-hub.com/oauth/token"
    SENTINEL_HUB_PROCESS = "https://services.sentinel-hub.com/api/v1/process"
    SENTINEL_HUB_CATALOG = "https://services.sentinel-hub.com/api/v1/catalog/search"
    
    # Smithsonian GVP
    GVP_WEEKLY = "https://volcano.si.edu/news/WeeklyVolcanoRSS.xml"
    GVP_DATABASE = "https://volcano.si.edu/database/search_eruption_results.cfm"

def get_auth_headers(service: str) -> Dict[str, str]:
    """Get authentication headers for specific service"""
    creds = APICredentials()
    
    if service == "earthdata":
        return {
            "Authorization": f"Bearer {creds.EARTHDATA_TOKEN}",
            "Accept": "application/json"
        }
    elif service == "usgs":
        return {
            "X-API-Key": creds.USGS_API_KEY,
            "Accept": "application/json"
        }
    elif service == "iris":
        # IRIS typically uses URL parameters for auth
        return {
            "Accept": "application/json"
        }
    elif service == "sentinel":
        # Sentinel Hub requires OAuth2 flow
        return {
            "Content-Type": "application/x-www-form-urlencoded"
        }
    else:
        return {}

def get_sentinel_hub_token() -> str:
    """Get Sentinel Hub OAuth2 token"""
    import requests
    
    creds = APICredentials()
    
    response = requests.post(
        APIEndpoints.SENTINEL_HUB_AUTH,
        data={
            "grant_type": "client_credentials",
            "client_id": creds.COPERNICUS_CLIENT_ID,
            "client_secret": creds.COPERNICUS_CLIENT_SECRET
        }
    )
    
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"Failed to get Sentinel Hub token: {response.text}")

# Example usage functions
def fetch_earthquake_data(start_time: str, end_time: str, min_magnitude: float = 4.0) -> Dict[str, Any]:
    """Fetch earthquake data from USGS"""
    import requests
    
    params = {
        "format": "geojson",
        "starttime": start_time,
        "endtime": end_time,
        "minmagnitude": min_magnitude,
        "orderby": "time"
    }
    
    headers = get_auth_headers("usgs")
    response = requests.get(APIEndpoints.USGS_EARTHQUAKE_QUERY, params=params, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch earthquake data: {response.text}")

def fetch_sentinel_thermal_data(bbox: list, date_from: str, date_to: str) -> Dict[str, Any]:
    """Fetch thermal data from Sentinel Hub"""
    import requests
    
    token = get_sentinel_hub_token()
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Search for Sentinel-3 SLSTR thermal data
    search_request = {
        "bbox": bbox,
        "datetime": f"{date_from}/{date_to}",
        "collections": ["sentinel-3-slstr"],
        "limit": 100
    }
    
    response = requests.post(
        APIEndpoints.SENTINEL_HUB_CATALOG,
        json=search_request,
        headers=headers
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch Sentinel data: {response.text}")

# Security warning function
def print_security_warning():
    """Print security warning about credentials"""
    print("="*60)
    print("SECURITY WARNING:")
    print("This file contains sensitive API credentials.")
    print("1. Add 'api_config.py' to .gitignore immediately")
    print("2. Use environment variables in production")
    print("3. Rotate these credentials regularly")
    print("4. Never commit this file to version control")
    print("="*60)

if __name__ == "__main__":
    print_security_warning()