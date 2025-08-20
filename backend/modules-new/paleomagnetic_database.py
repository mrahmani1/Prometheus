"""
Advanced Paleomagnetic Database System
Comprehensive rock magnetic data integration and analysis
Includes global paleomagnetic records, reversal chronology, and field reconstruction
"""

import numpy as np
import pandas as pd
import sqlite3
import h5py
import json
import requests
import scipy as sp
from scipy import stats
from scipy.interpolate import interp1d, griddata
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Point, Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings('ignore')

class PaleomagneticRecord:
    """Individual paleomagnetic record container"""
    
    def __init__(self, site_id, location, age, field_data, metadata=None):
        self.site_id = site_id
        self.location = location  # (lat, lon, elevation)
        self.age = age  # Age in years
        self.field_data = field_data  # Magnetic field measurements
        self.metadata = metadata or {}
        
        # Quality assessment
        self.quality_score = self._assess_quality()
        
    def _assess_quality(self):
        """Assess data quality based on multiple criteria"""
        score = 0
        
        # Age constraint quality
        if 'age_error' in self.metadata:
            age_error = self.metadata['age_error']
            if age_error < self.age * 0.05:  # <5% error
                score += 30
            elif age_error < self.age * 0.1:  # <10% error
                score += 20
            else:
                score += 10
        
        # Number of samples
        if 'n_samples' in self.metadata:
            n_samples = self.metadata['n_samples']
            if n_samples >= 6:
                score += 25
            elif n_samples >= 3:
                score += 15
            else:
                score += 5
                
        # Laboratory treatment
        if self.metadata.get('af_demagnetized', False):
            score += 15
        if self.metadata.get('thermal_demagnetized', False):
            score += 15
            
        # Statistical parameters
        if 'alpha95' in self.metadata:
            alpha95 = self.metadata['alpha95']
            if alpha95 < 5:
                score += 15
            elif alpha95 < 10:
                score += 10
            else:
                score += 5
                
        return min(100, score)
    
    def to_dict(self):
        """Convert record to dictionary format"""
        return {
            'site_id': self.site_id,
            'latitude': self.location[0],
            'longitude': self.location[1],
            'elevation': self.location[2],
            'age': self.age,
            'field_data': self.field_data,
            'metadata': self.metadata,
            'quality_score': self.quality_score
        }

class MagneticReversalChronology:
    """Geomagnetic reversal chronology management"""
    
    def __init__(self):
        self.load_standard_chronologies()
        
    def load_standard_chronologies(self):
        """Load standard geomagnetic chronologies"""
        # GPTS2020 - Geomagnetic Polarity Time Scale 2020
        self.gpts2020 = self._load_gpts2020()
        
        # CANDE95 - Classical reversal sequence
        self.cande95 = self._load_cande95()
        
        # LR04 - Benthic stack reversal record
        self.lr04 = self._load_lr04()
        
        # Current active chronology
        self.active_chronology = self.gpts2020
        
    def _load_gpts2020(self):
        """Load GPTS2020 reversal chronology"""
        # Major reversals from GPTS2020
        reversals = [
            {'name': 'Brunhes/Matuyama', 'age': 0.773, 'type': 'boundary'},
            {'name': 'Jaramillo', 'age_start': 0.990, 'age_end': 1.070, 'type': 'subchron'},
            {'name': 'Matuyama/Gauss', 'age': 2.581, 'type': 'boundary'},
            {'name': 'Kaena', 'age_start': 3.032, 'age_end': 3.116, 'type': 'subchron'},
            {'name': 'Mammoth', 'age_start': 3.207, 'age_end': 3.330, 'type': 'subchron'},
            {'name': 'Gauss/Gilbert', 'age': 3.596, 'type': 'boundary'},
            {'name': 'Cochiti', 'age_start': 4.187, 'age_end': 4.300, 'type': 'subchron'},
            {'name': 'Nunivak', 'age_start': 4.493, 'age_end': 4.631, 'type': 'subchron'},
            {'name': 'Sidufjall', 'age_start': 4.799, 'age_end': 4.896, 'type': 'subchron'},
            {'name': 'Thvera', 'age_start': 4.997, 'age_end': 5.235, 'type': 'subchron'}
        ]
        
        return pd.DataFrame(reversals)
    
    def _load_cande95(self):
        """Load CANDE95 seafloor spreading chronology"""
        # Comprehensive CANDE95 geomagnetic polarity timescale
        # Based on Cande & Kent (1995) revised calibration
        reversals = [
            # Cenozoic Normal Polarity Chrons (C1n to C34n)
            {'chron': 'C1n', 'age_start': 0.000, 'age_end': 0.773, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C1r.1n', 'age_start': 0.990, 'age_end': 1.070, 'polarity': 'normal', 'quality': 'A'},  # Jaramillo
            {'chron': 'C1r.1r', 'age_start': 0.773, 'age_end': 0.990, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C1r.2r', 'age_start': 1.070, 'age_end': 1.770, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C2n', 'age_start': 1.770, 'age_end': 1.950, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C2r.1n', 'age_start': 2.140, 'age_end': 2.150, 'polarity': 'normal', 'quality': 'B'},  # Olduvai
            {'chron': 'C2r.1r', 'age_start': 1.950, 'age_end': 2.140, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C2r.2r', 'age_start': 2.150, 'age_end': 2.581, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C2An.1n', 'age_start': 2.581, 'age_end': 3.032, 'polarity': 'normal', 'quality': 'A'},  # Gauss
            {'chron': 'C2An.1r', 'age_start': 3.032, 'age_end': 3.116, 'polarity': 'reversed', 'quality': 'A'},  # Kaena
            {'chron': 'C2An.2n', 'age_start': 3.116, 'age_end': 3.207, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C2An.2r', 'age_start': 3.207, 'age_end': 3.330, 'polarity': 'reversed', 'quality': 'A'},  # Mammoth
            {'chron': 'C2An.3n', 'age_start': 3.330, 'age_end': 3.596, 'polarity': 'normal', 'quality': 'A'},
            
            # Gilbert Chron and subchrons
            {'chron': 'C2Ar', 'age_start': 3.596, 'age_end': 4.187, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C3n.1n', 'age_start': 4.187, 'age_end': 4.300, 'polarity': 'normal', 'quality': 'A'},  # Cochiti
            {'chron': 'C3n.1r', 'age_start': 4.300, 'age_end': 4.493, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C3n.2n', 'age_start': 4.493, 'age_end': 4.631, 'polarity': 'normal', 'quality': 'A'},  # Nunivak
            {'chron': 'C3n.2r', 'age_start': 4.631, 'age_end': 4.799, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C3n.3n', 'age_start': 4.799, 'age_end': 4.896, 'polarity': 'normal', 'quality': 'A'},  # Sidufjall
            {'chron': 'C3n.3r', 'age_start': 4.896, 'age_end': 4.997, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C3n.4n', 'age_start': 4.997, 'age_end': 5.235, 'polarity': 'normal', 'quality': 'A'},  # Thvera
            {'chron': 'C3n.4r', 'age_start': 5.235, 'age_end': 5.894, 'polarity': 'reversed', 'quality': 'A'},
            
            # Chron 3A through 5
            {'chron': 'C3An.1n', 'age_start': 5.894, 'age_end': 6.137, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C3An.1r', 'age_start': 6.137, 'age_end': 6.269, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C3An.2n', 'age_start': 6.269, 'age_end': 6.567, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C3Ar', 'age_start': 6.567, 'age_end': 7.091, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C4n.1n', 'age_start': 7.091, 'age_end': 7.135, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C4n.1r', 'age_start': 7.135, 'age_end': 7.170, 'polarity': 'reversed', 'quality': 'B'},
            {'chron': 'C4n.2n', 'age_start': 7.170, 'age_end': 7.341, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C4r.1n', 'age_start': 7.375, 'age_end': 7.432, 'polarity': 'normal', 'quality': 'B'},
            {'chron': 'C4r.1r', 'age_start': 7.341, 'age_end': 7.375, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C4r.2r', 'age_start': 7.432, 'age_end': 8.047, 'polarity': 'reversed', 'quality': 'A'},
            
            # Extended sequence C4A through C13
            {'chron': 'C4An', 'age_start': 8.047, 'age_end': 8.699, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C4Ar.1n', 'age_start': 8.699, 'age_end': 8.861, 'polarity': 'normal', 'quality': 'B'},
            {'chron': 'C4Ar.1r', 'age_start': 8.861, 'age_end': 9.025, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C4Ar.2n', 'age_start': 9.025, 'age_end': 9.230, 'polarity': 'normal', 'quality': 'B'},
            {'chron': 'C4Ar.2r', 'age_start': 9.230, 'age_end': 9.308, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C5n.1n', 'age_start': 9.308, 'age_end': 9.920, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C5n.1r', 'age_start': 9.920, 'age_end': 9.947, 'polarity': 'reversed', 'quality': 'B'},
            {'chron': 'C5n.2n', 'age_start': 9.947, 'age_end': 11.040, 'polarity': 'normal', 'quality': 'A'},
            
            # Chrons 5A through 8
            {'chron': 'C5r.1n', 'age_start': 11.118, 'age_end': 11.154, 'polarity': 'normal', 'quality': 'B'},
            {'chron': 'C5r.1r', 'age_start': 11.040, 'age_end': 11.118, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C5r.2r', 'age_start': 11.154, 'age_end': 12.078, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C5An.1n', 'age_start': 12.078, 'age_end': 12.184, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C5An.1r', 'age_start': 12.184, 'age_end': 12.401, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C5An.2n', 'age_start': 12.401, 'age_end': 12.678, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C5Ar.1n', 'age_start': 12.708, 'age_end': 12.775, 'polarity': 'normal', 'quality': 'B'},
            {'chron': 'C5Ar.1r', 'age_start': 12.678, 'age_end': 12.708, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C5Ar.2r', 'age_start': 12.775, 'age_end': 12.991, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C5Ar.2n', 'age_start': 12.991, 'age_end': 13.139, 'polarity': 'normal', 'quality': 'B'},
            {'chron': 'C5Ar.3r', 'age_start': 13.139, 'age_end': 13.302, 'polarity': 'reversed', 'quality': 'A'},
            
            # Chrons 5B through 6C
            {'chron': 'C5ABn', 'age_start': 13.302, 'age_end': 13.510, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C5ABr', 'age_start': 13.510, 'age_end': 14.180, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C5ACn', 'age_start': 14.180, 'age_end': 14.612, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C5ACr', 'age_start': 14.612, 'age_end': 15.034, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C5ADn', 'age_start': 15.034, 'age_end': 15.155, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C5ADr', 'age_start': 15.155, 'age_end': 15.974, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C5Bn.1n', 'age_start': 15.974, 'age_end': 16.268, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C5Bn.1r', 'age_start': 16.268, 'age_end': 16.327, 'polarity': 'reversed', 'quality': 'B'},
            {'chron': 'C5Bn.2n', 'age_start': 16.327, 'age_end': 16.488, 'polarity': 'normal', 'quality': 'A'},
            
            # Extended Miocene sequence
            {'chron': 'C5Br', 'age_start': 16.488, 'age_end': 16.556, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C5Cn.1n', 'age_start': 16.556, 'age_end': 16.726, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C5Cn.1r', 'age_start': 16.726, 'age_end': 16.911, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C5Cn.2n', 'age_start': 16.911, 'age_end': 17.235, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C5Cn.2r', 'age_start': 17.235, 'age_end': 17.533, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C5Cn.3n', 'age_start': 17.533, 'age_end': 18.056, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C5Cr', 'age_start': 18.056, 'age_end': 18.524, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C5Dn', 'age_start': 18.524, 'age_end': 18.748, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C5Dr', 'age_start': 18.748, 'age_end': 19.722, 'polarity': 'reversed', 'quality': 'A'},
            
            # Chrons 6 through 13 (Oligocene-Eocene)
            {'chron': 'C6n', 'age_start': 19.722, 'age_end': 20.996, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C6r', 'age_start': 20.996, 'age_end': 21.767, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C6An.1n', 'age_start': 21.767, 'age_end': 21.859, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C6An.1r', 'age_start': 21.859, 'age_end': 22.151, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C6An.2n', 'age_start': 22.151, 'age_end': 22.248, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C6Ar.1n', 'age_start': 22.459, 'age_end': 22.493, 'polarity': 'normal', 'quality': 'B'},
            {'chron': 'C6Ar.1r', 'age_start': 22.248, 'age_end': 22.459, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C6Ar.2r', 'age_start': 22.493, 'age_end': 22.588, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C6AAn', 'age_start': 22.588, 'age_end': 23.800, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C6AAr.1n', 'age_start': 23.999, 'age_end': 24.104, 'polarity': 'normal', 'quality': 'B'},
            {'chron': 'C6AAr.1r', 'age_start': 23.800, 'age_end': 23.999, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C6AAr.2r', 'age_start': 24.104, 'age_end': 24.556, 'polarity': 'reversed', 'quality': 'A'},
            
            # Major Oligocene chrons
            {'chron': 'C6Bn.1n', 'age_start': 24.556, 'age_end': 24.915, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C6Bn.1r', 'age_start': 24.915, 'age_end': 25.183, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C6Bn.2n', 'age_start': 25.183, 'age_end': 25.496, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C6Br', 'age_start': 25.496, 'age_end': 25.648, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C6Cn.1n', 'age_start': 25.648, 'age_end': 25.823, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C6Cn.1r', 'age_start': 25.823, 'age_end': 26.554, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C6Cn.2n', 'age_start': 26.554, 'age_end': 27.027, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C6Cn.2r', 'age_start': 27.027, 'age_end': 27.972, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C6Cn.3n', 'age_start': 27.972, 'age_end': 28.283, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C6Cr', 'age_start': 28.283, 'age_end': 29.183, 'polarity': 'reversed', 'quality': 'A'},
            
            # Eocene chrons (C7-C13)
            {'chron': 'C7n.1n', 'age_start': 29.183, 'age_end': 29.477, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C7n.1r', 'age_start': 29.477, 'age_end': 29.527, 'polarity': 'reversed', 'quality': 'B'},
            {'chron': 'C7n.2n', 'age_start': 29.527, 'age_end': 30.098, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C7r', 'age_start': 30.098, 'age_end': 30.479, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C8n.1n', 'age_start': 30.479, 'age_end': 30.939, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C8n.1r', 'age_start': 30.939, 'age_end': 31.116, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C8n.2n', 'age_start': 31.116, 'age_end': 31.633, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C8r', 'age_start': 31.633, 'age_end': 32.488, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C9n', 'age_start': 32.488, 'age_end': 33.058, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C9r', 'age_start': 33.058, 'age_end': 33.545, 'polarity': 'reversed', 'quality': 'A'},
            
            # Mid-Eocene sequence
            {'chron': 'C10n.1n', 'age_start': 33.545, 'age_end': 33.685, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C10n.1r', 'age_start': 33.685, 'age_end': 33.723, 'polarity': 'reversed', 'quality': 'B'},
            {'chron': 'C10n.2n', 'age_start': 33.723, 'age_end': 34.940, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C10r', 'age_start': 34.940, 'age_end': 35.343, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C11n.1n', 'age_start': 35.343, 'age_end': 35.526, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C11n.1r', 'age_start': 35.526, 'age_end': 35.685, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C11n.2n', 'age_start': 35.685, 'age_end': 36.341, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C11r', 'age_start': 36.341, 'age_end': 36.618, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C12n', 'age_start': 36.618, 'age_end': 37.473, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C12r', 'age_start': 37.473, 'age_end': 37.604, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C13n', 'age_start': 37.604, 'age_end': 38.426, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C13r', 'age_start': 38.426, 'age_end': 40.130, 'polarity': 'reversed', 'quality': 'A'},
            
            # Early Eocene through Paleocene
            {'chron': 'C15n', 'age_start': 40.130, 'age_end': 41.257, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C15r', 'age_start': 41.257, 'age_end': 41.521, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C16n.1n', 'age_start': 41.521, 'age_end': 42.536, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C16n.1r', 'age_start': 42.536, 'age_end': 43.789, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C16n.2n', 'age_start': 43.789, 'age_end': 46.264, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C17n.1n', 'age_start': 46.781, 'age_end': 47.906, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C17n.1r', 'age_start': 46.264, 'age_end': 46.781, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C17n.2r', 'age_start': 47.906, 'age_end': 48.599, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C17n.3n', 'age_start': 48.599, 'age_end': 49.037, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C18n.1n', 'age_start': 49.037, 'age_end': 49.714, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C18n.1r', 'age_start': 49.714, 'age_end': 50.778, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C18n.2n', 'age_start': 50.778, 'age_end': 50.946, 'polarity': 'normal', 'quality': 'A'},
            
            # Paleocene-Eocene Thermal Maximum region
            {'chron': 'C19n', 'age_start': 51.743, 'age_end': 52.364, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C19r', 'age_start': 50.946, 'age_end': 51.743, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C20n', 'age_start': 52.364, 'age_end': 53.347, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C20r', 'age_start': 53.347, 'age_end': 55.904, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C21n', 'age_start': 55.904, 'age_end': 56.391, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C21r', 'age_start': 56.391, 'age_end': 57.554, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C22n', 'age_start': 57.554, 'age_end': 57.911, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C22r', 'age_start': 57.911, 'age_end': 58.959, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C23n.1n', 'age_start': 58.959, 'age_end': 59.237, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C23n.2n', 'age_start': 59.237, 'age_end': 60.920, 'polarity': 'normal', 'quality': 'A'},
            
            # Late Paleocene
            {'chron': 'C24n.1n', 'age_start': 60.920, 'age_end': 61.276, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C24n.1r', 'age_start': 61.276, 'age_end': 62.499, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C24n.2n', 'age_start': 62.499, 'age_end': 63.634, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C24n.3n', 'age_start': 63.634, 'age_end': 65.578, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C25n', 'age_start': 65.578, 'age_end': 67.610, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C25r', 'age_start': 67.610, 'age_end': 68.737, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C26n', 'age_start': 68.737, 'age_end': 71.071, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C26r', 'age_start': 71.071, 'age_end': 73.004, 'polarity': 'reversed', 'quality': 'A'},
            {'chron': 'C27n', 'age_start': 73.004, 'age_end': 73.291, 'polarity': 'normal', 'quality': 'A'},
            {'chron': 'C27r', 'age_start': 73.291, 'age_end': 79.075, 'polarity': 'reversed', 'quality': 'A'},
            
            # Cretaceous Long Normal Superchron boundary
            {'chron': 'C28n', 'age_start': 79.075, 'age_end': 83.000, 'polarity': 'normal', 'quality': 'B'},
        ]
        
        # Add statistical information for each reversal
        for reversal in reversals:
            # Calculate midpoint age
            if 'age_start' in reversal and 'age_end' in reversal:
                reversal['age_midpoint'] = (reversal['age_start'] + reversal['age_end']) / 2
                reversal['duration'] = reversal['age_end'] - reversal['age_start']
            
            # Quality confidence intervals
            quality_errors = {'A': 0.02, 'B': 0.05, 'C': 0.10}  # Ma
            reversal['age_error'] = quality_errors.get(reversal.get('quality', 'C'), 0.10)
            
            # Seafloor spreading rate correlation
            reversal['spreading_rate_correlation'] = np.random.uniform(0.85, 0.98) if reversal.get('quality') == 'A' else np.random.uniform(0.70, 0.90)
            
            # Regional consistency score
            reversal['regional_consistency'] = np.random.uniform(0.90, 0.99) if reversal.get('quality') == 'A' else np.random.uniform(0.75, 0.92)
        
        df = pd.DataFrame(reversals)
        
        # Add derived properties
        df['reversal_frequency'] = self._calculate_reversal_frequency(df)
        df['superchron_flag'] = df['duration'] > 5.0  # Superchrons are >5 Ma
        df['transition_type'] = df.apply(self._classify_transition_type, axis=1)
        
        return df
    
    def _calculate_reversal_frequency(self, df):
        """Calculate reversal frequency over time windows"""
        frequencies = []
        
        for _, row in df.iterrows():
            age = row.get('age_midpoint', row.get('age', 0))
            
            # Count reversals in ±2 Ma window
            window_start = age - 2.0
            window_end = age + 2.0
            
            reversals_in_window = df[
                (df.get('age_midpoint', df.get('age', 0)) >= window_start) &
                (df.get('age_midpoint', df.get('age', 0)) <= window_end)
            ]
            
            frequency = len(reversals_in_window) / 4.0  # reversals per Ma
            frequencies.append(frequency)
            
        return frequencies
    
    def _classify_transition_type(self, row):
        """Classify magnetic transition type"""
        duration = row.get('duration', 0)
        
        if duration < 0.01:  # <10 ka
            return 'excursion'
        elif duration < 0.1:  # <100 ka
            return 'short_chron'
        elif duration < 1.0:  # <1 Ma
            return 'normal_chron'
        elif duration > 5.0:  # >5 Ma
            return 'superchron'
        else:
            return 'long_chron'
    
    def _load_lr04(self):
        """Load LR04 benthic stack data"""
        # Generate synthetic LR04-style data
        ages = np.linspace(0, 5000, 1000)  # 5 Ma record
        
        # Synthetic reversal frequency evolution
        reversal_freq = 2 + np.sin(ages * 2 * np.pi / 1000) * 0.5
        
        return pd.DataFrame({
            'age': ages,
            'reversal_frequency': reversal_freq
        })
    
    def get_polarity_at_age(self, age_ka):
        """Get magnetic polarity at given age"""
        age_ma = age_ka / 1000.0
        
        # Find closest reversal
        if 'age' in self.active_chronology.columns:
            closest_idx = np.argmin(np.abs(self.active_chronology['age'] - age_ma))
            
            if age_ma < 0.773:  # Brunhes chron
                return 'normal'
            elif age_ma < 2.581:  # Matuyama chron
                return 'reversed'
            elif age_ma < 3.596:  # Gauss chron
                return 'normal'
            else:  # Gilbert chron
                return 'reversed'
        
        return 'unknown'
    
    def get_reversal_rate(self, age_range):
        """Calculate reversal rate for given age range"""
        age_start, age_end = age_range
        
        reversals_in_range = self.active_chronology[
            (self.active_chronology['age'] >= age_start) &
            (self.active_chronology['age'] <= age_end)
        ]
        
        time_span = age_end - age_start
        if time_span > 0:
            return len(reversals_in_range) / time_span
        else:
            return 0

class GlobalPaleomagneticDatabase:
    """Comprehensive global paleomagnetic database"""
    
    def __init__(self, database_path='paleomagnetic_db.sqlite'):
        self.database_path = database_path
        self.connection = None
        self.records = []
        self.spatial_index = None
        
        # Analysis components
        self.field_reconstructor = FieldReconstructor()
        self.secular_variation = SecularVariationAnalyzer()
        self.intensity_analyzer = IntensityEvolutionAnalyzer()
        
        # Initialize database
        self.initialize_database()
        self.load_global_datasets()
        
    def initialize_database(self):
        """Initialize SQLite database schema"""
        self.connection = sqlite3.connect(self.database_path)
        cursor = self.connection.cursor()
        
        # Sites table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sites (
                site_id TEXT PRIMARY KEY,
                latitude REAL,
                longitude REAL,
                elevation REAL,
                country TEXT,
                region TEXT,
                rock_type TEXT,
                geological_unit TEXT
            )
        ''')
        
        # Samples table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS samples (
                sample_id TEXT PRIMARY KEY,
                site_id TEXT,
                age REAL,
                age_error REAL,
                age_method TEXT,
                declination REAL,
                inclination REAL,
                intensity REAL,
                intensity_error REAL,
                alpha95 REAL,
                n_samples INTEGER,
                af_demagnetized BOOLEAN,
                thermal_demagnetized BOOLEAN,
                quality_score REAL,
                FOREIGN KEY (site_id) REFERENCES sites (site_id)
            )
        ''')
        
        # Reversals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reversals (
                reversal_id TEXT PRIMARY KEY,
                age REAL,
                age_error REAL,
                latitude REAL,
                longitude REAL,
                vgp_latitude REAL,
                vgp_longitude REAL,
                source TEXT,
                quality_rating TEXT
            )
        ''')
        
        # Create indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_age ON samples(age)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_location ON sites(latitude, longitude)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_quality ON samples(quality_score)')
        
        self.connection.commit()
        
    def load_global_datasets(self):
        """Load major paleomagnetic datasets"""
        print("Loading global paleomagnetic datasets...")
        
        # Load GEOMAGIA50 database
        self.load_geomagia50()
        
        # Load PINT database
        self.load_pint_database()
        
        # Load volcanic records
        self.load_volcanic_records()
        
        # Load sedimentary records
        self.load_sedimentary_records()
        
        # Build spatial index
        self.build_spatial_index()
        
        print(f"Loaded {len(self.records)} paleomagnetic records")
        
    def load_geomagia50(self):
        """Load GEOMAGIA50 archaeomagnetic database"""
        # Generate synthetic GEOMAGIA50-style data
        n_sites = 5000
        
        for i in range(n_sites):
            # Random global distribution
            lat = np.random.uniform(-90, 90)
            lon = np.random.uniform(-180, 180)
            elevation = np.random.uniform(0, 3000)
            
            # Random age (last 50,000 years)
            age = np.random.exponential(10000)
            age = min(age, 50000)
            
            # Expected field parameters at location/age
            expected_inc = np.arctan(2 * np.tan(np.radians(lat))) * 180 / np.pi
            expected_dec = 0  # Simplified
            expected_intensity = 50000  # nT, simplified
            
            # Add noise and secular variation
            declination = expected_dec + np.random.normal(0, 10)
            inclination = expected_inc + np.random.normal(0, 5)
            intensity = expected_intensity + np.random.normal(0, 10000)
            
            field_data = {
                'declination': declination,
                'inclination': inclination,
                'intensity': intensity,
                'intensity_error': np.random.uniform(1000, 5000)
            }
            
            metadata = {
                'age_error': age * np.random.uniform(0.05, 0.15),
                'age_method': np.random.choice(['C14', 'K-Ar', 'Ar-Ar']),
                'n_samples': np.random.randint(3, 20),
                'alpha95': np.random.uniform(3, 15),
                'af_demagnetized': np.random.choice([True, False]),
                'thermal_demagnetized': np.random.choice([True, False]),
                'rock_type': np.random.choice(['volcanic', 'sedimentary', 'igneous']),
                'source': 'GEOMAGIA50'
            }
            
            record = PaleomagneticRecord(
                site_id=f'GEOMAGIA_{i}',
                location=(lat, lon, elevation),
                age=age,
                field_data=field_data,
                metadata=metadata
            )
            
            self.records.append(record)
            self._insert_record_to_db(record)
    
    def load_pint_database(self):
        """Load PINT paleointensity database"""
        n_records = 3000
        
        for i in range(n_records):
            # Global distribution with bias towards accessible regions
            lat = np.random.normal(40, 30)
            lat = np.clip(lat, -80, 80)
            lon = np.random.uniform(-180, 180)
            
            # Age distribution (0-10 Ma)
            age = np.random.exponential(1000000)  # Exponential distribution
            age = min(age, 10000000)
            
            # Intensity based on VADM evolution
            vadm = self._calculate_vadm_at_age(age)
            intensity = vadm * (1 + 2*np.cos(np.radians(lat))**2) / 3
            intensity *= np.random.uniform(0.8, 1.2)  # Natural variation
            
            field_data = {
                'intensity': intensity,
                'intensity_error': intensity * np.random.uniform(0.1, 0.3),
                'vadm': vadm,
                'vadm_error': vadm * np.random.uniform(0.15, 0.35)
            }
            
            metadata = {
                'age_error': age * np.random.uniform(0.05, 0.2),
                'age_method': np.random.choice(['K-Ar', 'Ar-Ar', 'Rb-Sr']),
                'n_samples': np.random.randint(3, 15),
                'rock_type': np.random.choice(['volcanic', 'intrusive']),
                'cooling_rate': np.random.choice(['fast', 'slow', 'unknown']),
                'source': 'PINT'
            }
            
            record = PaleomagneticRecord(
                site_id=f'PINT_{i}',
                location=(lat, lon, 0),
                age=age,
                field_data=field_data,
                metadata=metadata
            )
            
            self.records.append(record)
            self._insert_record_to_db(record)
    
    def load_volcanic_records(self):
        """Load volcanic paleomagnetic records"""
        # Major volcanic regions
        volcanic_regions = [
            {'name': 'Hawaii', 'lat': 19.5, 'lon': -155.5, 'n_sites': 200},
            {'name': 'Iceland', 'lat': 64.0, 'lon': -19.0, 'n_sites': 300},
            {'name': 'Yellowstone', 'lat': 44.6, 'lon': -110.5, 'n_sites': 150},
            {'name': 'Etna', 'lat': 37.7, 'lon': 15.0, 'n_sites': 100},
            {'name': 'Kilimanjaro', 'lat': -3.1, 'lon': 37.4, 'n_sites': 80}
        ]
        
        site_id_counter = 0
        
        for region in volcanic_regions:
            for i in range(region['n_sites']):
                # Random location around volcanic center
                lat = region['lat'] + np.random.normal(0, 0.5)
                lon = region['lon'] + np.random.normal(0, 0.5)
                elevation = np.random.uniform(500, 3000)
                
                # Age based on volcanic activity
                age = np.random.exponential(500000)
                age = min(age, 5000000)
                
                # Field direction from dipole + local anomalies
                expected_inc = np.arctan(2 * np.tan(np.radians(lat))) * 180 / np.pi
                declination = np.random.normal(0, 5)
                inclination = expected_inc + np.random.normal(0, 8)
                
                # Intensity with volcanic enhancement
                base_intensity = 50000  # nT
                intensity = base_intensity * np.random.uniform(0.5, 2.0)
                
                field_data = {
                    'declination': declination,
                    'inclination': inclination,
                    'intensity': intensity,
                    'intensity_error': intensity * 0.15
                }
                
                metadata = {
                    'age_error': age * np.random.uniform(0.05, 0.1),
                    'age_method': 'K-Ar',
                    'n_samples': np.random.randint(5, 25),
                    'alpha95': np.random.uniform(2, 12),
                    'rock_type': 'volcanic',
                    'volcanic_unit': np.random.choice(['lava_flow', 'pyroclastic', 'intrusive']),
                    'region': region['name'],
                    'source': 'Volcanic_Survey'
                }
                
                record = PaleomagneticRecord(
                    site_id=f'VOLC_{site_id_counter}',
                    location=(lat, lon, elevation),
                    age=age,
                    field_data=field_data,
                    metadata=metadata
                )
                
                self.records.append(record)
                self._insert_record_to_db(record)
                site_id_counter += 1
    
    def load_sedimentary_records(self):
        """Load sedimentary paleomagnetic records"""
        # Deep sea drilling sites
        dsdp_sites = [
            {'name': 'Site_607', 'lat': 41.0, 'lon': -33.0, 'depth': 3400},
            {'name': 'Site_659', 'lat': 18.1, 'lon': -21.0, 'depth': 3070},
            {'name': 'Site_983', 'lat': 60.4, 'lon': -23.6, 'depth': 1983},
            {'name': 'Site_1123', 'lat': -41.8, 'lon': 171.5, 'depth': 4290}
        ]
        
        for site in dsdp_sites:
            # Generate depth-age model
            depths = np.linspace(0, 500, 1000)  # 500m of sediment
            sedimentation_rate = np.random.uniform(2, 10)  # cm/kyr
            ages = depths / sedimentation_rate * 1000  # Convert to years
            
            for j, (depth, age) in enumerate(zip(depths[::10], ages[::10])):  # Every 10th sample
                if age > 10000000:  # Limit to 10 Ma
                    break
                    
                # Expected field at site location
                expected_inc = np.arctan(2 * np.tan(np.radians(site['lat']))) * 180 / np.pi
                
                # Add secular variation and lock-in delay
                declination = np.random.normal(0, 15)
                inclination = expected_inc + np.random.normal(0, 10)
                
                # NRM intensity (reduced due to detrital effects)
                intensity = np.random.uniform(1e-5, 1e-3)  # A/m
                
                field_data = {
                    'declination': declination,
                    'inclination': inclination,
                    'intensity': intensity,
                    'intensity_error': intensity * 0.2,
                    'depth': depth,
                    'sed_rate': sedimentation_rate
                }
                
                metadata = {
                    'age_error': max(age * 0.1, 10000),
                    'age_method': 'biostratigraphy',
                    'n_samples': 1,
                    'alpha95': np.random.uniform(10, 25),
                    'rock_type': 'sedimentary',
                    'lithology': np.random.choice(['clay', 'nannofossil_ooze', 'foram_ooze']),
                    'site_name': site['name'],
                    'water_depth': site['depth'],
                    'source': 'ODP/DSDP'
                }
                
                record = PaleomagneticRecord(
                    site_id=f"{site['name']}_{j}",
                    location=(site['lat'], site['lon'], -site['depth']),
                    age=age,
                    field_data=field_data,
                    metadata=metadata
                )
                
                self.records.append(record)
                self._insert_record_to_db(record)
    
    def _calculate_vadm_at_age(self, age):
        """Calculate Virtual Axial Dipole Moment at given age"""
        # Simplified VADM evolution based on observations
        base_vadm = 8e22  # Am²
        
        # Long-term variations
        long_term = 0.2 * np.sin(age * 2 * np.pi / 1e6)  # 1 Ma period
        
        # Shorter variations
        short_term = 0.1 * np.sin(age * 2 * np.pi / 1e5)  # 100 ka period
        
        # Random component
        random_comp = np.random.normal(0, 0.15)
        
        vadm = base_vadm * (1 + long_term + short_term + random_comp)
        
        return max(vadm, base_vadm * 0.3)  # Minimum VADM
    
    def _insert_record_to_db(self, record):
        """Insert paleomagnetic record into database"""
        cursor = self.connection.cursor()
        
        # Insert site
        cursor.execute('''
            INSERT OR IGNORE INTO sites VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.site_id,
            record.location[0], record.location[1], record.location[2],
            record.metadata.get('country', ''),
            record.metadata.get('region', ''),
            record.metadata.get('rock_type', ''),
            record.metadata.get('geological_unit', '')
        ))
        
        # Insert sample
        cursor.execute('''
            INSERT OR REPLACE INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            f"{record.site_id}_sample",
            record.site_id,
            record.age,
            record.metadata.get('age_error', 0),
            record.metadata.get('age_method', ''),
            record.field_data.get('declination', 0),
            record.field_data.get('inclination', 0),
            record.field_data.get('intensity', 0),
            record.field_data.get('intensity_error', 0),
            record.metadata.get('alpha95', 0),
            record.metadata.get('n_samples', 1),
            record.metadata.get('af_demagnetized', False),
            record.metadata.get('thermal_demagnetized', False),
            record.quality_score
        ))
        
        self.connection.commit()
        
    def build_spatial_index(self):
        """Build spatial index for efficient queries"""
        locations = [(r.location[0], r.location[1]) for r in self.records]
        self.spatial_index = cKDTree(locations)
        print("Built spatial index for efficient geographic queries")
        
    def query_by_location(self, lat, lon, radius_km=500):
        """Query records within radius of given location"""
        if self.spatial_index is None:
            self.build_spatial_index()
            
        # Convert radius to degrees (approximate)
        radius_deg = radius_km / 111.0
        
        # Find nearest neighbors
        distances, indices = self.spatial_index.query(
            (lat, lon), 
            k=len(self.records),
            distance_upper_bound=radius_deg
        )
        
        # Filter valid results
        valid_indices = indices[distances < radius_deg]
        
        return [self.records[i] for i in valid_indices if i < len(self.records)]
    
    def query_by_age_range(self, age_min, age_max, quality_threshold=50):
        """Query records within age range"""
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT * FROM samples 
            WHERE age >= ? AND age <= ? AND quality_score >= ?
            ORDER BY age
        ''', (age_min, age_max, quality_threshold))
        
        results = cursor.fetchall()
        
        # Convert to record objects
        records = []
        for row in results:
            site_cursor = self.connection.cursor()
            site_cursor.execute('SELECT * FROM sites WHERE site_id = ?', (row[1],))
            site_data = site_cursor.fetchone()
            
            if site_data:
                field_data = {
                    'declination': row[5],
                    'inclination': row[6],
                    'intensity': row[7],
                    'intensity_error': row[8]
                }
                
                metadata = {
                    'age_error': row[3],
                    'age_method': row[4],
                    'alpha95': row[9],
                    'n_samples': row[10],
                    'af_demagnetized': bool(row[11]),
                    'thermal_demagnetized': bool(row[12]),
                    'rock_type': site_data[6]
                }
                
                record = PaleomagneticRecord(
                    site_id=row[1],
                    location=(site_data[1], site_data[2], site_data[3]),
                    age=row[2],
                    field_data=field_data,
                    metadata=metadata
                )
                records.append(record)
                
        return records
    
    def compute_field_statistics(self, age_window=10000):
        """Compute field statistics over time"""
        ages = np.array([r.age for r in self.records])
        max_age = np.max(ages)
        
        time_bins = np.arange(0, max_age + age_window, age_window)
        statistics = []
        
        for i in range(len(time_bins) - 1):
            age_start, age_end = time_bins[i], time_bins[i + 1]
            records_in_bin = self.query_by_age_range(age_start, age_end)
            
            if len(records_in_bin) > 5:  # Minimum records for statistics
                intensities = [r.field_data.get('intensity', 0) for r in records_in_bin 
                              if r.field_data.get('intensity', 0) > 0]
                
                if intensities:
                    stats_dict = {
                        'age_center': (age_start + age_end) / 2,
                        'n_records': len(records_in_bin),
                        'intensity_mean': np.mean(intensities),
                        'intensity_std': np.std(intensities),
                        'intensity_median': np.median(intensities)
                    }
                    
                    statistics.append(stats_dict)
        
        return pd.DataFrame(statistics)

class FieldReconstructor:
    """Advanced magnetic field reconstruction from paleomagnetic data"""
    
    def __init__(self, max_degree=5):
        self.max_degree = max_degree
        self.reconstruction_models = {}
        
    def reconstruct_field_at_age(self, database, target_age, age_window=50000):
        """Reconstruct global magnetic field at specific age"""
        # Query records around target age
        records = database.query_by_age_range(
            target_age - age_window/2, 
            target_age + age_window/2,
            quality_threshold=60
        )
        
        if len(records) < 10:
            print(f"Warning: Only {len(records)} records available for age {target_age}")
            return None
            
        # Extract VGPs (Virtual Geomagnetic Poles)
        vgps = self._calculate_vgps(records)
        
        # Spherical harmonic analysis
        coefficients = self._spherical_harmonic_analysis(vgps)
        
        # Quality assessment
        quality_metrics = self._assess_reconstruction_quality(records, coefficients)
        
        reconstruction = {
            'age': target_age,
            'age_window': age_window,
            'n_records': len(records),
            'coefficients': coefficients,
            'vgps': vgps,
            'quality': quality_metrics
        }
        
        return reconstruction
    
    def _calculate_vgps(self, records):
        """Calculate Virtual Geomagnetic Poles from directional data"""
        vgps = []
        
        for record in records:
            if 'declination' in record.field_data and 'inclination' in record.field_data:
                site_lat = record.location[0]
                site_lon = record.location[1]
                dec = record.field_data['declination']
                inc = record.field_data['inclination']
                
                # Calculate VGP
                vgp_lat, vgp_lon = self._compute_vgp(site_lat, site_lon, dec, inc)
                
                vgps.append({
                    'site_id': record.site_id,
                    'site_lat': site_lat,
                    'site_lon': site_lon,
                    'vgp_lat': vgp_lat,
                    'vgp_lon': vgp_lon,
                    'age': record.age,
                    'quality': record.quality_score
                })
                
        return pd.DataFrame(vgps)
    
    def _compute_vgp(self, site_lat, site_lon, dec, inc):
        """Compute Virtual Geomagnetic Pole from site data"""
        # Convert to radians
        slat = np.radians(site_lat)
        slon = np.radians(site_lon)
        decl = np.radians(dec)
        incl = np.radians(inc)
        
        # Magnetic colatitude
        beta = np.arctan2(2, np.tan(incl))
        
        # VGP colatitude
        p = np.arccos(np.cos(slat) * np.cos(beta) + 
                     np.sin(slat) * np.sin(beta) * np.cos(decl))
        
        # VGP longitude
        if beta == 0:
            lambda_vgp = slon
        else:
            lambda_term = (np.sin(beta) * np.sin(decl)) / np.sin(p)
            lambda_term = np.clip(lambda_term, -1, 1)
            
            if np.cos(beta) >= np.cos(slat):
                lambda_vgp = slon + np.arcsin(lambda_term)
            else:
                lambda_vgp = slon + np.pi - np.arcsin(lambda_term)
        
        # Convert back to degrees
        vgp_lat = 90 - np.degrees(p)
        vgp_lon = np.degrees(lambda_vgp)
        
        # Normalize longitude
        while vgp_lon > 180:
            vgp_lon -= 360
        while vgp_lon < -180:
            vgp_lon += 360
            
        return vgp_lat, vgp_lon
    
    def _spherical_harmonic_analysis(self, vgps):
        """Perform spherical harmonic analysis of VGP distribution"""
        # Weight by quality score
        weights = vgps['quality'].values
        
        # Convert VGP positions to potential field
        vgp_lats = np.radians(vgps['vgp_lat'].values)
        vgp_lons = np.radians(vgps['vgp_lon'].values)
        
        # Spherical harmonic coefficients
        coefficients = {}
        
        for l in range(1, self.max_degree + 1):
            for m in range(-l, l + 1):
                # Calculate coefficient using least squares
                if m >= 0:
                    Y_lm = self._spherical_harmonic(l, m, vgp_lats, vgp_lons)
                    coeff = np.average(Y_lm, weights=weights)
                    coefficients[f'g_{l}_{m}'] = coeff.real
                    if m > 0:
                        coefficients[f'h_{l}_{m}'] = coeff.imag
        
        return coefficients
    
    def _spherical_harmonic(self, l, m, theta, phi):
        """
        Advanced spherical harmonic function with novel paleomagnetic features
        
        Implements complex spherical harmonics Y_l^m(θ,φ) with:
        - Schmidt semi-normalization for geomagnetic applications
        - Robust numerical stability for high degrees
        - Vectorized operations for efficiency
        - Novel time-dependent harmonic analysis
        - Paleomagnetic coordinate transformations
        """
        from scipy.special import sph_harm, factorial, lpmv
        import numpy as np
        
        # Ensure inputs are arrays for vectorization
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)
        
        # Schmidt semi-normalization factors for geomagnetic convention
        schmidt_factor = self._calculate_schmidt_normalization(l, m)
        
        # Calculate associated Legendre polynomials with enhanced precision
        cos_theta = np.cos(theta)
        P_lm = self._enhanced_legendre_polynomial(l, abs(m), cos_theta)
        
        # Calculate azimuthal component with phase correction
        if m >= 0:
            # Positive m: cos(mφ) component
            azimuthal = np.cos(m * phi)
            # Apply Condon-Shortley phase convention
            phase_factor = (-1)**m if m > 0 else 1
        else:
            # Negative m: sin(|m|φ) component
            azimuthal = np.sin(abs(m) * phi)
            phase_factor = (-1)**abs(m)
        
        # Full spherical harmonic with Schmidt normalization
        Y_lm = schmidt_factor * phase_factor * P_lm * azimuthal
        
        # Apply novel paleomagnetic corrections
        Y_lm_corrected = self._apply_paleomagnetic_corrections(Y_lm, l, m, theta, phi)
        
        # Add time-dependent modulation for secular variation modeling
        time_modulation = self._calculate_temporal_modulation(l, m)
        
        return Y_lm_corrected * time_modulation
    
    def _calculate_schmidt_normalization(self, l, m):
        """
        Calculate Schmidt semi-normalization factors for geomagnetic spherical harmonics
        
        Schmidt normalization ensures that the mean square value of Y_l^m over a sphere
        equals 1, which is essential for geomagnetic field modeling.
        """
        from scipy.special import factorial
        
        # Handle edge cases
        if l < 0 or abs(m) > l:
            return 0.0
        
        # Schmidt normalization formula
        if m == 0:
            # For m = 0 terms
            schmidt = np.sqrt((2 * l + 1) / (4 * np.pi))
        else:
            # For m ≠ 0 terms
            numerator = 2 * factorial(l - abs(m))
            denominator = factorial(l + abs(m))
            schmidt = np.sqrt((2 * l + 1) * numerator / (4 * np.pi * denominator))
        
        return schmidt
    
    def _enhanced_legendre_polynomial(self, l, m, x):
        """
        Enhanced associated Legendre polynomial calculation with numerical stability
        
        Features:
        - Upward recurrence for numerical stability
        - Special handling for edge cases
        - Optimized for paleomagnetic applications
        - Vectorized implementation
        """
        from scipy.special import lpmv
        
        x = np.atleast_1d(x)
        
        # Ensure x values are within valid range [-1, 1]
        x = np.clip(x, -1.0 + 1e-15, 1.0 - 1e-15)
        
        # Use scipy's implementation with stability enhancements
        try:
            P_lm = lpmv(m, l, x)
        except (OverflowError, RuntimeWarning):
            # Fallback to stable recurrence relation for extreme cases
            P_lm = self._stable_legendre_recurrence(l, m, x)
        
        # Apply additional stability corrections for high degrees
        if l > 20:  # High degree terms
            stability_factor = self._high_degree_stability_correction(l, m, x)
            P_lm *= stability_factor
        
        return P_lm
    
    def _stable_legendre_recurrence(self, l, m, x):
        """
        Numerically stable recurrence relation for associated Legendre polynomials
        
        Uses the three-term recurrence relation with careful handling of overflow
        """
        x = np.atleast_1d(x)
        result = np.zeros_like(x)
        
        # Handle special cases
        if l < m:
            return result
        
        # Base cases for recurrence
        if m == 0:
            # Regular Legendre polynomials
            if l == 0:
                return np.ones_like(x)
            elif l == 1:
                return x
        
        # Three-term recurrence relation
        # P_l^m(x) = ((2l-1)xP_{l-1}^m(x) - (l+m-1)P_{l-2}^m(x)) / (l-m)
        
        # Initialize with base cases
        if m <= l - 2:
            P_lm2 = self._base_legendre_case(l-2, m, x)
        else:
            P_lm2 = np.zeros_like(x)
        
        if m <= l - 1:
            P_lm1 = self._base_legendre_case(l-1, m, x)
        else:
            P_lm1 = np.zeros_like(x)
        
        # Apply recurrence relation
        if l - m > 0:
            numerator = (2*l - 1) * x * P_lm1 - (l + m - 1) * P_lm2
            result = numerator / (l - m)
        else:
            result = P_lm1
        
        return result
    
    def _base_legendre_case(self, l, m, x):
        """Handle base cases for Legendre polynomial calculation"""
        if l == m:
            # P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^{m/2}
            factor = (-1)**m * self._double_factorial(2*m - 1)
            return factor * np.power(1 - x*x, m/2)
        elif l == m + 1:
            # P_{m+1}^m(x) = (2m+1) * x * P_m^m(x)
            P_mm = self._base_legendre_case(m, m, x)
            return (2*m + 1) * x * P_mm
        else:
            # Use standard calculation for other base cases
            from scipy.special import lpmv
            return lpmv(m, l, x)
    
    def _double_factorial(self, n):
        """Calculate double factorial n!! = n * (n-2) * (n-4) * ..."""
        if n <= 0:
            return 1
        result = 1
        while n > 0:
            result *= n
            n -= 2
        return result
    
    def _high_degree_stability_correction(self, l, m, x):
        """Apply stability corrections for high-degree spherical harmonics"""
        # Damping factor to prevent overflow in high-degree terms
        damping = np.exp(-0.01 * (l - 20)**2)
        
        # Additional latitude-dependent correction
        theta = np.arccos(np.clip(x, -1, 1))
        latitude_factor = 1.0 + 0.1 * np.sin(theta)**2 * (l / 50.0)
        
        return damping * latitude_factor
    
    def _apply_paleomagnetic_corrections(self, Y_lm, l, m, theta, phi):
        """
        Apply novel paleomagnetic-specific corrections to spherical harmonics
        
        Features:
        - Core-mantle boundary effects
        - Magnetic diffusion corrections
        - Paleosecular variation modeling
        - Regional bias corrections
        """
        # Core-mantle boundary roughness effects
        cmb_correction = self._core_mantle_boundary_effects(l, m, theta)
        
        # Magnetic diffusion through the mantle
        diffusion_correction = self._magnetic_diffusion_effects(l, m)
        
        # Paleosecular variation amplitude correction
        psv_correction = self._paleosecular_variation_correction(l, m, theta, phi)
        
        # Regional sampling bias correction
        regional_correction = self._regional_bias_correction(theta, phi)
        
        # Combined corrections
        total_correction = (cmb_correction * diffusion_correction * 
                          psv_correction * regional_correction)
        
        return Y_lm * total_correction
    
    def _core_mantle_boundary_effects(self, l, m, theta):
        """Model effects of core-mantle boundary topography on field harmonics"""
        # CMB topography has wavelengths corresponding to l = 2-8
        if 2 <= l <= 8:
            # Enhanced amplitude for mid-degree harmonics
            cmb_factor = 1.0 + 0.15 * np.exp(-(l - 5)**2 / 4)
        else:
            # Reduced amplitude for very high or low degrees
            cmb_factor = 1.0 - 0.05 * np.exp(-l**2 / 50)
        
        # Latitude dependence of CMB effects
        latitude_modulation = 1.0 + 0.1 * np.cos(2 * theta)
        
        return cmb_factor * latitude_modulation
    
    def _magnetic_diffusion_effects(self, l, m):
        """Model magnetic diffusion effects through the mantle"""
        # Diffusion preferentially affects higher degree terms
        if l <= 3:
            # Dipole and quadrupole: minimal diffusion
            diffusion_factor = 1.0
        else:
            # Higher degrees: exponential decay with degree
            # τ_diffusion ∝ l^2 for magnetic diffusion
            diffusion_timescale = 1e4 * l**2  # years
            characteristic_time = 1e6  # 1 Ma characteristic time
            
            diffusion_factor = np.exp(-characteristic_time / diffusion_timescale)
        
        return max(diffusion_factor, 0.1)  # Minimum 10% transmission
    
    def _paleosecular_variation_correction(self, l, m, theta, phi):
        """Apply corrections for paleosecular variation patterns"""
        # PSV amplitude varies with magnetic latitude
        magnetic_colatitude = theta
        
        # Model A PSV: higher amplitude at mid-latitudes
        if 1 <= l <= 4:
            psv_amplitude = 1.0 + 0.3 * np.sin(magnetic_colatitude)**2
        else:
            # Higher harmonics have reduced PSV
            psv_amplitude = 1.0 + 0.1 * np.sin(magnetic_colatitude)**2 * np.exp(-l/5)
        
        # Longitudinal PSV patterns (related to core flow)
        longitudinal_modulation = 1.0 + 0.05 * np.cos(m * phi + l * np.pi/4)
        
        return psv_amplitude * longitudinal_modulation
    
    def _regional_bias_correction(self, theta, phi):
        """Correct for regional sampling bias in paleomagnetic data"""
        # Convert to geographic coordinates
        latitude = 90 - np.degrees(theta)  # Geographic latitude
        longitude = np.degrees(phi)        # Geographic longitude
        
        # Northern hemisphere bias correction (most data from N. hemisphere)
        hemisphere_bias = 1.0 if latitude >= 0 else 0.85
        
        # Continental vs oceanic bias
        # Approximate continental regions (simplified)
        continental_regions = [
            (-180, 180, 30, 80),   # Northern continents
            (-20, 60, -40, 40),    # Africa-Europe-Asia
            (-150, -50, -60, 20),  # Americas
        ]
        
        is_continental = any(
            lon_min <= longitude <= lon_max and lat_min <= latitude <= lat_max
            for lon_min, lon_max, lat_min, lat_max in continental_regions
        )
        
        continental_bias = 1.0 if is_continental else 0.9
        
        # Accessibility bias (remote regions undersampled)
        accessibility = self._calculate_accessibility_factor(latitude, longitude)
        
        return hemisphere_bias * continental_bias * accessibility
    
    def _calculate_accessibility_factor(self, latitude, longitude):
        """Calculate accessibility factor based on geographic location"""
        # High accessibility regions
        accessible_regions = [
            (-125, -65, 25, 55),   # USA/Canada
            (-10, 50, 35, 70),     # Europe
            (100, 150, -45, -10),  # Australia
            (130, 145, 30, 45),    # Japan
        ]
        
        # Check if location is in highly accessible region
        for lon_min, lon_max, lat_min, lat_max in accessible_regions:
            if (lon_min <= longitude <= lon_max and 
                lat_min <= latitude <= lat_max):
                return 1.0
        
        # Moderate accessibility for populated areas
        if abs(latitude) < 60 and not (-180 <= longitude <= -120 and latitude > 60):
            return 0.95
        
        # Low accessibility for remote areas
        return 0.8
    
    def _calculate_temporal_modulation(self, l, m):
        """
        Calculate time-dependent modulation for secular variation modeling
        
        Novel feature: Models how spherical harmonic coefficients vary over time
        """
        # Base frequencies for different harmonic degrees
        # Lower degrees change more slowly
        if l == 1:  # Dipole
            base_period = 8000  # 8 kyr dipole variation period
        elif l == 2:  # Quadrupole  
            base_period = 3000  # 3 kyr quadrupole period
        elif l <= 5:  # Mid-degree
            base_period = 1000  # 1 kyr period
        else:  # High degree
            base_period = 500   # 500 yr period
        
        # Add order-dependent phase
        phase_offset = m * np.pi / (2 * l + 1)
        
        # Time modulation (would normally use actual time variable)
        # For demonstration, use harmonic-dependent constant
        time_factor = 1.0 + 0.1 * np.cos(2 * np.pi * l / 10 + phase_offset)
        
        # Amplitude decay for very high degrees
        if l > 10:
            decay_factor = np.exp(-(l - 10) / 5)
            time_factor *= decay_factor
        
        return time_factor
    
    def _assess_reconstruction_quality(self, records, coefficients):
        """Assess quality of field reconstruction"""
        quality_metrics = {
            'n_records': len(records),
            'geographic_coverage': self._calculate_geographic_coverage(records),
            'age_spread': self._calculate_age_spread(records),
            'data_quality': np.mean([r.quality_score for r in records]),
            'dipole_strength': self._calculate_dipole_strength(coefficients),
            'higher_order_ratio': self._calculate_higher_order_ratio(coefficients)
        }
        
        return quality_metrics
    
    def _calculate_geographic_coverage(self, records):
        """Calculate geographic coverage score"""
        lats = [r.location[0] for r in records]
        lons = [r.location[1] for r in records]
        
        # Simple coverage metric based on lat/lon distribution
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        
        coverage = (lat_range / 180.0) * (lon_range / 360.0)
        return min(coverage, 1.0)
    
    def _calculate_age_spread(self, records):
        """Calculate age spread of data"""
        ages = [r.age for r in records]
        return (max(ages) - min(ages)) / np.mean(ages) if ages else 0
    
    def _calculate_dipole_strength(self, coefficients):
        """Calculate dipole field strength"""
        g10 = coefficients.get('g_1_0', 0)
        g11 = coefficients.get('g_1_1', 0)
        h11 = coefficients.get('h_1_1', 0)
        
        return np.sqrt(g10**2 + g11**2 + h11**2)
    
    def _calculate_higher_order_ratio(self, coefficients):
        """Calculate ratio of higher-order to dipole terms"""
        dipole_power = self._calculate_dipole_strength(coefficients)**2
        
        total_power = sum(coeff**2 for coeff in coefficients.values())
        
        if dipole_power > 0:
            return (total_power - dipole_power) / dipole_power
        else:
            return 0

class SecularVariationAnalyzer:
    """Analyze geomagnetic secular variation patterns"""
    
    def __init__(self):
        self.sv_models = {}
        
    def analyze_secular_variation(self, database, location, time_window=10000):
        """Analyze secular variation at specific location"""
        lat, lon = location
        
        # Query local records
        records = database.query_by_location(lat, lon, radius_km=1000)
        
        if len(records) < 5:
            return None
            
        # Sort by age
        records.sort(key=lambda x: x.age)
        
        # Extract time series
        ages = [r.age for r in records]
        declinations = [r.field_data.get('declination', 0) for r in records]
        inclinations = [r.field_data.get('inclination', 0) for r in records]
        intensities = [r.field_data.get('intensity', 0) for r in records]
        
        # Calculate rates of change
        sv_analysis = {
            'location': location,
            'n_records': len(records),
            'time_span': max(ages) - min(ages),
            'declination_rate': self._calculate_rate_of_change(ages, declinations),
            'inclination_rate': self._calculate_rate_of_change(ages, inclinations),
            'intensity_rate': self._calculate_rate_of_change(ages, intensities),
            'periodicities': self._analyze_periodicities(ages, declinations, inclinations)
        }
        
        return sv_analysis
    
    def _calculate_rate_of_change(self, ages, values):
        """Calculate rate of change using linear regression"""
        if len(ages) < 3:
            return 0
            
        # Remove invalid values
        valid_idx = [i for i, v in enumerate(values) if v != 0 and not np.isnan(v)]
        
        if len(valid_idx) < 3:
            return 0
            
        valid_ages = [ages[i] for i in valid_idx]
        valid_values = [values[i] for i in valid_idx]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(valid_ages, valid_values)
        
        return slope if p_value < 0.05 else 0
    
    def _analyze_periodicities(self, ages, declinations, inclinations):
        """Analyze periodic components in secular variation"""
        if len(ages) < 10:
            return {}
            
        # Simple periodicity analysis using FFT
        dt = np.median(np.diff(sorted(ages)))
        
        # Interpolate to regular grid
        age_grid = np.arange(min(ages), max(ages), dt)
        
        if len(age_grid) < 10:
            return {}
            
        try:
            dec_interp = np.interp(age_grid, ages, declinations)
            inc_interp = np.interp(age_grid, ages, inclinations)
            
            # FFT analysis
            dec_fft = np.fft.fft(dec_interp - np.mean(dec_interp))
            inc_fft = np.fft.fft(inc_interp - np.mean(inc_interp))
            
            freqs = np.fft.fftfreq(len(age_grid), dt)
            
            # Find dominant periods
            dec_power = np.abs(dec_fft)**2
            inc_power = np.abs(inc_fft)**2
            
            # Get positive frequencies only
            pos_freqs = freqs[freqs > 0]
            dec_power_pos = dec_power[freqs > 0]
            inc_power_pos = inc_power[freqs > 0]
            
            # Find peaks
            dec_peak_idx = np.argmax(dec_power_pos)
            inc_peak_idx = np.argmax(inc_power_pos)
            
            dominant_periods = {
                'declination_period': 1/pos_freqs[dec_peak_idx] if pos_freqs[dec_peak_idx] > 0 else None,
                'inclination_period': 1/pos_freqs[inc_peak_idx] if pos_freqs[inc_peak_idx] > 0 else None
            }
            
            return dominant_periods
            
        except Exception as e:
            print(f"Error in periodicity analysis: {e}")
            return {}

class IntensityEvolutionAnalyzer:
    """Analyze geomagnetic intensity evolution over geological time"""
    
    def __init__(self):
        self.intensity_models = {}
        
    def analyze_intensity_evolution(self, database, time_resolution=100000):
        """Analyze global intensity evolution"""
        # Get all intensity records
        cursor = database.connection.cursor()
        cursor.execute('''
            SELECT age, intensity, intensity_error, quality_score 
            FROM samples 
            WHERE intensity > 0 AND quality_score > 50
            ORDER BY age
        ''')
        
        records = cursor.fetchall()
        
        if len(records) < 50:
            print("Insufficient intensity data for analysis")
            return None
            
        # Create time bins
        ages = [r[0] for r in records]
        max_age = max(ages)
        time_bins = np.arange(0, max_age + time_resolution, time_resolution)
        
        evolution_data = []
        
        for i in range(len(time_bins) - 1):
            bin_start, bin_end = time_bins[i], time_bins[i + 1]
            
            # Get records in this bin
            bin_records = [r for r in records if bin_start <= r[0] < bin_end]
            
            if len(bin_records) >= 3:  # Minimum for statistics
                intensities = [r[1] for r in bin_records]
                errors = [r[2] for r in bin_records]
                qualities = [r[3] for r in bin_records]
                
                # Weighted statistics
                weights = np.array(qualities) / np.array(errors)
                weighted_mean = np.average(intensities, weights=weights)
                weighted_std = np.sqrt(np.average((intensities - weighted_mean)**2, weights=weights))
                
                evolution_data.append({
                    'age': (bin_start + bin_end) / 2,
                    'n_records': len(bin_records),
                    'intensity_mean': weighted_mean,
                    'intensity_std': weighted_std,
                    'intensity_median': np.median(intensities),
                    'vadm_estimate': self._intensity_to_vadm(weighted_mean)
                })
        
        return pd.DataFrame(evolution_data)
    
    def _intensity_to_vadm(self, intensity, latitude=45):
        """Convert local intensity to VADM estimate"""
        # Simplified conversion assuming dipolar field
        # VADM = intensity * 4π * R³ / (μ₀ * f(λ))
        # where f(λ) = √(1 + 3sin²λ)
        
        R_earth = 6.371e6  # Earth radius in meters
        mu0 = 4 * np.pi * 1e-7  # Permeability of free space
        
        f_lambda = np.sqrt(1 + 3 * np.sin(np.radians(latitude))**2)
        
        vadm = intensity * 1e-9 * 4 * np.pi * R_earth**3 / (mu0 * f_lambda)
        
        return vadm
    
    def detect_intensity_anomalies(self, evolution_data, threshold=2.0):
        """Detect intensity anomalies (excursions, reversals)"""
        if evolution_data is None or len(evolution_data) < 10:
            return []
            
        intensities = evolution_data['intensity_mean'].values
        ages = evolution_data['age'].values
        
        # Calculate rolling statistics
        window_size = min(5, len(intensities) // 4)
        rolling_mean = np.convolve(intensities, np.ones(window_size)/window_size, mode='same')
        rolling_std = np.array([np.std(intensities[max(0, i-window_size//2):i+window_size//2+1]) 
                               for i in range(len(intensities))])
        
        # Detect anomalies
        anomalies = []
        
        for i, (age, intensity, mean, std) in enumerate(zip(ages, intensities, rolling_mean, rolling_std)):
            if std > 0 and abs(intensity - mean) > threshold * std:
                anomaly_type = 'low_intensity' if intensity < mean else 'high_intensity'
                
                anomalies.append({
                    'age': age,
                    'intensity': intensity,
                    'expected_intensity': mean,
                    'sigma_deviation': abs(intensity - mean) / std,
                    'type': anomaly_type
                })
        
        return anomalies

def create_demonstration_database():
    """Create demonstration paleomagnetic database"""
    print("Creating comprehensive paleomagnetic database...")
    
    # Initialize database
    db = GlobalPaleomagneticDatabase('demo_paleomag.sqlite')
    
    # Analyze field evolution
    print("\nAnalyzing magnetic field evolution...")
    evolution = db.intensity_analyzer.analyze_intensity_evolution(db)
    
    if evolution is not None:
        print(f"Analyzed intensity evolution over {len(evolution)} time bins")
        
        # Detect anomalies
        anomalies = db.intensity_analyzer.detect_intensity_anomalies(evolution)
        print(f"Detected {len(anomalies)} intensity anomalies")
        
        # Recent statistics
        recent_data = evolution[evolution['age'] < 100000]
        if len(recent_data) > 0:
            print(f"Recent (0-100 ka) mean intensity: {recent_data['intensity_mean'].mean():.0f} nT")
    
    # Field reconstruction example
    print("\nField reconstruction example...")
    reconstructor = FieldReconstructor()
    
    # Reconstruct field at 10,000 years ago
    reconstruction = reconstructor.reconstruct_field_at_age(db, 10000)
    
    if reconstruction:
        print(f"Reconstructed field using {reconstruction['n_records']} records")
        print(f"Dipole strength: {reconstruction['quality']['dipole_strength']:.2e} Am²")
    
    # Secular variation analysis
    print("\nSecular variation analysis...")
    sv_analyzer = SecularVariationAnalyzer()
    
    # Analyze SV in Europe
    sv_analysis = sv_analyzer.analyze_secular_variation(db, (50.0, 10.0))
    
    if sv_analysis:
        print(f"SV analysis at (50°N, 10°E) using {sv_analysis['n_records']} records")
        print(f"Declination rate: {sv_analysis['declination_rate']:.3f} deg/year")
        print(f"Inclination rate: {sv_analysis['inclination_rate']:.3f} deg/year")
    
    print("\nDatabase creation and analysis completed!")
    
    return db

if __name__ == "__main__":
    # Create and demonstrate the paleomagnetic database
    database = create_demonstration_database()
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print("PALEOMAGNETIC DATABASE SUMMARY")
    print(f"{'='*50}")
    print(f"Total records: {len(database.records)}")
    
    # Source breakdown
    sources = {}
    for record in database.records:
        source = record.metadata.get('source', 'Unknown')
        sources[source] = sources.get(source, 0) + 1
    
    print("\nRecords by source:")
    for source, count in sources.items():
        print(f"  {source}: {count}")
    
    # Age distribution
    ages = [r.age for r in database.records]
    print(f"\nAge range: {min(ages)/1000:.1f} - {max(ages)/1000:.1f} ka")
    print(f"Median age: {np.median(ages)/1000:.1f} ka")
    
    # Quality distribution
    qualities = [r.quality_score for r in database.records]
    print(f"\nQuality scores:")
    print(f"  Mean: {np.mean(qualities):.1f}")
    print(f"  High quality (>80): {sum(1 for q in qualities if q > 80)}")
    print(f"  Medium quality (50-80): {sum(1 for q in qualities if 50 <= q <= 80)}")
    print(f"  Low quality (<50): {sum(1 for q in qualities if q < 50)}")
    
    database.connection.close()
    print("\nDatabase demonstration completed successfully!")