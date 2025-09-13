"""
Export capabilities for standardized geological data formats
"""

import json
import csv
import h5py
import netCDF4 as nc
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, IO
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml

from .schemas import StandardizedDataFormat, DataType, CoordinateSystem


logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for data export"""
    format: str  # 'netcdf', 'hdf5', 'csv', 'json', 'geojson', 'xml', 'yaml'
    output_path: str
    include_metadata: bool = True
    include_processing_history: bool = True
    include_quality_metrics: bool = True
    include_uncertainty: bool = True
    compression: Optional[str] = None  # 'gzip', 'lzf', 'szip'
    compression_level: int = 6
    chunk_size: Optional[int] = None
    coordinate_format: str = 'decimal'  # 'decimal', 'dms'
    time_format: str = 'iso'  # 'iso', 'unix', 'matlab'
    precision: int = 6  # decimal places for floating point numbers


@dataclass
class ExportResult:
    """Result of data export operation"""
    success: bool
    output_files: List[str]
    format_used: str
    records_exported: int
    file_size_bytes: int
    export_time_seconds: float
    warnings: List[str]
    errors: List[str]


class ExportManager:
    """
    Manages export of standardized geological data to various formats
    """
    
    def __init__(self):
        self.supported_formats = {
            'netcdf': self._export_netcdf,
            'hdf5': self._export_hdf5,
            'csv': self._export_csv,
            'json': self._export_json,
            'geojson': self._export_geojson,
            'xml': self._export_xml,
            'yaml': self._export_yaml,
            'parquet': self._export_parquet,
            'excel': self._export_excel,
        }
        
        self.format_extensions = {
            'netcdf': '.nc',
            'hdf5': '.h5',
            'csv': '.csv',
            'json': '.json',
            'geojson': '.geojson',
            'xml': '.xml',
            'yaml': '.yaml',
            'parquet': '.parquet',
            'excel': '.xlsx',
        }
    
    def export_data(
        self,
        data_list: List[StandardizedDataFormat],
        config: ExportConfig
    ) -> ExportResult:
        """
        Export geological data to specified format
        
        Args:
            data_list: Data to export
            config: Export configuration
            
        Returns:
            Export result with status and metadata
        """
        start_time = datetime.now()
        warnings = []
        errors = []
        
        try:
            # Validate format
            if config.format not in self.supported_formats:
                raise ValueError(f"Unsupported export format: {config.format}")
            
            # Ensure output directory exists
            output_path = Path(config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add extension if not present
            if not output_path.suffix:
                extension = self.format_extensions.get(config.format, '')
                output_path = output_path.with_suffix(extension)
            
            # Export data using appropriate method
            export_func = self.supported_formats[config.format]
            output_files = export_func(data_list, str(output_path), config, warnings, errors)
            
            # Calculate file sizes
            total_size = sum(os.path.getsize(f) for f in output_files if os.path.exists(f))
            
            # Calculate export time
            export_time = (datetime.now() - start_time).total_seconds()
            
            return ExportResult(
                success=len(errors) == 0,
                output_files=output_files,
                format_used=config.format,
                records_exported=len(data_list),
                file_size_bytes=total_size,
                export_time_seconds=export_time,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            export_time = (datetime.now() - start_time).total_seconds()
            
            return ExportResult(
                success=False,
                output_files=[],
                format_used=config.format,
                records_exported=0,
                file_size_bytes=0,
                export_time_seconds=export_time,
                warnings=warnings,
                errors=[str(e)]
            )
    
    def _export_netcdf(
        self,
        data_list: List[StandardizedDataFormat],
        output_path: str,
        config: ExportConfig,
        warnings: List[str],
        errors: List[str]
    ) -> List[str]:
        """Export data to NetCDF format"""
        try:
            with nc.Dataset(output_path, 'w', format='NETCDF4') as dataset:
                # Create dimensions
                n_records = len(data_list)
                dataset.createDimension('time', n_records)
                dataset.createDimension('string_length', 256)
                
                # Create coordinate variables
                times = dataset.createVariable('time', 'f8', ('time',))
                times.units = 'seconds since 1970-01-01 00:00:00'
                times.calendar = 'gregorian'
                times.long_name = 'time'
                
                latitudes = dataset.createVariable('latitude', 'f8', ('time',))
                latitudes.units = 'degrees_north'
                latitudes.long_name = 'latitude'
                
                longitudes = dataset.createVariable('longitude', 'f8', ('time',))
                longitudes.units = 'degrees_east'
                longitudes.long_name = 'longitude'
                
                elevations = dataset.createVariable('elevation', 'f8', ('time',))
                elevations.units = 'meters'
                elevations.long_name = 'elevation above sea level'
                
                # Collect all unique value keys
                all_keys = set()
                for data in data_list:
                    all_keys.update(data.values.keys())
                
                # Create data variables
                data_vars = {}
                for key in all_keys:
                    # Determine data type and create variable
                    sample_values = [d.values.get(key) for d in data_list if key in d.values]
                    if sample_values:
                        sample_value = next(v for v in sample_values if v is not None)
                        
                        if isinstance(sample_value, (int, float)):
                            var = dataset.createVariable(key, 'f8', ('time',), 
                                                       compression=config.compression,
                                                       complevel=config.compression_level)
                            data_vars[key] = var
                            
                            # Add units if available
                            if data_list[0].units and key in data_list[0].units:
                                var.units = data_list[0].units[key]
                
                # Create uncertainty variables if requested
                uncertainty_vars = {}
                if config.include_uncertainty:
                    for key in all_keys:
                        unc_var = dataset.createVariable(f'{key}_uncertainty', 'f8', ('time',),
                                                       compression=config.compression,
                                                       complevel=config.compression_level)
                        uncertainty_vars[key] = unc_var
                
                # Fill data
                for i, data in enumerate(data_list):
                    # Time coordinate
                    times[i] = (data.timestamp - datetime(1970, 1, 1)).total_seconds()
                    
                    # Spatial coordinates
                    latitudes[i] = data.location.get('y', 0.0)
                    longitudes[i] = data.location.get('x', 0.0)
                    elevations[i] = data.location.get('z', 0.0)
                    
                    # Data values
                    for key, var in data_vars.items():
                        if key in data.values and isinstance(data.values[key], (int, float)):
                            var[i] = data.values[key]
                        else:
                            var[i] = np.nan
                    
                    # Uncertainty values
                    if config.include_uncertainty:
                        for key, var in uncertainty_vars.items():
                            if (data.value_uncertainty and key in data.value_uncertainty):
                                var[i] = data.value_uncertainty[key]
                            else:
                                var[i] = np.nan
                
                # Add global attributes
                if config.include_metadata:
                    dataset.title = f"Geological Data Export - {data_list[0].data_type.value}"
                    dataset.institution = "Prometheus AI Platform"
                    dataset.source = "Real geological data integration"
                    dataset.history = f"Created on {datetime.now().isoformat()}"
                    dataset.conventions = "CF-1.8"
                    
                    # Add data type specific metadata
                    if data_list:
                        dataset.data_type = data_list[0].data_type.value
                        dataset.coordinate_system = data_list[0].coordinate_system.value
                        if data_list[0].source:
                            dataset.data_source = data_list[0].source
            
            return [output_path]
            
        except Exception as e:
            errors.append(f"NetCDF export failed: {e}")
            return []
    
    def _export_hdf5(
        self,
        data_list: List[StandardizedDataFormat],
        output_path: str,
        config: ExportConfig,
        warnings: List[str],
        errors: List[str]
    ) -> List[str]:
        """Export data to HDF5 format"""
        try:
            with h5py.File(output_path, 'w') as f:
                # Create groups
                data_group = f.create_group('data')
                metadata_group = f.create_group('metadata')
                
                n_records = len(data_list)
                
                # Create coordinate datasets
                times = data_group.create_dataset('time', (n_records,), dtype='f8',
                                                compression=config.compression)
                latitudes = data_group.create_dataset('latitude', (n_records,), dtype='f8',
                                                    compression=config.compression)
                longitudes = data_group.create_dataset('longitude', (n_records,), dtype='f8',
                                                     compression=config.compression)
                elevations = data_group.create_dataset('elevation', (n_records,), dtype='f8',
                                                     compression=config.compression)
                
                # Add attributes
                times.attrs['units'] = 'seconds since 1970-01-01 00:00:00'
                latitudes.attrs['units'] = 'degrees_north'
                longitudes.attrs['units'] = 'degrees_east'
                elevations.attrs['units'] = 'meters'
                
                # Collect all unique value keys
                all_keys = set()
                for data in data_list:
                    all_keys.update(data.values.keys())
                
                # Create data datasets
                data_datasets = {}
                for key in all_keys:
                    dataset = data_group.create_dataset(key, (n_records,), dtype='f8',
                                                      compression=config.compression)
                    data_datasets[key] = dataset
                    
                    # Add units
                    if data_list[0].units and key in data_list[0].units:
                        dataset.attrs['units'] = data_list[0].units[key]
                
                # Create uncertainty datasets if requested
                if config.include_uncertainty:
                    uncertainty_group = f.create_group('uncertainty')
                    for key in all_keys:
                        unc_dataset = uncertainty_group.create_dataset(key, (n_records,), dtype='f8',
                                                                     compression=config.compression)
                
                # Fill data
                for i, data in enumerate(data_list):
                    times[i] = (data.timestamp - datetime(1970, 1, 1)).total_seconds()
                    latitudes[i] = data.location.get('y', 0.0)
                    longitudes[i] = data.location.get('x', 0.0)
                    elevations[i] = data.location.get('z', 0.0)
                    
                    for key, dataset in data_datasets.items():
                        if key in data.values and isinstance(data.values[key], (int, float)):
                            dataset[i] = data.values[key]
                        else:
                            dataset[i] = np.nan
                    
                    if config.include_uncertainty and 'uncertainty' in f:
                        for key in all_keys:
                            if (data.value_uncertainty and key in data.value_uncertainty):
                                f['uncertainty'][key][i] = data.value_uncertainty[key]
                            else:
                                f['uncertainty'][key][i] = np.nan
                
                # Add metadata
                if config.include_metadata and data_list:
                    metadata_group.attrs['data_type'] = data_list[0].data_type.value
                    metadata_group.attrs['coordinate_system'] = data_list[0].coordinate_system.value
                    metadata_group.attrs['export_time'] = datetime.now().isoformat()
                    metadata_group.attrs['n_records'] = n_records
            
            return [output_path]
            
        except Exception as e:
            errors.append(f"HDF5 export failed: {e}")
            return []
    
    def _export_csv(
        self,
        data_list: List[StandardizedDataFormat],
        output_path: str,
        config: ExportConfig,
        warnings: List[str],
        errors: List[str]
    ) -> List[str]:
        """Export data to CSV format"""
        try:
            # Prepare data for DataFrame
            records = []
            
            for data in data_list:
                record = {
                    'data_id': data.data_id,
                    'timestamp': data.timestamp.isoformat() if config.time_format == 'iso' 
                               else data.timestamp.timestamp(),
                    'latitude': data.location.get('y', 0.0),
                    'longitude': data.location.get('x', 0.0),
                    'elevation': data.location.get('z', 0.0),
                    'data_type': data.data_type.value,
                    'source': data.source,
                }
                
                # Add data values
                for key, value in data.values.items():
                    if isinstance(value, (int, float)):
                        record[key] = round(value, config.precision)
                    else:
                        record[key] = str(value)
                
                # Add units
                if config.include_metadata:
                    for key, unit in data.units.items():
                        record[f'{key}_unit'] = unit
                
                # Add uncertainty
                if config.include_uncertainty and data.value_uncertainty:
                    for key, uncertainty in data.value_uncertainty.items():
                        record[f'{key}_uncertainty'] = round(uncertainty, config.precision)
                
                # Add quality metrics
                if config.include_quality_metrics and data.quality_metrics:
                    record['quality_overall'] = round(data.quality_metrics.overall_score, config.precision)
                    record['quality_completeness'] = round(data.quality_metrics.completeness, config.precision)
                    record['quality_accuracy'] = round(data.quality_metrics.accuracy, config.precision)
                
                records.append(record)
            
            # Create DataFrame and export
            df = pd.DataFrame(records)
            df.to_csv(output_path, index=False)
            
            return [output_path]
            
        except Exception as e:
            errors.append(f"CSV export failed: {e}")
            return []
    
    def _export_json(
        self,
        data_list: List[StandardizedDataFormat],
        output_path: str,
        config: ExportConfig,
        warnings: List[str],
        errors: List[str]
    ) -> List[str]:
        """Export data to JSON format"""
        try:
            # Convert data to JSON-serializable format
            json_data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'format_version': '1.0',
                    'record_count': len(data_list),
                    'coordinate_system': data_list[0].coordinate_system.value if data_list else None,
                },
                'data': []
            }
            
            for data in data_list:
                record = {
                    'data_id': data.data_id,
                    'timestamp': data.timestamp.isoformat(),
                    'data_type': data.data_type.value,
                    'location': data.location,
                    'coordinate_system': data.coordinate_system.value,
                    'values': data.values,
                    'units': data.units,
                    'source': data.source,
                }
                
                if config.include_uncertainty and data.value_uncertainty:
                    record['value_uncertainty'] = data.value_uncertainty
                
                if config.include_quality_metrics and data.quality_metrics:
                    record['quality_metrics'] = asdict(data.quality_metrics)
                
                if config.include_processing_history:
                    record['processing_history'] = data.processing_history
                
                if config.include_metadata:
                    record['metadata'] = data.metadata
                
                json_data['data'].append(record)
            
            # Write JSON file
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            return [output_path]
            
        except Exception as e:
            errors.append(f"JSON export failed: {e}")
            return []
    
    def _export_geojson(
        self,
        data_list: List[StandardizedDataFormat],
        output_path: str,
        config: ExportConfig,
        warnings: List[str],
        errors: List[str]
    ) -> List[str]:
        """Export data to GeoJSON format"""
        try:
            geojson = {
                'type': 'FeatureCollection',
                'features': []
            }
            
            for data in data_list:
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [
                            data.location.get('x', 0.0),
                            data.location.get('y', 0.0),
                            data.location.get('z', 0.0)
                        ]
                    },
                    'properties': {
                        'data_id': data.data_id,
                        'timestamp': data.timestamp.isoformat(),
                        'data_type': data.data_type.value,
                        'source': data.source,
                    }
                }
                
                # Add data values to properties
                for key, value in data.values.items():
                    if isinstance(value, (int, float)):
                        feature['properties'][key] = round(value, config.precision)
                    else:
                        feature['properties'][key] = str(value)
                
                # Add units
                if config.include_metadata:
                    feature['properties']['units'] = data.units
                
                geojson['features'].append(feature)
            
            # Write GeoJSON file
            with open(output_path, 'w') as f:
                json.dump(geojson, f, indent=2)
            
            return [output_path]
            
        except Exception as e:
            errors.append(f"GeoJSON export failed: {e}")
            return []
    
    def _export_xml(
        self,
        data_list: List[StandardizedDataFormat],
        output_path: str,
        config: ExportConfig,
        warnings: List[str],
        errors: List[str]
    ) -> List[str]:
        """Export data to XML format"""
        try:
            root = ET.Element('geological_data')
            root.set('export_time', datetime.now().isoformat())
            root.set('record_count', str(len(data_list)))
            
            for data in data_list:
                record_elem = ET.SubElement(root, 'record')
                record_elem.set('data_id', data.data_id)
                
                # Basic information
                ET.SubElement(record_elem, 'timestamp').text = data.timestamp.isoformat()
                ET.SubElement(record_elem, 'data_type').text = data.data_type.value
                ET.SubElement(record_elem, 'source').text = data.source
                
                # Location
                location_elem = ET.SubElement(record_elem, 'location')
                location_elem.set('coordinate_system', data.coordinate_system.value)
                ET.SubElement(location_elem, 'longitude').text = str(data.location.get('x', 0.0))
                ET.SubElement(location_elem, 'latitude').text = str(data.location.get('y', 0.0))
                ET.SubElement(location_elem, 'elevation').text = str(data.location.get('z', 0.0))
                
                # Values
                values_elem = ET.SubElement(record_elem, 'values')
                for key, value in data.values.items():
                    value_elem = ET.SubElement(values_elem, 'value')
                    value_elem.set('name', key)
                    if key in data.units:
                        value_elem.set('unit', data.units[key])
                    value_elem.text = str(value)
            
            # Write XML file
            tree = ET.ElementTree(root)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
            return [output_path]
            
        except Exception as e:
            errors.append(f"XML export failed: {e}")
            return []
    
    def _export_yaml(
        self,
        data_list: List[StandardizedDataFormat],
        output_path: str,
        config: ExportConfig,
        warnings: List[str],
        errors: List[str]
    ) -> List[str]:
        """Export data to YAML format"""
        try:
            # Convert to YAML-serializable format
            yaml_data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'record_count': len(data_list),
                },
                'records': []
            }
            
            for data in data_list:
                record = {
                    'data_id': data.data_id,
                    'timestamp': data.timestamp.isoformat(),
                    'data_type': data.data_type.value,
                    'location': data.location,
                    'values': data.values,
                    'units': data.units,
                    'source': data.source,
                }
                yaml_data['records'].append(record)
            
            # Write YAML file
            with open(output_path, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False)
            
            return [output_path]
            
        except Exception as e:
            errors.append(f"YAML export failed: {e}")
            return []
    
    def _export_parquet(
        self,
        data_list: List[StandardizedDataFormat],
        output_path: str,
        config: ExportConfig,
        warnings: List[str],
        errors: List[str]
    ) -> List[str]:
        """Export data to Parquet format"""
        try:
            # Prepare data for DataFrame (similar to CSV)
            records = []
            
            for data in data_list:
                record = {
                    'data_id': data.data_id,
                    'timestamp': data.timestamp,
                    'latitude': data.location.get('y', 0.0),
                    'longitude': data.location.get('x', 0.0),
                    'elevation': data.location.get('z', 0.0),
                    'data_type': data.data_type.value,
                    'source': data.source,
                }
                
                # Add data values
                for key, value in data.values.items():
                    if isinstance(value, (int, float)):
                        record[key] = value
                
                records.append(record)
            
            # Create DataFrame and export
            df = pd.DataFrame(records)
            df.to_parquet(output_path, compression=config.compression)
            
            return [output_path]
            
        except Exception as e:
            errors.append(f"Parquet export failed: {e}")
            return []
    
    def _export_excel(
        self,
        data_list: List[StandardizedDataFormat],
        output_path: str,
        config: ExportConfig,
        warnings: List[str],
        errors: List[str]
    ) -> List[str]:
        """Export data to Excel format"""
        try:
            # Prepare data for DataFrame
            records = []
            
            for data in data_list:
                record = {
                    'data_id': data.data_id,
                    'timestamp': data.timestamp.isoformat(),
                    'latitude': data.location.get('y', 0.0),
                    'longitude': data.location.get('x', 0.0),
                    'elevation': data.location.get('z', 0.0),
                    'data_type': data.data_type.value,
                    'source': data.source,
                }
                
                # Add data values
                for key, value in data.values.items():
                    if isinstance(value, (int, float)):
                        record[key] = value
                    else:
                        record[key] = str(value)
                
                records.append(record)
            
            # Create DataFrame and export
            df = pd.DataFrame(records)
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)
                
                # Add metadata sheet if requested
                if config.include_metadata and data_list:
                    metadata_df = pd.DataFrame([{
                        'export_time': datetime.now().isoformat(),
                        'record_count': len(data_list),
                        'data_type': data_list[0].data_type.value,
                        'coordinate_system': data_list[0].coordinate_system.value,
                    }])
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            return [output_path]
            
        except Exception as e:
            errors.append(f"Excel export failed: {e}")
            return []
    
    def batch_export(
        self,
        data_groups: Dict[str, List[StandardizedDataFormat]],
        base_config: ExportConfig
    ) -> Dict[str, ExportResult]:
        """
        Export multiple groups of data
        
        Args:
            data_groups: Dictionary mapping group names to data lists
            base_config: Base export configuration
            
        Returns:
            Dictionary mapping group names to export results
        """
        results = {}
        
        for group_name, data_list in data_groups.items():
            # Create group-specific config
            group_config = ExportConfig(
                format=base_config.format,
                output_path=f"{base_config.output_path}_{group_name}",
                include_metadata=base_config.include_metadata,
                include_processing_history=base_config.include_processing_history,
                include_quality_metrics=base_config.include_quality_metrics,
                include_uncertainty=base_config.include_uncertainty,
                compression=base_config.compression,
                compression_level=base_config.compression_level,
                coordinate_format=base_config.coordinate_format,
                time_format=base_config.time_format,
                precision=base_config.precision
            )
            
            try:
                result = self.export_data(data_list, group_config)
                results[group_name] = result
                logger.info(f"Exported {len(data_list)} records for group {group_name}")
            except Exception as e:
                logger.error(f"Failed to export group {group_name}: {e}")
                results[group_name] = ExportResult(
                    success=False,
                    output_files=[],
                    format_used=base_config.format,
                    records_exported=0,
                    file_size_bytes=0,
                    export_time_seconds=0.0,
                    warnings=[],
                    errors=[str(e)]
                )
        
        return results