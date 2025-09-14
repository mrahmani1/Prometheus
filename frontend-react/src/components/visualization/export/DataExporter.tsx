import React, { useState, useCallback, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  FormControlLabel,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Grid,
  Chip,
  Alert,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  Download,
  Settings,
  ExpandMore,
  // Image as ImageIcon, // Unused
  PictureAsPdf,
  TableChart,
  Code,
  Map as MapIcon
} from '@mui/icons-material';
import { GeologicalData, SeismicEvent, VolcanicState } from '../../../types/geological';
// import * as turf from '@turf/turf'; // Temporarily disabled due to type issues

interface DataExporterProps {
  data: (GeologicalData | SeismicEvent | VolcanicState)[];
  dataType: 'geological' | 'seismic' | 'volcanic';
  onExport?: (exportedData: any, format: string, filename: string) => void;
  availableFormats?: ExportFormat[];
  includeMetadata?: boolean;
  customFields?: string[];
}

interface ExportFormat {
  id: string;
  name: string;
  extension: string;
  description: string;
  icon: React.ReactNode;
  supportsGeometry: boolean;
  supportsMetadata: boolean;
}

interface ExportOptions {
  format: string;
  filename: string;
  includeMetadata: boolean;
  includeGeometry: boolean;
  selectedFields: string[];
  coordinateSystem: string;
  compression: boolean;
  dateFormat: string;
  precision: number;
  customProjection?: string;
}

// Available export formats with geological domain expertise
const DEFAULT_EXPORT_FORMATS: ExportFormat[] = [
  {
    id: 'csv',
    name: 'CSV',
    extension: 'csv',
    description: 'Comma-separated values for spreadsheet applications',
    icon: <TableChart />,
    supportsGeometry: false,
    supportsMetadata: true
  },
  {
    id: 'geojson',
    name: 'GeoJSON',
    extension: 'geojson',
    description: 'Geographic JSON format with spatial data',
    icon: <MapIcon />,
    supportsGeometry: true,
    supportsMetadata: true
  },
  {
    id: 'kml',
    name: 'KML',
    extension: 'kml',
    description: 'Keyhole Markup Language for Google Earth',
    icon: <MapIcon />,
    supportsGeometry: true,
    supportsMetadata: true
  },
  {
    id: 'shapefile',
    name: 'Shapefile',
    extension: 'zip',
    description: 'ESRI Shapefile format (zipped)',
    icon: <MapIcon />,
    supportsGeometry: true,
    supportsMetadata: true
  },
  {
    id: 'json',
    name: 'JSON',
    extension: 'json',
    description: 'JavaScript Object Notation',
    icon: <Code />,
    supportsGeometry: true,
    supportsMetadata: true
  },
  {
    id: 'xml',
    name: 'XML',
    extension: 'xml',
    description: 'Extensible Markup Language',
    icon: <Code />,
    supportsGeometry: true,
    supportsMetadata: true
  },
  {
    id: 'excel',
    name: 'Excel',
    extension: 'xlsx',
    description: 'Microsoft Excel spreadsheet',
    icon: <TableChart />,
    supportsGeometry: false,
    supportsMetadata: true
  },
  {
    id: 'pdf',
    name: 'PDF Report',
    extension: 'pdf',
    description: 'Formatted geological report',
    icon: <PictureAsPdf />,
    supportsGeometry: true,
    supportsMetadata: true
  }
];

// Coordinate reference systems commonly used in geology
const COORDINATE_SYSTEMS = [
  { id: 'EPSG:4326', name: 'WGS84 (Geographic)', description: 'World Geodetic System 1984' },
  { id: 'EPSG:3857', name: 'Web Mercator', description: 'Web mapping standard' },
  { id: 'EPSG:32633', name: 'UTM Zone 33N', description: 'Universal Transverse Mercator' },
  { id: 'EPSG:2154', name: 'RGF93 / Lambert-93', description: 'French national grid' },
  { id: 'EPSG:3031', name: 'Antarctic Polar Stereographic', description: 'Antarctic mapping' },
  { id: 'EPSG:3413', name: 'NSIDC Sea Ice Polar Stereographic North', description: 'Arctic mapping' }
];

export const DataExporter: React.FC<DataExporterProps> = ({
  data,
  dataType,
  onExport,
  availableFormats = DEFAULT_EXPORT_FORMATS,
  includeMetadata = true,
  customFields = []
}) => {
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportOptions, setExportOptions] = useState<ExportOptions>({
    format: 'csv',
    filename: `${dataType}_data_${new Date().toISOString().split('T')[0]}`,
    includeMetadata: includeMetadata,
    includeGeometry: true,
    selectedFields: [],
    coordinateSystem: 'EPSG:4326',
    compression: false,
    dateFormat: 'ISO',
    precision: 6
  });
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [exportError, setExportError] = useState<string | null>(null);

  // Available fields from data
  const availableFields = useMemo(() => {
    if (!data.length) return [];
    
    const sample = data[0];
    const fields = Object.keys(sample).filter(key => {
      const value = sample[key as keyof typeof sample];
      return typeof value !== 'function' && key !== 'id';
    });
    
    return [...fields, ...customFields];
  }, [data, customFields]);

  // Initialize selected fields
  React.useEffect(() => {
    if (exportOptions.selectedFields.length === 0 && availableFields.length > 0) {
      setExportOptions(prev => ({
        ...prev,
        selectedFields: availableFields
      }));
    }
  }, [availableFields, exportOptions.selectedFields.length]);

  // Export data to CSV format
  const exportToCSV = useCallback((data: any[], options: ExportOptions): string => {
    const headers = options.selectedFields;
    const rows = data.map(item => {
      return headers.map(header => {
        let value = item[header];
        
        // Format dates
        if (value instanceof Date) {
          value = options.dateFormat === 'ISO' ? value.toISOString() : value.toLocaleString();
        }
        
        // Format coordinates with precision
        if ((header === 'latitude' || header === 'longitude') && typeof value === 'number') {
          value = value.toFixed(options.precision);
        }
        
        // Escape CSV special characters
        if (typeof value === 'string' && (value.includes(',') || value.includes('"') || value.includes('\n'))) {
          value = `"${value.replace(/"/g, '""')}"`;
        }
        
        return value || '';
      });
    });

    return [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
  }, []);

  // Export data to GeoJSON format
  const exportToGeoJSON = useCallback((data: any[], options: ExportOptions): any => {
    const features = data.map((item, index) => {
      const properties: any = {};
      
      // Include selected fields in properties
      options.selectedFields.forEach(field => {
        if (field !== 'latitude' && field !== 'longitude' && field !== 'geometry') {
          properties[field] = item[field];
        }
      });

      // Create geometry
      let geometry = null;
      if (options.includeGeometry) {
        if (item.latitude && item.longitude) {
          geometry = {
            type: 'Point',
            coordinates: [
              parseFloat(item.longitude.toFixed(options.precision)),
              parseFloat(item.latitude.toFixed(options.precision))
            ]
          };
        } else if (item.geometry) {
          geometry = item.geometry;
        }
      }

      return {
        type: 'Feature',
        id: item.id || index,
        geometry,
        properties
      };
    });

    const geoJSON = {
      type: 'FeatureCollection',
      features,
      crs: {
        type: 'name',
        properties: {
          name: options.coordinateSystem
        }
      }
    };

    // Add metadata if requested
    if (options.includeMetadata) {
      (geoJSON as any).metadata = {
        dataType,
        exportDate: new Date().toISOString(),
        recordCount: data.length,
        coordinateSystem: options.coordinateSystem,
        precision: options.precision
      };
    }

    return geoJSON;
  }, [dataType]);

  // Export data to KML format
  const exportToKML = useCallback((data: any[], options: ExportOptions): string => {
    const kmlHeader = `<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>${dataType.charAt(0).toUpperCase() + dataType.slice(1)} Data</name>
    <description>Exported geological data</description>`;

    const kmlFooter = `  </Document>
</kml>`;

    const placemarks = data.map((item, index) => {
      if (!item.latitude || !item.longitude) return '';

      const name = item.name || item.id || `${dataType}_${index}`;
      const description = options.selectedFields
        .map(field => `${field}: ${item[field] || 'N/A'}`)
        .join('<br/>');

      return `    <Placemark>
      <name>${name}</name>
      <description><![CDATA[${description}]]></description>
      <Point>
        <coordinates>${item.longitude.toFixed(options.precision)},${item.latitude.toFixed(options.precision)},${item.elevation || 0}</coordinates>
      </Point>
    </Placemark>`;
    }).filter(Boolean);

    return kmlHeader + '\n' + placemarks.join('\n') + '\n' + kmlFooter;
  }, [dataType]);

  // Export data to XML format
  const exportToXML = useCallback((data: any[], options: ExportOptions): string => {
    const xmlHeader = `<?xml version="1.0" encoding="UTF-8"?>
<geologicalData type="${dataType}" exportDate="${new Date().toISOString()}">`;

    const xmlFooter = `</geologicalData>`;

    const records = data.map((item, index) => {
      const fields = options.selectedFields
        .map(field => {
          const value = item[field];
          if (value === null || value === undefined) return '';
          
          // Escape XML special characters
          const escapedValue = String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&apos;');
          
          return `    <${field}>${escapedValue}</${field}>`;
        })
        .filter(Boolean)
        .join('\n');

      return `  <record id="${item.id || index}">
${fields}
  </record>`;
    });

    return xmlHeader + '\n' + records.join('\n') + '\n' + xmlFooter;
  }, [dataType]);

  // Main export function
  const performExport = useCallback(async () => {
    setIsExporting(true);
    setExportProgress(0);
    setExportError(null);

    try {
      // Filter data based on selected fields
      const filteredData = data.map(item => {
        const filtered: any = {};
        exportOptions.selectedFields.forEach(field => {
          filtered[field] = item[field as keyof typeof item];
        });
        return { ...filtered, id: item.id };
      });

      setExportProgress(25);

      let exportedData: any;
      let mimeType: string;

      switch (exportOptions.format) {
        case 'csv':
          exportedData = exportToCSV(filteredData, exportOptions);
          mimeType = 'text/csv';
          break;

        case 'geojson':
          exportedData = JSON.stringify(exportToGeoJSON(filteredData, exportOptions), null, 2);
          mimeType = 'application/geo+json';
          break;

        case 'kml':
          exportedData = exportToKML(filteredData, exportOptions);
          mimeType = 'application/vnd.google-earth.kml+xml';
          break;

        case 'json':
          exportedData = JSON.stringify(filteredData, null, 2);
          mimeType = 'application/json';
          break;

        case 'xml':
          exportedData = exportToXML(filteredData, exportOptions);
          mimeType = 'application/xml';
          break;

        default:
          throw new Error(`Unsupported export format: ${exportOptions.format}`);
      }

      setExportProgress(75);

      // Create and download file
      const format = availableFormats.find(f => f.id === exportOptions.format);
      const filename = `${exportOptions.filename}.${format?.extension || exportOptions.format}`;
      
      const blob = new Blob([exportedData], { type: mimeType });
      const url = URL.createObjectURL(blob);
      
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setExportProgress(100);

      // Callback for custom handling
      onExport?.(exportedData, exportOptions.format, filename);

      // Close dialog after successful export
      setTimeout(() => {
        setExportDialogOpen(false);
        setIsExporting(false);
        setExportProgress(0);
      }, 1000);

    } catch (error) {
      setExportError(error instanceof Error ? error.message : 'Export failed');
      setIsExporting(false);
      setExportProgress(0);
    }
  }, [data, exportOptions, availableFormats, onExport, exportToCSV, exportToGeoJSON, exportToKML, exportToXML]);

  // Quick export functions
  const quickExport = useCallback((format: string) => {
    setExportOptions(prev => ({ ...prev, format }));
    performExport();
  }, [performExport]);

  const selectedFormat = availableFormats.find(f => f.id === exportOptions.format);

  return (
    <Box>
      {/* Quick Export Buttons */}
      <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
        <Button
          startIcon={<TableChart />}
          onClick={() => quickExport('csv')}
          variant="outlined"
          size="small"
        >
          CSV
        </Button>
        <Button
          startIcon={<MapIcon />}
          onClick={() => quickExport('geojson')}
          variant="outlined"
          size="small"
        >
          GeoJSON
        </Button>
        <Button
          startIcon={<Code />}
          onClick={() => quickExport('json')}
          variant="outlined"
          size="small"
        >
          JSON
        </Button>
        <Button
          startIcon={<Settings />}
          onClick={() => setExportDialogOpen(true)}
          variant="contained"
          size="small"
        >
          Advanced Export
        </Button>
      </Box>

      {/* Advanced Export Dialog */}
      <Dialog
        open={exportDialogOpen}
        onClose={() => setExportDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Export {dataType.charAt(0).toUpperCase() + dataType.slice(1)} Data
        </DialogTitle>
        
        <DialogContent>
          <Grid container spacing={3} sx={{ mt: 1 }}>
            {/* Format Selection */}
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Export Format</InputLabel>
                <Select
                  value={exportOptions.format}
                  onChange={(e) => setExportOptions(prev => ({ ...prev, format: e.target.value }))}
                >
                  {availableFormats.map(format => (
                    <MenuItem key={format.id} value={format.id}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {format.icon}
                        {format.name}
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              {selectedFormat && (
                <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
                  {selectedFormat.description}
                </Typography>
              )}
            </Grid>

            {/* Filename */}
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Filename"
                value={exportOptions.filename}
                onChange={(e) => setExportOptions(prev => ({ ...prev, filename: e.target.value }))}
              />
            </Grid>

            {/* Options */}
            <Grid item xs={12}>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography>Export Options</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={exportOptions.includeMetadata}
                            onChange={(e) => setExportOptions(prev => ({ ...prev, includeMetadata: e.target.checked }))}
                          />
                        }
                        label="Include Metadata"
                      />
                    </Grid>
                    
                    {selectedFormat?.supportsGeometry && (
                      <Grid item xs={12} md={6}>
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={exportOptions.includeGeometry}
                              onChange={(e) => setExportOptions(prev => ({ ...prev, includeGeometry: e.target.checked }))}
                            />
                          }
                          label="Include Geometry"
                        />
                      </Grid>
                    )}

                    <Grid item xs={12} md={6}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Coordinate System</InputLabel>
                        <Select
                          value={exportOptions.coordinateSystem}
                          onChange={(e) => setExportOptions(prev => ({ ...prev, coordinateSystem: e.target.value }))}
                        >
                          {COORDINATE_SYSTEMS.map(crs => (
                            <MenuItem key={crs.id} value={crs.id}>
                              {crs.name}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>

                    <Grid item xs={12} md={6}>
                      <TextField
                        fullWidth
                        size="small"
                        label="Coordinate Precision"
                        type="number"
                        value={exportOptions.precision}
                        onChange={(e) => setExportOptions(prev => ({ ...prev, precision: parseInt(e.target.value) }))}
                        inputProps={{ min: 1, max: 15 }}
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </Grid>

            {/* Field Selection */}
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>
                Select Fields to Export ({exportOptions.selectedFields.length} of {availableFields.length})
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, maxHeight: 200, overflow: 'auto' }}>
                {availableFields.map(field => (
                  <Chip
                    key={field}
                    label={field}
                    clickable
                    color={exportOptions.selectedFields.includes(field) ? 'primary' : 'default'}
                    onClick={() => {
                      setExportOptions(prev => ({
                        ...prev,
                        selectedFields: prev.selectedFields.includes(field)
                          ? prev.selectedFields.filter(f => f !== field)
                          : [...prev.selectedFields, field]
                      }));
                    }}
                  />
                ))}
              </Box>
            </Grid>

            {/* Export Progress */}
            {isExporting && (
              <Grid item xs={12}>
                <Typography variant="body2" gutterBottom>
                  Exporting... {exportProgress}%
                </Typography>
                <LinearProgress variant="determinate" value={exportProgress} />
              </Grid>
            )}

            {/* Export Error */}
            {exportError && (
              <Grid item xs={12}>
                <Alert severity="error">{exportError}</Alert>
              </Grid>
            )}

            {/* Export Summary */}
            <Grid item xs={12}>
              <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                <Typography variant="subtitle2" gutterBottom>
                  Export Summary
                </Typography>
                <Typography variant="body2">
                  Records: {data.length} | Format: {selectedFormat?.name} | 
                  Fields: {exportOptions.selectedFields.length} | 
                  Coordinate System: {exportOptions.coordinateSystem}
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </DialogContent>

        <DialogActions>
          <Button onClick={() => setExportDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={performExport}
            variant="contained"
            startIcon={<Download />}
            disabled={isExporting || exportOptions.selectedFields.length === 0}
          >
            {isExporting ? 'Exporting...' : 'Export'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DataExporter;