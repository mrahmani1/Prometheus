import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import {
  DataGrid,
  GridColDef,
  GridRowsProp,
  GridToolbar,
  GridFilterModel,
  GridSortModel,
  GridRowSelectionModel,
  GridRenderCellParams,
  GridValueFormatterParams,
  GridValueGetterParams
} from '@mui/x-data-grid';
import {
  Box,
  Paper,
  Typography,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  // FormControl, // Unused
  // InputLabel, // Unused
  // Select, // Unused
  // Switch, // Unused
  // FormControlLabel, // Unused
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Alert
} from '@mui/material';
import {
  MoreVert,
  Download,
  // Refresh, // Unused
  // FilterList, // Unused
  Visibility,
  Edit,
  Delete,
  Map as MapIcon,
  Timeline,
  Assessment
} from '@mui/icons-material';
import { GeologicalData, SeismicEvent, VolcanicState } from '../../../types/geological';
import { format, parseISO } from 'date-fns';

interface GeologicalDataGridProps {
  data: (GeologicalData | SeismicEvent | VolcanicState)[];
  dataType: 'geological' | 'seismic' | 'volcanic';
  onRowSelect?: (selectedRows: any[]) => void;
  onRowEdit?: (row: any) => void;
  onRowDelete?: (row: any) => void;
  onExport?: (data: any[], format: string) => void;
  onVisualize?: (data: any[], type: string) => void;
  realTimeUpdate?: boolean;
  enableVirtualization?: boolean;
  customColumns?: GridColDef[];
  height?: number;
}

// Advanced geological data type formatters
const formatters = {
  coordinate: (value: number) => value?.toFixed(6) || 'N/A',
  magnitude: (value: number) => value?.toFixed(2) || 'N/A',
  depth: (value: number) => `${value?.toFixed(1) || 'N/A'} km`,
  temperature: (value: number) => `${value?.toFixed(1) || 'N/A'}°C`,
  pressure: (value: number) => `${value?.toFixed(2) || 'N/A'} MPa`,
  percentage: (value: number) => `${(value * 100)?.toFixed(1) || 'N/A'}%`,
  datetime: (value: string | Date) => {
    try {
      const date = typeof value === 'string' ? parseISO(value) : value;
      return format(date, 'yyyy-MM-dd HH:mm:ss');
    } catch {
      return 'Invalid Date';
    }
  }
};

// Geological data type validators
const validators = {
  coordinate: (value: number) => value >= -180 && value <= 180,
  latitude: (value: number) => value >= -90 && value <= 90,
  longitude: (value: number) => value >= -180 && value <= 180,
  magnitude: (value: number) => value >= 0 && value <= 10,
  depth: (value: number) => value >= 0 && value <= 1000,
  percentage: (value: number) => value >= 0 && value <= 1
};

// Custom cell renderers for geological data
const GeologicalCellRenderer: React.FC<GridRenderCellParams> = ({ value, field }) => {
  switch (field) {
    case 'alert_level':
      const alertColors = {
        normal: 'success',
        advisory: 'warning',
        watch: 'error',
        warning: 'error',
        critical: 'error'
      };
      return (
        <Chip
          label={value}
          color={alertColors[value as keyof typeof alertColors] as any}
          size="small"
          variant="filled"
        />
      );

    case 'data_type':
    case 'event_type':
      const typeColors = {
        igneous: '#FF6B6B',
        sedimentary: '#4ECDC4',
        metamorphic: '#45B7D1',
        volcanic: '#FF8E53',
        seismic: '#FF4757',
        tectonic: '#2ED573'
      };
      return (
        <Chip
          label={value}
          size="small"
          style={{ backgroundColor: typeColors[value as keyof typeof typeColors] || '#888888' }}
        />
      );

    case 'quality_score':
    case 'confidence':
      const score = parseFloat(value) * 100;
      const color = score > 80 ? 'success' : score > 60 ? 'warning' : 'error';
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Chip label={`${score.toFixed(1)}%`} color={color} size="small" />
        </Box>
      );

    case 'location':
      if (value && typeof value === 'object') {
        return (
          <Box sx={{ fontSize: '0.875rem' }}>
            {formatters.coordinate(value.latitude)}, {formatters.coordinate(value.longitude)}
          </Box>
        );
      }
      return <Typography variant="body2">N/A</Typography>;

    default:
      return <Typography variant="body2">{value}</Typography>;
  }
};

// Generate column definitions based on data type
const generateColumns = (dataType: string, customColumns?: GridColDef[]): GridColDef[] => {
  if (customColumns) return customColumns;

  const baseColumns: GridColDef[] = [
    {
      field: 'id',
      headerName: 'ID',
      width: 120
    },
    {
      field: 'timestamp',
      headerName: 'Timestamp',
      width: 180,
      valueFormatter: (params: GridValueFormatterParams) => formatters.datetime(params.value),
      type: 'dateTime'
    }
  ];

  switch (dataType) {
    case 'geological':
      return [
        ...baseColumns,
        {
          field: 'data_type',
          headerName: 'Type',
          width: 120,
          renderCell: GeologicalCellRenderer
        },
        {
          field: 'latitude',
          headerName: 'Latitude',
          width: 120,
          type: 'number',
          valueFormatter: (params: GridValueFormatterParams) => formatters.coordinate(params.value)
        },
        {
          field: 'longitude',
          headerName: 'Longitude',
          width: 120,
          type: 'number',
          valueFormatter: (params: GridValueFormatterParams) => formatters.coordinate(params.value)
        },
        {
          field: 'elevation',
          headerName: 'Elevation (m)',
          width: 120,
          type: 'number'
        },
        {
          field: 'quality_score',
          headerName: 'Quality',
          width: 100,
          renderCell: GeologicalCellRenderer
        }
      ];

    case 'seismic':
      return [
        ...baseColumns,
        {
          field: 'event_id',
          headerName: 'Event ID',
          width: 150
        },
        {
          field: 'magnitude',
          headerName: 'Magnitude',
          width: 100,
          type: 'number',
          valueFormatter: (params: GridValueFormatterParams) => formatters.magnitude(params.value)
        },
        {
          field: 'depth',
          headerName: 'Depth',
          width: 100,
          type: 'number',
          valueFormatter: (params: GridValueFormatterParams) => formatters.depth(params.value)
        },
        {
          field: 'location',
          headerName: 'Location',
          width: 200,
          renderCell: GeologicalCellRenderer,
          valueGetter: (params: GridValueGetterParams) => params.row.location
        },
        {
          field: 'event_type',
          headerName: 'Type',
          width: 120,
          renderCell: GeologicalCellRenderer
        }
      ];

    case 'volcanic':
      return [
        ...baseColumns,
        {
          field: 'volcano_id',
          headerName: 'Volcano ID',
          width: 150
        },
        {
          field: 'alert_level',
          headerName: 'Alert Level',
          width: 120,
          renderCell: GeologicalCellRenderer
        },
        {
          field: 'eruption_probability',
          headerName: 'Eruption Probability',
          width: 150,
          type: 'number',
          renderCell: GeologicalCellRenderer
        },
        {
          field: 'magma_chamber_pressure',
          headerName: 'Chamber Pressure',
          width: 150,
          type: 'number',
          valueFormatter: (params: GridValueFormatterParams) => formatters.pressure(params.value)
        },
        {
          field: 'deformation_rate',
          headerName: 'Deformation Rate',
          width: 150,
          type: 'number'
        }
      ];

    default:
      return baseColumns;
  }
};

export const GeologicalDataGrid: React.FC<GeologicalDataGridProps> = ({
  data,
  dataType,
  onRowSelect,
  onRowEdit,
  onRowDelete,
  onExport,
  onVisualize,
  realTimeUpdate = false,
  // enableVirtualization = true, // Unused
  customColumns,
  height = 600
}) => {
  const [selectionModel, setSelectionModel] = useState<GridRowSelectionModel>([]);
  const [filterModel, setFilterModel] = useState<GridFilterModel>({ items: [] });
  const [sortModel, setSortModel] = useState<GridSortModel>([]);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editingRow, setEditingRow] = useState<any>(null);
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});

  const gridRef = useRef<any>(null);

  // Memoized columns
  const columns = useMemo(() => {
    const baseColumns = generateColumns(dataType, customColumns);
    
    // Add action column
    const actionColumn: GridColDef = {
      field: 'actions',
      headerName: 'Actions',
      width: 120,
      sortable: false,
      filterable: false,
      renderCell: (params: GridRenderCellParams) => (
        <Box>
          <IconButton
            size="small"
            onClick={(e: React.MouseEvent) => {
              e.stopPropagation();
              setAnchorEl(e.currentTarget as HTMLElement);
              setEditingRow(params.row);
            }}
          >
            <MoreVert />
          </IconButton>
        </Box>
      )
    };

    return [...baseColumns, actionColumn];
  }, [dataType, customColumns]);

  // Memoized rows with validation
  const rows: GridRowsProp = useMemo(() => {
    return data.map((item, index) => ({
      ...item,
      id: item.id || `${dataType}_${index}`,
      _isValid: validateRow(item)
    }));
  }, [data, dataType]);

  // Row validation
  const validateRow = useCallback((row: any): boolean => {
    const errors: Record<string, string> = {};

    // Validate coordinates
    if (row.latitude && !validators.latitude(row.latitude)) {
      errors.latitude = 'Invalid latitude';
    }
    if (row.longitude && !validators.longitude(row.longitude)) {
      errors.longitude = 'Invalid longitude';
    }

    // Validate magnitude for seismic events
    if (dataType === 'seismic' && row.magnitude && !validators.magnitude(row.magnitude)) {
      errors.magnitude = 'Invalid magnitude';
    }

    // Validate depth
    if (row.depth && !validators.depth(row.depth)) {
      errors.depth = 'Invalid depth';
    }

    setValidationErrors(prev => ({ ...prev, [row.id]: Object.keys(errors).length > 0 ? errors : {} }));
    return Object.keys(errors).length === 0;
  }, [dataType]);

  // Handle selection change
  const handleSelectionChange = useCallback((newSelection: GridRowSelectionModel) => {
    setSelectionModel(newSelection);
    const selectedRows = rows.filter(row => newSelection.includes(row.id));
    onRowSelect?.(selectedRows);
  }, [rows, onRowSelect]);

  // Handle row edit
  const handleRowEdit = useCallback((row: any) => {
    setEditingRow(row);
    setEditDialogOpen(true);
    setAnchorEl(null);
  }, []);

  // Handle row delete
  const handleRowDelete = useCallback((row: any) => {
    onRowDelete?.(row);
    setAnchorEl(null);
  }, [onRowDelete]);

  // Export data
  const handleExport = useCallback((format: string) => {
    const selectedRows = rows.filter(row => selectionModel.includes(row.id));
    const exportData = selectedRows.length > 0 ? selectedRows : rows;
    onExport?.([...exportData], format);
  }, [rows, selectionModel, onExport]);

  // Visualize data
  const handleVisualize = useCallback((type: string) => {
    const selectedRows = rows.filter(row => selectionModel.includes(row.id));
    const visualizeData = selectedRows.length > 0 ? selectedRows : rows;
    onVisualize?.([...visualizeData], type);
  }, [rows, selectionModel, onVisualize]);

  // Auto-refresh for real-time updates
  useEffect(() => {
    if (!realTimeUpdate) return;

    const interval = setInterval(() => {
      if (gridRef.current) {
        gridRef.current.forceUpdate();
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [realTimeUpdate]);

  return (
    <Box sx={{ width: '100%', height }}>
      {/* Toolbar */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            {dataType.charAt(0).toUpperCase() + dataType.slice(1)} Data Grid
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              startIcon={<MapIcon />}
              onClick={() => handleVisualize('map')}
              disabled={selectionModel.length === 0 && rows.length === 0}
            >
              Map View
            </Button>
            <Button
              startIcon={<Timeline />}
              onClick={() => handleVisualize('timeseries')}
              disabled={selectionModel.length === 0 && rows.length === 0}
            >
              Time Series
            </Button>
            <Button
              startIcon={<Assessment />}
              onClick={() => handleVisualize('chart')}
              disabled={selectionModel.length === 0 && rows.length === 0}
            >
              Charts
            </Button>
            <Button
              startIcon={<Download />}
              onClick={() => handleExport('csv')}
              disabled={rows.length === 0}
            >
              Export
            </Button>
          </Box>
        </Box>

        {/* Statistics */}
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <Chip label={`Total: ${rows.length}`} />
          <Chip label={`Selected: ${selectionModel.length}`} />
          <Chip 
            label={`Valid: ${rows.filter(row => row._isValid).length}`} 
            color="success" 
          />
          <Chip 
            label={`Invalid: ${rows.filter(row => !row._isValid).length}`} 
            color="error" 
          />
          {realTimeUpdate && (
            <Chip label="Real-time" color="primary" variant="outlined" />
          )}
        </Box>
      </Paper>

      {/* Data Grid */}
      <Paper sx={{ height: height - 120 }}>
        <DataGrid
          ref={gridRef}
          rows={rows}
          columns={columns}
          checkboxSelection
          disableSelectionOnClick
          selectionModel={selectionModel}
          onSelectionModelChange={handleSelectionChange}
          filterModel={filterModel}
          onFilterModelChange={setFilterModel}
          sortModel={sortModel}
          onSortModelChange={setSortModel}
          components={{
            Toolbar: GridToolbar
          }}
          componentsProps={{
            toolbar: {
              showQuickFilter: true,
              quickFilterProps: { debounceMs: 500 }
            }
          }}
          experimentalFeatures={{ 
            columnGrouping: true
          }}
          getRowClassName={(params) => 
            !params.row._isValid ? 'invalid-row' : ''
          }
          sx={{
            '& .invalid-row': {
              backgroundColor: 'rgba(255, 0, 0, 0.1)',
              '&:hover': {
                backgroundColor: 'rgba(255, 0, 0, 0.2)'
              }
            }
          }}
        />
      </Paper>

      {/* Action Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
      >
        <MenuItem onClick={() => handleRowEdit(editingRow)}>
          <Edit sx={{ mr: 1 }} />
          Edit
        </MenuItem>
        <MenuItem onClick={() => handleVisualize('detail')}>
          <Visibility sx={{ mr: 1 }} />
          View Details
        </MenuItem>
        <MenuItem onClick={() => handleRowDelete(editingRow)} sx={{ color: 'error.main' }}>
          <Delete sx={{ mr: 1 }} />
          Delete
        </MenuItem>
      </Menu>

      {/* Edit Dialog */}
      <Dialog
        open={editDialogOpen}
        onClose={() => setEditDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Edit {dataType} Data</DialogTitle>
        <DialogContent>
          {editingRow && (
            <Box sx={{ pt: 2 }}>
              {Object.entries(editingRow).map(([key, value]) => {
                if (key === 'id' || key === '_isValid' || key === 'actions') return null;
                
                return (
                  <TextField
                    key={key}
                    fullWidth
                    label={key.charAt(0).toUpperCase() + key.slice(1)}
                    value={value || ''}
                    onChange={(e) => {
                      setEditingRow((prev: any) => ({ ...prev, [key]: (e.target as HTMLInputElement).value }));
                    }}
                    error={Boolean(validationErrors[editingRow.id]?.[key as string])}
                    helperText={validationErrors[editingRow.id]?.[key as string]}
                    sx={{ mb: 2 }}
                  />
                );
              })}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={() => {
              onRowEdit?.(editingRow);
              setEditDialogOpen(false);
            }}
            variant="contained"
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>

      {/* Validation Errors Alert */}
      {Object.keys(validationErrors).some(key => Object.keys(validationErrors[key]).length > 0) && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          Some rows contain validation errors. Please review and correct the highlighted data.
        </Alert>
      )}
    </Box>
  );
};

export default GeologicalDataGrid;