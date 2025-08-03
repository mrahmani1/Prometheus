# Kaggle TPU Training Setup Instructions

## Prerequisites
1. Kaggle account with phone verification (required for TPU access)
2. The following files from this repository:
   - `prometheus_tpu_training.ipynb`
   - `api_config.py`
   - `data_fetcher_notebook.ipynb` (optional, for data exploration)

## Step 1: Upload Files to Kaggle

1. Go to [Kaggle](https://www.kaggle.com) and sign in
2. Click "Code" → "New Notebook"
3. Upload the notebook:
   - Click "File" → "Upload Notebook"
   - Select `prometheus_tpu_training.ipynb`
4. Upload API configuration:
   - In the notebook, click "Add data" → "Upload"
   - Upload `api_config.py` to `/kaggle/working/`

## Step 2: Enable TPU

1. In the notebook editor, click the three dots menu (⋮) in the top right
2. Select "Accelerator" → "TPU v3-8"
3. Confirm the selection

## Step 3: Remote Jupyter Access (Optional)

To connect your local IDE to the Kaggle notebook:

1. Run the first two cells in the notebook to set up Jupyter server
2. Install ngrok on Kaggle:
   ```bash
   !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
   !unzip ngrok-stable-linux-amd64.zip
   ```
3. Set up ngrok tunnel:
   ```bash
   !./ngrok authtoken YOUR_NGROK_TOKEN  # Get token from ngrok.com
   !./ngrok http 8888 &
   ```
4. Get the public URL from ngrok output
5. Connect your IDE using the ngrok URL and the password you set

## Step 4: Run Training

The notebook will:
1. Attempt to fetch real data from USGS (earthquake data)
2. Connect to Sentinel Hub for thermal imagery metadata
3. Fall back to synthetic data if APIs are unavailable
4. Train models using TPU acceleration

## Data Sources Used

The training notebook integrates with these APIs using your provided credentials:

### USGS Earthquake Data
- **Endpoint**: `https://earthquake.usgs.gov/fdsnws/event/1/query`
- **Data**: Recent earthquake events with magnitude, location, depth
- **Features**: 8 seismic parameters per event

### Sentinel Hub Thermal Imagery
- **Endpoint**: `https://services.sentinel-hub.com/api/v1/catalog/search`
- **Data**: Sentinel-3 SLSTR thermal band metadata
- **Note**: Full image download requires additional processing

### IRIS Seismic Stations (Metadata only in training)
- **Endpoint**: `http://service.iris.edu/fdsnws/station/1/query`
- **Data**: Station locations for waveform context

### NASA EarthData (Metadata only in training)
- **Endpoint**: `https://cmr.earthdata.nasa.gov/search/granules.json`
- **Data**: InSAR deformation products metadata

## Important Notes

1. **API Keys Security**: 
   - Never commit `api_config.py` to public repositories
   - The notebook uses environment variables as fallback
   - Consider using Kaggle Secrets for production

2. **Data Limitations**:
   - Real-time data fetching may hit rate limits
   - Full satellite imagery requires additional download steps
   - Training uses synthetic data when real data is unavailable

3. **TPU Optimization**:
   - Batch size is automatically scaled by TPU replicas
   - Models use mixed precision for better TPU performance
   - Data pipeline uses tf.data for optimal throughput

4. **Model Export**:
   - Models are saved in multiple formats (SavedModel, TFLite)
   - Download trained models before session ends
   - Kaggle sessions have a 9-hour limit

## Troubleshooting

### TPU Not Available
- Ensure phone verification is complete
- Check TPU quota hasn't been exceeded
- Try switching to GPU as fallback

### API Connection Issues
- Verify credentials in `api_config.py`
- Check API rate limits
- Network requests may be blocked in some regions

### Memory Issues
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

## Next Steps

After training:
1. Download trained models from `/kaggle/working/`
2. Use `data_fetcher_notebook.ipynb` locally for detailed data analysis
3. Deploy models using the inference pipeline in the notebook
4. Set up continuous monitoring with real-time data feeds