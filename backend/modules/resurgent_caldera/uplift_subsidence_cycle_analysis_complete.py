    async def _characterize_cycles(
        self, 
        cycle_detection: Dict[str, Any], 
        processed_data: Dict[str, Any]
    ) -> List[UpliftSubsidenceCycle]:
        """Characterize detected cycles with detailed properties"""
        cycles = cycle_detection['cycles']
        time_years = processed_data['time_years']
        displacements = processed_data['displacements']
        
        characterized_cycles = []
        
        for i, cycle_data in enumerate(cycles):
            start_idx = cycle_data['start_index']
            end_idx = cycle_data['end_index']
            
            # Extract cycle segment
            cycle_time = time_years[start_idx:end_idx]
            cycle_displacement = displacements[start_idx:end_idx]
            
            # Calculate cycle properties
            max_uplift = np.max(cycle_displacement)
            max_subsidence = np.min(cycle_displacement)
            cycle_amplitude = max_uplift - max_subsidence
            
            # Calculate rates
            uplift_rate = max_uplift / (cycle_data['duration'] / 2) if cycle_data['duration'] > 0 else 0
            subsidence_rate = abs(max_subsidence) / (cycle_data['duration'] / 2) if cycle_data['duration'] > 0 else 0
            
            # Classify pattern
            pattern = await self._classify_cycle_pattern(cycle_displacement)
            
            # Determine driving mechanism (simplified heuristic)
            mechanism = await self._infer_driving_mechanism(cycle_displacement, cycle_data['duration'])
            
            # Calculate confidence score
            confidence = await self._calculate_cycle_confidence(cycle_displacement, cycle_data)
            
            cycle = UpliftSubsidenceCycle(
                cycle_id=f"cycle_{i:03d}",
                caldera_id="unknown",  # Will be set by caller
                start_date=datetime.now() - timedelta(days=365.25 * (time_years[-1] - cycle_data['start_time'])),
                end_date=datetime.now() - timedelta(days=365.25 * (time_years[-1] - cycle_data['end_time'])),
                cycle_duration=cycle_data['duration'],
                maximum_uplift=max_uplift,
                maximum_subsidence=max_subsidence,
                uplift_rate=uplift_rate,
                subsidence_rate=subsidence_rate,
                cycle_amplitude=cycle_amplitude,
                cycle_pattern=pattern,
                driving_mechanism=mechanism,
                confidence_score=confidence
            )
            
            characterized_cycles.append(cycle)
        
        return characterized_cycles

    async def _classify_cycle_pattern(self, cycle_displacement: np.ndarray) -> str:
        """Classify the pattern of a cycle"""
        # Calculate symmetry metrics
        mid_point = len(cycle_displacement) // 2
        first_half = cycle_displacement[:mid_point]
        second_half = cycle_displacement[mid_point:]
        
        # Reverse second half for comparison
        second_half_reversed = second_half[::-1]
        
        # Calculate correlation between halves
        if len(first_half) == len(second_half_reversed):
            correlation = np.corrcoef(first_half, second_half_reversed)[0, 1]
        else:
            # Handle unequal lengths
            min_len = min(len(first_half), len(second_half_reversed))
            correlation = np.corrcoef(first_half[:min_len], second_half_reversed[:min_len])[0, 1]
        
        # Classify based on correlation and other metrics
        if correlation > 0.8:
            return "symmetric"
        elif correlation > 0.5:
            return "asymmetric"
        else:
            return "complex"

    async def _infer_driving_mechanism(self, cycle_displacement: np.ndarray, duration: float) -> str:
        """Infer the driving mechanism based on cycle characteristics"""
        # Simple heuristic based on duration and amplitude characteristics
        max_displacement = np.max(np.abs(cycle_displacement))
        
        if duration < 2.0 and max_displacement > 50:  # Short, high amplitude
            return "magma_injection"
        elif duration > 10.0 and max_displacement < 20:  # Long, low amplitude
            return "volatile_exsolution"
        elif duration > 5.0 and max_displacement > 30:  # Medium duration, high amplitude
            return "thermal_expansion"
        else:
            return "unknown"

    async def _calculate_cycle_confidence(self, cycle_displacement: np.ndarray, cycle_data: Dict[str, Any]) -> float:
        """Calculate confidence score for cycle identification"""
        # Base confidence on signal-to-noise ratio and cycle clarity
        signal_strength = np.std(cycle_displacement)
        cycle_amplitude = cycle_data['amplitude']
        
        # Simple confidence metric
        if cycle_amplitude > 2 * signal_strength and len(cycle_displacement) > 10:
            return 0.9
        elif cycle_amplitude > signal_strength and len(cycle_displacement) > 5:
            return 0.7
        else:
            return 0.5

    async def _assess_trend_significance(
        self, 
        trend_analysis: Dict[str, Any], 
        processed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess statistical significance of identified trends"""
        trend = trend_analysis['trend']
        detrended = trend_analysis['detrended']
        
        # Perform statistical tests
        # Mann-Kendall test for trend significance
        n = len(trend)
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(trend[j] - trend[i])
        
        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate Z statistic
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Calculate p-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {
            'mann_kendall_statistic': s,
            'z_statistic': z,
            'p_value': p_value,
            'trend_significant': p_value < 0.05,
            'trend_strength': trend_analysis['trend_strength']
        }

    async def _calculate_trend_uncertainties(
        self, 
        trend_analysis: Dict[str, Any], 
        processed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate uncertainties in trend analysis"""
        trend = trend_analysis['trend']
        detrended = trend_analysis['detrended']
        
        # Bootstrap uncertainty estimation
        n_bootstrap = 1000
        trend_slopes = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(trend), size=len(trend), replace=True)
            bootstrap_trend = trend[indices]
            bootstrap_time = processed_data['time_years'][indices]
            
            # Calculate slope
            slope = np.polyfit(bootstrap_time, bootstrap_trend, 1)[0]
            trend_slopes.append(slope)
        
        trend_slopes = np.array(trend_slopes)
        
        return {
            'trend_slope_mean': np.mean(trend_slopes),
            'trend_slope_std': np.std(trend_slopes),
            'trend_slope_95_ci': np.percentile(trend_slopes, [2.5, 97.5]),
            'bootstrap_samples': n_bootstrap
        }   
 async def _calculate_basic_statistics(self, cycles: List[UpliftSubsidenceCycle]) -> Dict[str, Any]:
        """Calculate basic statistical properties of cycles"""
        if not cycles:
            return {}
        
        durations = [cycle.cycle_duration for cycle in cycles]
        amplitudes = [cycle.cycle_amplitude for cycle in cycles]
        uplift_rates = [cycle.uplift_rate for cycle in cycles]
        subsidence_rates = [cycle.subsidence_rate for cycle in cycles]
        
        return {
            'duration_stats': {
                'mean': np.mean(durations),
                'std': np.std(durations),
                'median': np.median(durations),
                'min': np.min(durations),
                'max': np.max(durations),
                'q25': np.percentile(durations, 25),
                'q75': np.percentile(durations, 75)
            },
            'amplitude_stats': {
                'mean': np.mean(amplitudes),
                'std': np.std(amplitudes),
                'median': np.median(amplitudes),
                'min': np.min(amplitudes),
                'max': np.max(amplitudes),
                'q25': np.percentile(amplitudes, 25),
                'q75': np.percentile(amplitudes, 75)
            },
            'uplift_rate_stats': {
                'mean': np.mean(uplift_rates),
                'std': np.std(uplift_rates),
                'median': np.median(uplift_rates),
                'min': np.min(uplift_rates),
                'max': np.max(uplift_rates)
            },
            'subsidence_rate_stats': {
                'mean': np.mean(subsidence_rates),
                'std': np.std(subsidence_rates),
                'median': np.median(subsidence_rates),
                'min': np.min(subsidence_rates),
                'max': np.max(subsidence_rates)
            },
            'n_cycles': len(cycles)
        }

    async def _test_normality(self, cycles: List[UpliftSubsidenceCycle]) -> Dict[str, Any]:
        """Test for normality in cycle parameters"""
        if len(cycles) < 3:
            return {'insufficient_data': True}
        
        durations = [cycle.cycle_duration for cycle in cycles]
        amplitudes = [cycle.cycle_amplitude for cycle in cycles]
        
        # Shapiro-Wilk test
        duration_shapiro = stats.shapiro(durations)
        amplitude_shapiro = stats.shapiro(amplitudes)
        
        # Kolmogorov-Smirnov test
        duration_ks = stats.kstest(durations, 'norm', args=(np.mean(durations), np.std(durations)))
        amplitude_ks = stats.kstest(amplitudes, 'norm', args=(np.mean(amplitudes), np.std(amplitudes)))
        
        return {
            'duration_normality': {
                'shapiro_statistic': duration_shapiro.statistic,
                'shapiro_pvalue': duration_shapiro.pvalue,
                'ks_statistic': duration_ks.statistic,
                'ks_pvalue': duration_ks.pvalue,
                'is_normal': duration_shapiro.pvalue > 0.05 and duration_ks.pvalue > 0.05
            },
            'amplitude_normality': {
                'shapiro_statistic': amplitude_shapiro.statistic,
                'shapiro_pvalue': amplitude_shapiro.pvalue,
                'ks_statistic': amplitude_ks.statistic,
                'ks_pvalue': amplitude_ks.pvalue,
                'is_normal': amplitude_shapiro.pvalue > 0.05 and amplitude_ks.pvalue > 0.05
            }
        }

    async def _test_stationarity(self, cycles: List[UpliftSubsidenceCycle]) -> Dict[str, Any]:
        """Test for stationarity in cycle time series"""
        if len(cycles) < 10:
            return {'insufficient_data': True}
        
        # Create time series of cycle properties
        durations = np.array([cycle.cycle_duration for cycle in cycles])
        amplitudes = np.array([cycle.cycle_amplitude for cycle in cycles])
        
        # Augmented Dickey-Fuller test (simplified version)
        def adf_test(series):
            # Simple trend test
            n = len(series)
            x = np.arange(n)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
            return {
                'trend_slope': slope,
                'trend_pvalue': p_value,
                'is_stationary': p_value > 0.05  # No significant trend
            }
        
        duration_stationarity = adf_test(durations)
        amplitude_stationarity = adf_test(amplitudes)
        
        return {
            'duration_stationarity': duration_stationarity,
            'amplitude_stationarity': amplitude_stationarity
        }

    async def _analyze_periodicity(self, cycles: List[UpliftSubsidenceCycle]) -> Dict[str, Any]:
        """Analyze periodicity in cycle occurrence"""
        if len(cycles) < 5:
            return {'insufficient_data': True}
        
        # Calculate inter-cycle intervals
        start_times = [cycle.start_date.timestamp() for cycle in cycles]
        start_times.sort()
        intervals = np.diff(start_times) / (365.25 * 24 * 3600)  # Convert to years
        
        # Fourier analysis of intervals
        fft = np.fft.fft(intervals)
        freqs = np.fft.fftfreq(len(intervals))
        power_spectrum = np.abs(fft) ** 2
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        dominant_period = 1 / freqs[dominant_freq_idx] if freqs[dominant_freq_idx] != 0 else np.inf
        
        # Calculate regularity metric
        interval_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else np.inf
        regularity = 1 / (1 + interval_cv)  # Higher values = more regular
        
        return {
            'mean_interval': np.mean(intervals),
            'interval_std': np.std(intervals),
            'interval_cv': interval_cv,
            'regularity_score': regularity,
            'dominant_period': dominant_period,
            'intervals': intervals.tolist()
        }

    async def _analyze_correlations(self, cycles: List[UpliftSubsidenceCycle]) -> Dict[str, Any]:
        """Analyze correlations between cycle parameters"""
        if len(cycles) < 3:
            return {'insufficient_data': True}
        
        # Extract parameters
        durations = np.array([cycle.cycle_duration for cycle in cycles])
        amplitudes = np.array([cycle.cycle_amplitude for cycle in cycles])
        uplift_rates = np.array([cycle.uplift_rate for cycle in cycles])
        subsidence_rates = np.array([cycle.subsidence_rate for cycle in cycles])
        
        # Calculate correlation matrix
        data_matrix = np.column_stack([durations, amplitudes, uplift_rates, subsidence_rates])
        correlation_matrix = np.corrcoef(data_matrix.T)
        
        # Calculate specific correlations
        duration_amplitude_corr = stats.pearsonr(durations, amplitudes)
        duration_uplift_corr = stats.pearsonr(durations, uplift_rates)
        amplitude_rate_corr = stats.pearsonr(amplitudes, uplift_rates)
        
        return {
            'correlation_matrix': correlation_matrix.tolist(),
            'parameter_names': ['duration', 'amplitude', 'uplift_rate', 'subsidence_rate'],
            'duration_amplitude': {
                'correlation': duration_amplitude_corr[0],
                'pvalue': duration_amplitude_corr[1]
            },
            'duration_uplift_rate': {
                'correlation': duration_uplift_corr[0],
                'pvalue': duration_uplift_corr[1]
            },
            'amplitude_uplift_rate': {
                'correlation': amplitude_rate_corr[0],
                'pvalue': amplitude_rate_corr[1]
            }
        } 
   async def _fit_distributions(self, cycles: List[UpliftSubsidenceCycle]) -> Dict[str, Any]:
        """Fit probability distributions to cycle parameters"""
        if len(cycles) < 5:
            return {'insufficient_data': True}
        
        durations = np.array([cycle.cycle_duration for cycle in cycles])
        amplitudes = np.array([cycle.cycle_amplitude for cycle in cycles])
        
        # Test different distributions
        distributions = ['norm', 'lognorm', 'gamma', 'weibull_min']
        
        def fit_distribution(data, dist_name):
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(data)
                ks_stat, p_value = stats.kstest(data, dist.cdf, args=params)
                return {
                    'parameters': params,
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'aic': 2 * len(params) - 2 * np.sum(dist.logpdf(data, *params))
                }
            except:
                return None
        
        duration_fits = {}
        amplitude_fits = {}
        
        for dist_name in distributions:
            duration_fit = fit_distribution(durations, dist_name)
            amplitude_fit = fit_distribution(amplitudes, dist_name)
            
            if duration_fit:
                duration_fits[dist_name] = duration_fit
            if amplitude_fit:
                amplitude_fits[dist_name] = amplitude_fit
        
        # Find best fitting distributions
        best_duration_dist = min(duration_fits.keys(), 
                               key=lambda x: duration_fits[x]['aic']) if duration_fits else None
        best_amplitude_dist = min(amplitude_fits.keys(), 
                                key=lambda x: amplitude_fits[x]['aic']) if amplitude_fits else None
        
        return {
            'duration_distributions': duration_fits,
            'amplitude_distributions': amplitude_fits,
            'best_duration_distribution': best_duration_dist,
            'best_amplitude_distribution': best_amplitude_dist
        }

    async def _detect_outliers(self, cycles: List[UpliftSubsidenceCycle]) -> Dict[str, Any]:
        """Detect outlier cycles using multiple methods"""
        if len(cycles) < 5:
            return {'insufficient_data': True}
        
        # Extract parameters
        durations = np.array([cycle.cycle_duration for cycle in cycles])
        amplitudes = np.array([cycle.cycle_amplitude for cycle in cycles])
        
        # Z-score method
        duration_z_scores = np.abs(stats.zscore(durations))
        amplitude_z_scores = np.abs(stats.zscore(amplitudes))
        
        # IQR method
        def iqr_outliers(data):
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return (data < lower_bound) | (data > upper_bound)
        
        duration_iqr_outliers = iqr_outliers(durations)
        amplitude_iqr_outliers = iqr_outliers(amplitudes)
        
        # Isolation Forest (if sklearn available)
        try:
            from sklearn.ensemble import IsolationForest
            data_matrix = np.column_stack([durations, amplitudes])
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(data_matrix)
            isolation_outliers = outlier_labels == -1
        except:
            isolation_outliers = np.zeros(len(cycles), dtype=bool)
        
        # Combine outlier detection methods
        outlier_indices = []
        for i in range(len(cycles)):
            is_outlier = (duration_z_scores[i] > 2.5 or 
                         amplitude_z_scores[i] > 2.5 or
                         duration_iqr_outliers[i] or 
                         amplitude_iqr_outliers[i] or
                         isolation_outliers[i])
            if is_outlier:
                outlier_indices.append(i)
        
        return {
            'outlier_indices': outlier_indices,
            'outlier_cycle_ids': [cycles[i].cycle_id for i in outlier_indices],
            'duration_z_scores': duration_z_scores.tolist(),
            'amplitude_z_scores': amplitude_z_scores.tolist(),
            'n_outliers': len(outlier_indices),
            'outlier_percentage': len(outlier_indices) / len(cycles) * 100
        }

    async def _prepare_ml_features(self, cycles: List[UpliftSubsidenceCycle]) -> Dict[str, Any]:
        """Prepare features for machine learning analysis"""
        if not cycles:
            return {'features': np.array([]), 'feature_names': []}
        
        # Extract basic features
        features = []
        feature_names = [
            'duration', 'amplitude', 'uplift_rate', 'subsidence_rate',
            'max_uplift', 'max_subsidence', 'confidence_score'
        ]
        
        for cycle in cycles:
            cycle_features = [
                cycle.cycle_duration,
                cycle.cycle_amplitude,
                cycle.uplift_rate,
                cycle.subsidence_rate,
                cycle.maximum_uplift,
                cycle.maximum_subsidence,
                cycle.confidence_score
            ]
            features.append(cycle_features)
        
        features = np.array(features)
        
        # Add derived features
        if len(features) > 0:
            # Ratios
            uplift_subsidence_ratio = features[:, 4] / (np.abs(features[:, 5]) + 1e-6)  # max_uplift / |max_subsidence|
            rate_ratio = features[:, 2] / (features[:, 3] + 1e-6)  # uplift_rate / subsidence_rate
            amplitude_duration_ratio = features[:, 1] / (features[:, 0] + 1e-6)  # amplitude / duration
            
            # Add to features
            features = np.column_stack([features, uplift_subsidence_ratio, rate_ratio, amplitude_duration_ratio])
            feature_names.extend(['uplift_subsidence_ratio', 'rate_ratio', 'amplitude_duration_ratio'])
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features) if len(features) > 0 else features
        
        return {
            'features': features,
            'normalized_features': normalized_features,
            'feature_names': feature_names,
            'scaler': scaler,
            'n_features': len(feature_names),
            'n_samples': len(cycles)
        }

    async def _apply_ml_clustering(
        self, 
        engineered_features: Dict[str, Any], 
        cycles: List[UpliftSubsidenceCycle]
    ) -> Dict[str, Any]:
        """Apply machine learning clustering to identify cycle groups"""
        features = engineered_features['normalized_features']
        
        if len(features) < 3:
            return {'insufficient_data': True}
        
        # K-means clustering
        optimal_k = min(5, len(features) // 2)  # Reasonable upper bound
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(features)
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        dbscan_labels = dbscan.fit_predict(features)
        
        # Calculate silhouette scores
        try:
            from sklearn.metrics import silhouette_score
            kmeans_silhouette = silhouette_score(features, kmeans_labels)
            dbscan_silhouette = silhouette_score(features, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1
        except:
            kmeans_silhouette = 0
            dbscan_silhouette = 0
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in set(kmeans_labels):
            cluster_mask = kmeans_labels == cluster_id
            cluster_cycles = [cycles[i] for i in range(len(cycles)) if cluster_mask[i]]
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'n_cycles': len(cluster_cycles),
                'mean_duration': np.mean([c.cycle_duration for c in cluster_cycles]),
                'mean_amplitude': np.mean([c.cycle_amplitude for c in cluster_cycles]),
                'dominant_pattern': max(set([c.cycle_pattern for c in cluster_cycles]), 
                                     key=[c.cycle_pattern for c in cluster_cycles].count)
            }
        
        return {
            'kmeans_labels': kmeans_labels.tolist(),
            'dbscan_labels': dbscan_labels.tolist(),
            'kmeans_silhouette_score': kmeans_silhouette,
            'dbscan_silhouette_score': dbscan_silhouette,
            'optimal_k': optimal_k,
            'cluster_analysis': cluster_analysis,
            'n_clusters_kmeans': len(set(kmeans_labels)),
            'n_clusters_dbscan': len(set(dbscan_labels))
        } 
   async def fetch_real_deformation_data(
        self, 
        caldera_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> DeformationTimeSeries:
        """Fetch real deformation data from external sources"""
        logger.info(f"Fetching real deformation data for {caldera_id}")
        
        if DATABASE_INTEGRATION_AVAILABLE:
            try:
                # Initialize connectors
                usgs_connector = USGSVolcanoConnector()
                iris_connector = IRISConnector()
                
                # Fetch InSAR data from USGS
                insar_data = await usgs_connector.fetch_insar_data(
                    caldera_id, start_date, end_date
                )
                
                # Fetch GPS data from IRIS
                gps_data = await iris_connector.fetch_gps_data(
                    caldera_id, start_date, end_date
                )
                
                # Combine and process data
                if insar_data and len(insar_data['timestamps']) > 0:
                    return DeformationTimeSeries(
                        caldera_id=caldera_id,
                        timestamps=np.array(insar_data['timestamps']),
                        vertical_displacement=np.array(insar_data['vertical_displacement']),
                        horizontal_displacement=np.array(insar_data.get('horizontal_displacement', [])),
                        uncertainties=np.array(insar_data.get('uncertainties', [])),
                        data_source='InSAR_USGS',
                        processing_level='L2',
                        quality_flags=np.array(insar_data.get('quality_flags', []))
                    )
                elif gps_data and len(gps_data['timestamps']) > 0:
                    return DeformationTimeSeries(
                        caldera_id=caldera_id,
                        timestamps=np.array(gps_data['timestamps']),
                        vertical_displacement=np.array(gps_data['vertical_displacement']),
                        horizontal_displacement=np.array(gps_data.get('horizontal_displacement', [])),
                        uncertainties=np.array(gps_data.get('uncertainties', [])),
                        data_source='GPS_IRIS',
                        processing_level='L2'
                    )
                
            except Exception as e:
                logger.warning(f"Failed to fetch real data: {e}")
        
        # Generate synthetic data as fallback
        return await self._generate_synthetic_deformation_data(caldera_id, start_date, end_date)
    
    async def _generate_synthetic_deformation_data(
        self, 
        caldera_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> DeformationTimeSeries:
        """Generate synthetic deformation data for testing"""
        logger.info(f"Generating synthetic deformation data for {caldera_id}")
        
        # Create time series
        time_range = (end_date - start_date).total_seconds() / (365.25 * 24 * 3600)  # years
        n_points = int(time_range * 12)  # Monthly sampling
        
        timestamps = [start_date + timedelta(days=i * 30.44) for i in range(n_points)]
        time_years = np.linspace(0, time_range, n_points)
        
        # Generate synthetic uplift-subsidence cycles
        displacement = np.zeros(n_points)
        
        # Add multiple cycles with different periods
        for cycle_period in [3.0, 7.0, 12.0]:  # years
            amplitude = np.random.uniform(10, 50)  # mm
            phase = np.random.uniform(0, 2 * np.pi)
            cycle_component = amplitude * np.sin(2 * np.pi * time_years / cycle_period + phase)
            displacement += cycle_component
        
        # Add long-term trend
        trend_slope = np.random.uniform(-5, 5)  # mm/year
        displacement += trend_slope * time_years
        
        # Add noise
        noise_level = np.std(displacement) * 0.1
        displacement += np.random.normal(0, noise_level, n_points)
        
        # Generate uncertainties
        uncertainties = np.random.uniform(1, 5, n_points)  # mm
        
        # Generate quality flags
        quality_flags = np.random.uniform(0.7, 1.0, n_points)
        
        return DeformationTimeSeries(
            caldera_id=caldera_id,
            timestamps=np.array(timestamps),
            vertical_displacement=displacement,
            uncertainties=uncertainties,
            data_source='Synthetic',
            processing_level='L2',
            quality_flags=quality_flags
        )
    
    async def store_analysis_results(
        self, 
        caldera_id: str, 
        cycles: List[UpliftSubsidenceCycle],
        analysis_results: Dict[str, Any]
    ) -> bool:
        """Store analysis results in database"""
        try:
            for cycle in cycles:
                # Update caldera_id
                cycle.caldera_id = caldera_id
                
                # Create database record
                db_record = UpliftSubsidenceCycleDatabase(
                    caldera_id=caldera_id,
                    cycle_id=cycle.cycle_id,
                    cycle_start_date=cycle.start_date,
                    cycle_end_date=cycle.end_date,
                    cycle_duration=cycle.cycle_duration,
                    maximum_uplift=cycle.maximum_uplift,
                    maximum_subsidence=cycle.maximum_subsidence,
                    uplift_rate=cycle.uplift_rate,
                    subsidence_rate=cycle.subsidence_rate,
                    cycle_amplitude=cycle.cycle_amplitude,
                    cycle_pattern=cycle.cycle_pattern,
                    driving_mechanism=cycle.driving_mechanism,
                    confidence_score=cycle.confidence_score,
                    statistical_metrics=analysis_results.get('statistical_analysis', {}),
                    cycle_data={
                        'duration': cycle.cycle_duration,
                        'amplitude': cycle.cycle_amplitude,
                        'pattern': cycle.cycle_pattern
                    },
                    prediction_results=analysis_results.get('prediction_results', {})
                )
                
                self.session.add(db_record)
            
            self.session.commit()
            logger.info(f"Stored {len(cycles)} cycle records for caldera {caldera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store analysis results: {e}")
            self.session.rollback()
            return False
    
    async def query_historical_cycles(
        self, 
        caldera_id: str = None, 
        min_confidence: float = 0.5
    ) -> List[UpliftSubsidenceCycle]:
        """Query historical cycles from database"""
        try:
            query = self.session.query(UpliftSubsidenceCycleDatabase)
            
            if caldera_id:
                query = query.filter(UpliftSubsidenceCycleDatabase.caldera_id == caldera_id)
            
            query = query.filter(UpliftSubsidenceCycleDatabase.confidence_score >= min_confidence)
            
            db_records = query.all()
            
            cycles = []
            for record in db_records:
                cycle = UpliftSubsidenceCycle(
                    cycle_id=record.cycle_id,
                    caldera_id=record.caldera_id,
                    start_date=record.cycle_start_date,
                    end_date=record.cycle_end_date,
                    cycle_duration=record.cycle_duration,
                    maximum_uplift=record.maximum_uplift,
                    maximum_subsidence=record.maximum_subsidence,
                    uplift_rate=record.uplift_rate,
                    subsidence_rate=record.subsidence_rate,
                    cycle_amplitude=record.cycle_amplitude,
                    cycle_pattern=record.cycle_pattern,
                    driving_mechanism=record.driving_mechanism,
                    confidence_score=record.confidence_score
                )
                cycles.append(cycle)
            
            logger.info(f"Retrieved {len(cycles)} historical cycles")
            return cycles
            
        except Exception as e:
            logger.error(f"Failed to query historical cycles: {e}")
            return []

    async def run_comprehensive_analysis(
        self, 
        caldera_id: str, 
        start_date: datetime = None, 
        end_date: datetime = None
    ) -> Dict[str, Any]:
        """Run comprehensive uplift-subsidence cycle analysis"""
        logger.info(f"Starting comprehensive analysis for caldera {caldera_id}")
        
        # Set default date range if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365.25 * 20)  # 20 years
        
        try:
            # 1. Fetch deformation data
            deformation_data = await self.fetch_real_deformation_data(
                caldera_id, start_date, end_date
            )
            
            # 2. Analyze long-term trends and detect cycles
            trend_analysis = await self.analyze_long_term_trends(deformation_data)
            
            # 3. Characterize cycles
            cycles = trend_analysis['cycle_characterization']
            
            # 4. Statistical analysis
            statistical_analysis = await self.perform_statistical_analysis(cycles)
            
            # 5. Pattern characterization
            pattern_analysis = await self.characterize_cycle_patterns(cycles)
            
            # 6. Machine learning analysis
            ml_analysis = await self.apply_machine_learning_pattern_recognition(cycles)
            
            # 7. Prediction
            prediction_results = await self.predict_future_cycles(cycles)
            
            # 8. Uncertainty quantification
            uncertainty_analysis = await self.quantify_prediction_uncertainties(prediction_results)
            
            # 9. Store results
            comprehensive_results = {
                'caldera_id': caldera_id,
                'analysis_period': {'start': start_date.isoformat(), 'end': end_date.isoformat()},
                'deformation_data_summary': {
                    'n_points': len(deformation_data.timestamps),
                    'data_source': deformation_data.data_source,
                    'quality': deformation_data.processing_level
                },
                'trend_analysis': trend_analysis,
                'statistical_analysis': statistical_analysis,
                'pattern_analysis': pattern_analysis,
                'ml_analysis': ml_analysis,
                'prediction_results': prediction_results,
                'uncertainty_analysis': uncertainty_analysis,
                'n_cycles_detected': len(cycles),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            # Store in database
            await self.store_analysis_results(caldera_id, cycles, comprehensive_results)
            
            logger.info(f"Comprehensive analysis completed for {caldera_id}")
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {
                'error': str(e),
                'caldera_id': caldera_id,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }

# Example usage and testing
async def main():
    """Example usage of the UpliftSubsidenceCycleAnalyzer"""
    
    # Initialize analyzer
    database_url = "postgresql://user:password@localhost/prometheus_ai"
    analyzer = UpliftSubsidenceCycleAnalyzer(database_url)
    
    # Run comprehensive analysis for Yellowstone
    yellowstone_results = await analyzer.run_comprehensive_analysis(
        caldera_id="yellowstone",
        start_date=datetime(2000, 1, 1),
        end_date=datetime(2023, 12, 31)
    )
    
    print(f"Analysis completed for Yellowstone:")
    print(f"- Cycles detected: {yellowstone_results.get('n_cycles_detected', 0)}")
    print(f"- Data points: {yellowstone_results.get('deformation_data_summary', {}).get('n_points', 0)}")
    
    # Query historical cycles
    historical_cycles = await analyzer.query_historical_cycles(
        caldera_id="yellowstone",
        min_confidence=0.7
    )
    
    print(f"Historical cycles retrieved: {len(historical_cycles)}")

if __name__ == "__main__":
    asyncio.run(main())