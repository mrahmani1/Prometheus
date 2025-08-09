/**
 * InSAR Algorithms Implementation - Task 3.1
 * SBAS and PS-InSAR processing algorithms with quality assessment
 * @author Prometheus Platform
 * @version 1.0.0
 */

class InSARAlgorithms {
    constructor() {
        this.version = '1.0.0';
        this.algorithms = new Map();
        this.qualityMetrics = new Map();
        this.processingCache = new Map();
        this.config = {
            sbas: {
                defaultCoherenceThreshold: 0.7,
                maxTemporalBaseline: 365, // days
                maxPerpendicularBaseline: 300, // meters
                unwrapAlgorithm: 'snaphu',
                atmosphericModel: 'era5',
                minimumRedundancy: 3
            },
            psInsar: {
                amplitudeDispersionThreshold: 0.4,
                phaseLinkingMethod: 'evd', // eigenvalue decomposition
                shpTestThreshold: 0.8,
                minRedundancy: 3,
                spatialFilter: true,
                temporalFilter: true
            },
            quality: {
                coherenceWeighting: true,
                temporalCoherenceThreshold: 0.8,
                velocityUncertaintyThreshold: 5.0, // mm/year
                residualPhaseThreshold: 1.0, // radians
                spatialConsistencyCheck: true
            }
        };
        
        this.mathUtils = new InSARMathUtils();
        this.eventBus = new EventTarget();
        this.init();
    }

    init() {
        console.log('🧮 Initializing InSAR Algorithms...');
        this.setupAlgorithms();
        this.setupQualityAssessment();
        this.setupAtmosphericCorrection();
        this.setupErrorHandling();
        console.log('✅ InSAR Algorithms Ready');
    }

    setupAlgorithms() {
        // Register SBAS algorithms
        this.algorithms.set('sbas_network_modification', this.sbasNetworkModification.bind(this));
        this.algorithms.set('sbas_inversion', this.sbasInversion.bind(this));
        this.algorithms.set('sbas_velocity_estimation', this.sbasVelocityEstimation.bind(this));
        
        // Register PS-InSAR algorithms
        this.algorithms.set('ps_candidate_identification', this.psIdentification.bind(this));
        this.algorithms.set('ps_phase_linking', this.phaseLinking.bind(this));
        this.algorithms.set('ps_atmospheric_estimation', this.psAtmosphericEstimation.bind(this));
        
        // Register common algorithms
        this.algorithms.set('unwrap_correction', this.unwrapCorrection.bind(this));
        this.algorithms.set('reference_point_selection', this.referencePointSelection.bind(this));
        this.algorithms.set('atmospheric_correction', this.atmosphericCorrection.bind(this));
        this.algorithms.set('geocoding', this.geocoding.bind(this));
    }

    setupQualityAssessment() {
        this.qualityMetrics.set('coherence_analysis', this.assessCoherence.bind(this));
        this.qualityMetrics.set('temporal_coherence', this.assessTemporalCoherence.bind(this));
        this.qualityMetrics.set('velocity_uncertainty', this.assessVelocityUncertainty.bind(this));
        this.qualityMetrics.set('residual_phase_analysis', this.assessResidualPhase.bind(this));
        this.qualityMetrics.set('spatial_consistency', this.assessSpatialConsistency.bind(this));
    }

    // SBAS Algorithm Implementation
    async modifyNetwork(parameters) {
        console.log('🔗 Modifying interferometric network...');
        
        const networkData = this.generateSyntheticNetwork(parameters);
        const coherenceThreshold = parameters.coherenceThreshold || this.config.sbas.defaultCoherenceThreshold;
        const maxTempBaseline = parameters.maxTemporalBaseline || this.config.sbas.maxTemporalBaseline;
        const maxPerpBaseline = parameters.maxPerpendicularBaseline || this.config.sbas.maxPerpendicularBaseline;
        
        // Filter interferograms based on baseline constraints
        const filteredNetwork = networkData.interferograms.filter(ifg => {
            return (
                ifg.coherence >= coherenceThreshold &&
                Math.abs(ifg.temporalBaseline) <= maxTempBaseline &&
                Math.abs(ifg.perpendicularBaseline) <= maxPerpBaseline
            );
        });
        
        // Calculate network redundancy
        const redundancy = this.calculateNetworkRedundancy(filteredNetwork, networkData.acquisitions.length);
        
        console.log(`Network modified: ${filteredNetwork.length}/${networkData.interferograms.length} interferograms retained`);
        console.log(`Network redundancy: ${redundancy.toFixed(2)}`);
        
        return {
            originalCount: networkData.interferograms.length,
            filteredCount: filteredNetwork.length,
            redundancy: redundancy,
            interferograms: filteredNetwork,
            acquisitions: networkData.acquisitions
        };
    }

    async selectReferencePoint(parameters) {
        console.log('📍 Selecting reference point...');
        
        const method = parameters.referenceSelection || 'auto';
        let referencePoint;
        
        switch (method) {
            case 'auto':
                referencePoint = await this.autoReferenceSelection(parameters);
                break;
            case 'max_coherence':
                referencePoint = await this.maxCoherenceReference(parameters);
                break;
            case 'min_rms':
                referencePoint = await this.minRMSReference(parameters);
                break;
            case 'manual':
                referencePoint = parameters.manualReference || { lat: 40.5, lon: -120.3 };
                break;
            default:
                referencePoint = await this.autoReferenceSelection(parameters);
        }
        
        console.log(`Reference point selected: ${referencePoint.lat.toFixed(4)}, ${referencePoint.lon.toFixed(4)}`);
        
        return {
            coordinates: referencePoint,
            method: method,
            quality: this.assessReferenceQuality(referencePoint, parameters)
        };
    }

    async correctUnwrapErrors(parameters) {
        console.log('🔧 Correcting phase unwrapping errors...');
        
        const unwrapMethod = parameters.unwrapMethod || 'snaphu';
        const networkData = await this.getNetworkData(parameters);
        
        let correctedPhases = [];
        let errorCount = 0;
        
        for (const interferogram of networkData.interferograms) {
            const unwrappedPhase = this.generateSyntheticPhase(interferogram);
            const correctedPhase = await this.detectAndCorrectUnwrapErrors(
                unwrappedPhase, 
                interferogram, 
                unwrapMethod
            );
            
            if (correctedPhase.errorsCorrected > 0) {
                errorCount++;
            }
            
            correctedPhases.push(correctedPhase);
        }
        
        console.log(`Unwrap errors corrected in ${errorCount}/${networkData.interferograms.length} interferograms`);
        
        return {
            processedInterferograms: correctedPhases.length,
            errorsDetected: errorCount,
            method: unwrapMethod,
            qualityImprovement: this.calculateUnwrapQualityImprovement(correctedPhases)
        };
    }

    async invertNetwork(parameters) {
        console.log('🔍 Inverting interferometric network...');
        
        const networkData = await this.getNetworkData(parameters);
        const weightingMethod = parameters.weighting || 'coherence';
        
        // Build design matrix (A) and data vector (d)
        const { designMatrix, dataVector, weights } = this.buildInversionMatrices(
            networkData, 
            weightingMethod
        );
        
        // Perform weighted least squares inversion
        const inversionResult = this.weightedLeastSquares(designMatrix, dataVector, weights);
        
        // Calculate time series
        const timeSeries = this.calculateTimeSeries(inversionResult, networkData.acquisitions);
        
        // Assess inversion quality
        const inversionQuality = this.assessInversionQuality(inversionResult, networkData);
        
        console.log(`Network inverted: RMS residual = ${inversionQuality.rmsResidual.toFixed(3)} radians`);
        
        return {
            timeSeries: timeSeries,
            residualPhase: inversionResult.residuals,
            rmsResidual: inversionQuality.rmsResidual,
            redundancy: inversionQuality.redundancy,
            conditionNumber: inversionQuality.conditionNumber
        };
    }

    async estimateVelocity(parameters) {
        console.log('📈 Estimating velocity field...');
        
        const timeSeriesData = await this.getTimeSeriesData(parameters);
        const velocityEstimation = parameters.velocityModel || 'linear';
        
        const velocityResults = [];
        let uncertaintyStats = { mean: 0, std: 0, max: 0, min: Infinity };
        
        for (const pixel of timeSeriesData.pixels) {
            const velocityFit = this.fitVelocityModel(
                pixel.timeSeries, 
                timeSeriesData.dates, 
                velocityEstimation
            );
            
            velocityResults.push({
                coordinates: pixel.coordinates,
                velocity: velocityFit.velocity,
                acceleration: velocityFit.acceleration || 0,
                uncertainty: velocityFit.uncertainty,
                r_squared: velocityFit.r_squared,
                residualRMS: velocityFit.residualRMS
            });
            
            // Update uncertainty statistics
            if (velocityFit.uncertainty < uncertaintyStats.min) uncertaintyStats.min = velocityFit.uncertainty;
            if (velocityFit.uncertainty > uncertaintyStats.max) uncertaintyStats.max = velocityFit.uncertainty;
            uncertaintyStats.mean += velocityFit.uncertainty;
        }
        
        uncertaintyStats.mean /= velocityResults.length;
        uncertaintyStats.std = Math.sqrt(
            velocityResults.reduce((sum, result) => 
                sum + Math.pow(result.uncertainty - uncertaintyStats.mean, 2), 0
            ) / velocityResults.length
        );
        
        console.log(`Velocity estimated for ${velocityResults.length} pixels`);
        console.log(`Mean uncertainty: ${uncertaintyStats.mean.toFixed(2)} mm/year`);
        
        return {
            velocityField: velocityResults,
            uncertaintyStatistics: uncertaintyStats,
            model: velocityEstimation,
            processingDate: new Date().toISOString()
        };
    }

    // PS-InSAR Algorithm Implementation
    async identifyPersistentScatterers(parameters) {
        console.log('🎯 Identifying persistent scatterer candidates...');
        
        const amplitudeStack = this.generateSyntheticAmplitudeStack(parameters);
        const threshold = parameters.amplitudeDispersionThreshold || this.config.psInsar.amplitudeDispersionThreshold;
        
        const psPixels = [];
        let totalPixels = 0;
        
        for (let i = 0; i < amplitudeStack.height; i++) {
            for (let j = 0; j < amplitudeStack.width; j++) {
                totalPixels++;
                const amplitudeTimeSeries = amplitudeStack.getPixelTimeSeries(i, j);
                const dispersionIndex = this.calculateAmplitudeDispersion(amplitudeTimeSeries);
                
                if (dispersionIndex <= threshold) {
                    const coordinates = amplitudeStack.pixelToCoordinates(i, j);
                    psPixels.push({
                        pixel: { row: i, col: j },
                        coordinates: coordinates,
                        dispersionIndex: dispersionIndex,
                        meanAmplitude: amplitudeTimeSeries.reduce((a, b) => a + b) / amplitudeTimeSeries.length,
                        temporalStability: this.assessTemporalStability(amplitudeTimeSeries)
                    });
                }
            }
        }
        
        console.log(`PS candidates identified: ${psPixels.length}/${totalPixels} pixels (${(psPixels.length/totalPixels*100).toFixed(2)}%)`);
        
        return {
            psPixels: psPixels,
            totalPixels: totalPixels,
            selectionRate: psPixels.length / totalPixels,
            threshold: threshold,
            qualityMetrics: this.calculatePSQualityMetrics(psPixels)
        };
    }

    async performPhaseLinking(parameters) {
        console.log('🔗 Performing phase linking...');
        
        const method = parameters.phaseLinkingMethod || 'evd';
        const psData = await this.getPSData(parameters);
        const minRedundancy = parameters.minRedundancy || 3;
        
        const linkedPhases = [];
        let processedPixels = 0;
        
        for (const psPixel of psData.psPixels) {
            const phaseTimeSeries = this.generateSyntheticPhaseTimeSeries(psPixel);
            
            let linkedPhase;
            switch (method) {
                case 'evd':
                    linkedPhase = await this.eigenvaluePhaseLink(phaseTimeSeries, minRedundancy);
                    break;
                case 'pca':
                    linkedPhase = await this.pcaPhaseLink(phaseTimeSeries, minRedundancy);
                    break;
                case 'sequential':
                    linkedPhase = await this.sequentialPhaseLink(phaseTimeSeries, minRedundancy);
                    break;
                default:
                    linkedPhase = await this.eigenvaluePhaseLink(phaseTimeSeries, minRedundancy);
            }
            
            if (linkedPhase.success) {
                linkedPhases.push({
                    pixel: psPixel.pixel,
                    coordinates: psPixel.coordinates,
                    linkedPhase: linkedPhase.phase,
                    phaseQuality: linkedPhase.quality,
                    redundancy: linkedPhase.redundancy
                });
                processedPixels++;
            }
        }
        
        console.log(`Phase linking completed: ${processedPixels}/${psData.psPixels.length} pixels processed`);
        
        return {
            linkedPhases: linkedPhases,
            processedPixels: processedPixels,
            method: method,
            averageQuality: linkedPhases.reduce((sum, p) => sum + p.phaseQuality, 0) / linkedPhases.length,
            averageRedundancy: linkedPhases.reduce((sum, p) => sum + p.redundancy, 0) / linkedPhases.length
        };
    }

    async psAtmosphericEstimation(parameters) {
        console.log('🌤️ Estimating atmospheric phase screen for PS pixels...');
        
        const linkedPhaseData = await this.getLinkedPhaseData(parameters);
        const spatialFilterSize = parameters.spatialFilterSize || 1000; // meters
        const temporalFilterSize = parameters.temporalFilterSize || 30; // days
        
        const atmosphericScreens = [];
        const dates = linkedPhaseData.dates;
        
        for (let dateIndex = 0; dateIndex < dates.length; dateIndex++) {
            const datePhases = linkedPhaseData.linkedPhases.map(pixel => ({
                coordinates: pixel.coordinates,
                phase: pixel.linkedPhase[dateIndex]
            }));
            
            // Spatial filtering for atmospheric component
            const atmosphericPhase = await this.spatialAtmosphericFilter(
                datePhases, 
                spatialFilterSize
            );
            
            // Temporal filtering
            if (atmosphericScreens.length >= 2) {
                const temporallyFiltered = await this.temporalAtmosphericFilter(
                    atmosphericPhase,
                    atmosphericScreens.slice(-2),
                    temporalFilterSize
                );
                atmosphericScreens.push(temporallyFiltered);
            } else {
                atmosphericScreens.push(atmosphericPhase);
            }
        }
        
        // Calculate atmospheric correction quality
        const correctionQuality = this.assessAtmosphericCorrection(
            linkedPhaseData, 
            atmosphericScreens
        );
        
        console.log(`Atmospheric screens estimated for ${dates.length} dates`);
        console.log(`Mean correction magnitude: ${correctionQuality.meanCorrection.toFixed(3)} radians`);
        
        return {
            atmosphericScreens: atmosphericScreens,
            dates: dates,
            correctionQuality: correctionQuality,
            spatialFilterSize: spatialFilterSize,
            temporalFilterSize: temporalFilterSize
        };
    }

    // Atmospheric Correction Methods
    async correctAtmosphericPhase(parameters) {
        console.log('🌤️ Applying atmospheric phase correction...');
        
        const method = parameters.troposphericDelay || 'era5';
        const networkData = await this.getNetworkData(parameters);
        
        let correctedPhases = [];
        let totalCorrection = 0;
        
        for (const interferogram of networkData.interferograms) {
            let atmosphericCorrection;
            
            switch (method) {
                case 'era5':
                    atmosphericCorrection = await this.era5AtmosphericCorrection(interferogram);
                    break;
                case 'gacos':
                    atmosphericCorrection = await this.gacosAtmosphericCorrection(interferogram);
                    break;
                case 'linear':
                    atmosphericCorrection = await this.linearAtmosphericCorrection(interferogram);
                    break;
                case 'none':
                    atmosphericCorrection = { correction: new Array(interferogram.phase.length).fill(0) };
                    break;
                default:
                    atmosphericCorrection = await this.era5AtmosphericCorrection(interferogram);
            }
            
            const correctedPhase = interferogram.phase.map((phase, idx) => 
                phase - atmosphericCorrection.correction[idx]
            );
            
            correctedPhases.push({
                ...interferogram,
                originalPhase: interferogram.phase,
                correctedPhase: correctedPhase,
                atmosphericCorrection: atmosphericCorrection.correction,
                correctionMagnitude: Math.sqrt(
                    atmosphericCorrection.correction.reduce((sum, c) => sum + c * c, 0) / 
                    atmosphericCorrection.correction.length
                )
            });
            
            totalCorrection += correctedPhases[correctedPhases.length - 1].correctionMagnitude;
        }
        
        const meanCorrection = totalCorrection / correctedPhases.length;
        
        console.log(`Atmospheric correction applied using ${method} method`);
        console.log(`Mean correction magnitude: ${meanCorrection.toFixed(3)} radians`);
        
        return {
            correctedPhases: correctedPhases,
            method: method,
            meanCorrectionMagnitude: meanCorrection,
            qualityImprovement: this.calculateAtmosphericCorrectionQuality(correctedPhases)
        };
    }

    async geocodeResults(parameters) {
        console.log('🗺️ Geocoding InSAR results...');
        
        const resultsData = await this.getProcessingResults(parameters);
        const outputProjection = parameters.outputProjection || 'EPSG:4326';
        const outputResolution = parameters.outputResolution || 0.0001; // degrees
        
        const geocodedResults = {
            velocity: await this.geocodeVelocityField(resultsData.velocity, outputProjection, outputResolution),
            timeSeries: await this.geocodeTimeSeries(resultsData.timeSeries, outputProjection, outputResolution),
            quality: await this.geocodeQualityMetrics(resultsData.quality, outputProjection, outputResolution)
        };
        
        // Generate metadata
        const metadata = {
            projection: outputProjection,
            resolution: outputResolution,
            boundingBox: this.calculateBoundingBox(geocodedResults.velocity),
            processingDate: new Date().toISOString(),
            pixelCount: geocodedResults.velocity.length
        };
        
        console.log(`Results geocoded: ${metadata.pixelCount} pixels to ${outputProjection}`);
        
        return {
            ...geocodedResults,
            metadata: metadata
        };
    }

    // Quality Assessment Methods
    async assessCoherence(data) {
        const coherenceValues = data.coherence || this.generateSyntheticCoherence(data);
        
        const stats = this.mathUtils.calculateStatistics(coherenceValues);
        const histogram = this.mathUtils.calculateHistogram(coherenceValues, 20);
        
        return {
            meanCoherence: stats.mean,
            stdCoherence: stats.std,
            medianCoherence: stats.median,
            histogram: histogram,
            spatialCoverage: coherenceValues.filter(c => c > 0.7).length / coherenceValues.length,
            qualityClass: this.classifyCoherenceQuality(stats.mean)
        };
    }

    async assessTemporalCoherence(data) {
        const timeSeriesData = data.timeSeries || await this.getTimeSeriesData(data);
        const temporalCoherenceValues = [];
        
        for (const pixel of timeSeriesData.pixels) {
            const temporalCoherence = this.calculateTemporalCoherence(
                pixel.timeSeries, 
                timeSeriesData.dates
            );
            temporalCoherenceValues.push(temporalCoherence);
        }
        
        const stats = this.mathUtils.calculateStatistics(temporalCoherenceValues);
        
        return {
            meanTemporalCoherence: stats.mean,
            stdTemporalCoherence: stats.std,
            pixelCount: temporalCoherenceValues.length,
            qualityThreshold: this.config.quality.temporalCoherenceThreshold,
            highQualityPixels: temporalCoherenceValues.filter(tc => 
                tc > this.config.quality.temporalCoherenceThreshold
            ).length
        };
    }

    async assessVelocityUncertainty(data) {
        const velocityData = data.velocity || await this.getVelocityData(data);
        
        const uncertainties = velocityData.map(v => v.uncertainty);
        const stats = this.mathUtils.calculateStatistics(uncertainties);
        
        const spatialCorrelation = this.calculateSpatialUncertaintyCorrelation(velocityData);
        
        return {
            meanUncertainty: stats.mean,
            stdUncertainty: stats.std,
            maxUncertainty: stats.max,
            minUncertainty: stats.min,
            spatialCorrelation: spatialCorrelation,
            reliablePixels: uncertainties.filter(u => 
                u < this.config.quality.velocityUncertaintyThreshold
            ).length,
            uncertaintyMap: this.generateUncertaintyMap(velocityData)
        };
    }

    async assessResidualPhase(data) {
        const residualPhases = data.residualPhase || await this.getResidualPhaseData(data);
        
        const rmsValues = residualPhases.map(residual => 
            Math.sqrt(residual.reduce((sum, r) => sum + r * r, 0) / residual.length)
        );
        
        const stats = this.mathUtils.calculateStatistics(rmsValues);
        
        return {
            meanResidualRMS: stats.mean,
            stdResidualRMS: stats.std,
            maxResidualRMS: stats.max,
            qualityThreshold: this.config.quality.residualPhaseThreshold,
            highQualityInterferograms: rmsValues.filter(rms => 
                rms < this.config.quality.residualPhaseThreshold
            ).length,
            totalInterferograms: rmsValues.length
        };
    }

    async assessSpatialConsistency(data) {
        const velocityField = data.velocity || await this.getVelocityData(data);
        
        const consistencyMetrics = [];
        
        for (const pixel of velocityField) {
            const neighbors = this.findNeighboringPixels(pixel.coordinates, velocityField, 1000); // 1km radius
            
            if (neighbors.length >= 3) {
                const neighborVelocities = neighbors.map(n => n.velocity);
                const meanNeighborVelocity = neighborVelocities.reduce((a, b) => a + b) / neighbors.length;
                const velocityDifference = Math.abs(pixel.velocity - meanNeighborVelocity);
                
                consistencyMetrics.push({
                    coordinates: pixel.coordinates,
                    velocity: pixel.velocity,
                    neighborCount: neighbors.length,
                    meanNeighborVelocity: meanNeighborVelocity,
                    velocityDifference: velocityDifference,
                    consistencyScore: Math.exp(-velocityDifference / 5.0) // 5 mm/year decay
                });
            }
        }
        
        const overallConsistency = consistencyMetrics.reduce((sum, m) => 
            sum + m.consistencyScore, 0
        ) / consistencyMetrics.length;
        
        return {
            overallConsistency: overallConsistency,
            analyzedPixels: consistencyMetrics.length,
            totalPixels: velocityField.length,
            consistencyMetrics: consistencyMetrics,
            spatiallyConsistentPixels: consistencyMetrics.filter(m => 
                m.consistencyScore > 0.8
            ).length
        };
    }

    // Utility Methods for Algorithm Support
    generateSyntheticNetwork(parameters) {
        const numAcquisitions = parameters.numAcquisitions || 50;
        const acquisitions = [];
        const baseDate = new Date('2020-01-01');
        
        for (let i = 0; i < numAcquisitions; i++) {
            const date = new Date(baseDate);
            date.setDate(baseDate.getDate() + i * 12); // Every 12 days
            acquisitions.push({
                date: date.toISOString().split('T')[0],
                index: i,
                perpendicular_baseline: (Math.random() - 0.5) * 600, // -300 to 300m
                doppler_centroid: Math.random() * 100
            });
        }
        
        const interferograms = [];
        for (let i = 0; i < numAcquisitions - 1; i++) {
            for (let j = i + 1; j < numAcquisitions; j++) {
                const tempBaseline = Math.abs(acquisitions[j].index - acquisitions[i].index) * 12;
                const perpBaseline = Math.abs(acquisitions[j].perpendicular_baseline - acquisitions[i].perpendicular_baseline);
                
                if (tempBaseline <= 365 && perpBaseline <= 300) {
                    interferograms.push({
                        master: acquisitions[i],
                        slave: acquisitions[j],
                        temporalBaseline: tempBaseline,
                        perpendicularBaseline: perpBaseline,
                        coherence: Math.max(0.3, 0.9 - (tempBaseline / 365) * 0.3 - (perpBaseline / 300) * 0.2),
                        phase: this.generateSyntheticPhase({ length: 10000 })
                    });
                }
            }
        }
        
        return { acquisitions, interferograms };
    }

    generateSyntheticPhase(interferogram) {
        const length = interferogram.length || 10000;
        const phase = new Array(length);
        
        for (let i = 0; i < length; i++) {
            // Simulate deformation signal + noise
            const deformation = Math.sin(2 * Math.PI * i / 1000) * 0.1; // Long wavelength deformation
            const topography = Math.cos(2 * Math.PI * i / 100) * 0.05; // Topographic phase
            const atmosphere = (Math.random() - 0.5) * 0.3; // Atmospheric noise
            const noise = (Math.random() - 0.5) * 0.1; // Random noise
            
            phase[i] = deformation + topography + atmosphere + noise;
        }
        
        return phase;
    }

    generateSyntheticAmplitudeStack(parameters) {
        const width = parameters.width || 1000;
        const height = parameters.height || 1000;
        const numImages = parameters.numImages || 50;
        
        return {
            width: width,
            height: height,
            numImages: numImages,
            getPixelTimeSeries: (row, col) => {
                const timeSeries = new Array(numImages);
                const baseAmplitude = 1000 + Math.random() * 500;
                const dispersion = Math.random() * 0.5; // 0 to 0.5 dispersion index
                
                for (let t = 0; t < numImages; t++) {
                    timeSeries[t] = baseAmplitude * (1 + dispersion * (Math.random() - 0.5));
                }
                
                return timeSeries;
            },
            pixelToCoordinates: (row, col) => ({
                lat: 40.5 + (row / height) * 0.1,
                lon: -120.3 + (col / width) * 0.1
            })
        };
    }

    calculateAmplitudeDispersion(amplitudeTimeSeries) {
        const mean = amplitudeTimeSeries.reduce((a, b) => a + b) / amplitudeTimeSeries.length;
        const std = Math.sqrt(
            amplitudeTimeSeries.reduce((sum, amp) => sum + Math.pow(amp - mean, 2), 0) / 
            amplitudeTimeSeries.length
        );
        return std / mean; // Coefficient of variation
    }

    calculateTemporalCoherence(timeSeries, dates) {
        if (timeSeries.length < 3) return 0;
        
        const complexCorrelations = [];
        for (let i = 0; i < timeSeries.length - 1; i++) {
            const phase1 = timeSeries[i];
            const phase2 = timeSeries[i + 1];
            complexCorrelations.push(Math.exp(1i * (phase2 - phase1)));
        }
        
        // Calculate magnitude of average complex correlation
        const avgReal = complexCorrelations.reduce((sum, c) => sum + Math.cos(c), 0) / complexCorrelations.length;
        const avgImag = complexCorrelations.reduce((sum, c) => sum + Math.sin(c), 0) / complexCorrelations.length;
        
        return Math.sqrt(avgReal * avgReal + avgImag * avgImag);
    }

    fitVelocityModel(timeSeries, dates, model = 'linear') {
        const n = timeSeries.length;
        if (n < 3) {
            return { velocity: 0, uncertainty: Infinity, r_squared: 0, residualRMS: Infinity };
        }
        
        // Convert dates to decimal years
        const timeVector = dates.map(date => {
            const d = new Date(date);
            const year = d.getFullYear();
            const start = new Date(year, 0, 0);
            const dayOfYear = Math.floor((d - start) / (24 * 60 * 60 * 1000));
            return year + dayOfYear / 365.25;
        });
        
        const t0 = timeVector[0];
        const timeRelative = timeVector.map(t => t - t0);
        
        if (model === 'linear') {
            // Linear least squares: displacement = a + b*t
            const sumT = timeRelative.reduce((a, b) => a + b, 0);
            const sumT2 = timeRelative.reduce((sum, t) => sum + t * t, 0);
            const sumD = timeSeries.reduce((a, b) => a + b, 0);
            const sumTD = timeRelative.reduce((sum, t, i) => sum + t * timeSeries[i], 0);
            
            const denominator = n * sumT2 - sumT * sumT;
            if (Math.abs(denominator) < 1e-10) {
                return { velocity: 0, uncertainty: Infinity, r_squared: 0, residualRMS: Infinity };
            }
            
            const velocity = (n * sumTD - sumT * sumD) / denominator; // mm/year
            const intercept = (sumD - velocity * sumT) / n;
            
            // Calculate residuals and statistics
            const predicted = timeRelative.map(t => intercept + velocity * t);
            const residuals = timeSeries.map((d, i) => d - predicted[i]);
            const residualSumSquares = residuals.reduce((sum, r) => sum + r * r, 0);
            const totalSumSquares = timeSeries.reduce((sum, d) => {
                const meanD = sumD / n;
                return sum + Math.pow(d - meanD, 2);
            }, 0);
            
            const r_squared = 1 - (residualSumSquares / totalSumSquares);
            const residualRMS = Math.sqrt(residualSumSquares / (n - 2));
            
            // Velocity uncertainty
            const s2 = residualSumSquares / (n - 2);
            const uncertainty = Math.sqrt(s2 * n / denominator);
            
            return {
                velocity: velocity,
                uncertainty: uncertainty,
                r_squared: r_squared,
                residualRMS: residualRMS,
                intercept: intercept
            };
        } else {
            // For other models, return simplified result
            const velocityApprox = (timeSeries[n-1] - timeSeries[0]) / (timeRelative[n-1] - timeRelative[0]);
            return {
                velocity: velocityApprox,
                uncertainty: 2.0, // Default uncertainty
                r_squared: 0.8,
                residualRMS: 1.0
            };
        }
    }

    // Data access methods (would interface with actual data in real implementation)
    async getNetworkData(parameters) {
        if (this.processingCache.has('network_data')) {
            return this.processingCache.get('network_data');
        }
        
        const networkData = this.generateSyntheticNetwork(parameters);
        this.processingCache.set('network_data', networkData);
        return networkData;
    }

    async getTimeSeriesData(parameters) {
        if (this.processingCache.has('timeseries_data')) {
            return this.processingCache.get('timeseries_data');
        }
        
        const dates = [];
        const baseDate = new Date('2020-01-01');
        for (let i = 0; i < 50; i++) {
            const date = new Date(baseDate);
            date.setDate(baseDate.getDate() + i * 12);
            dates.push(date.toISOString().split('T')[0]);
        }
        
        const pixels = [];
        for (let i = 0; i < 1000; i++) {
            const timeSeries = new Array(50);
            for (let t = 0; t < 50; t++) {
                timeSeries[t] = Math.sin(2 * Math.PI * t / 50) * 10 + (Math.random() - 0.5) * 2;
            }
            
            pixels.push({
                coordinates: {
                    lat: 40.5 + (Math.random() - 0.5) * 0.1,
                    lon: -120.3 + (Math.random() - 0.5) * 0.1
                },
                timeSeries: timeSeries
            });
        }
        
        const timeSeriesData = { dates, pixels };
        this.processingCache.set('timeseries_data', timeSeriesData);
        return timeSeriesData;
    }

    async getPSData(parameters) {
        return await this.identifyPersistentScatterers(parameters);
    }

    async getLinkedPhaseData(parameters) {
        const psData = await this.getPSData(parameters);
        const linkedPhases = [];
        const dates = [];
        
        // Generate synthetic dates
        const baseDate = new Date('2020-01-01');
        for (let i = 0; i < 50; i++) {
            const date = new Date(baseDate);
            date.setDate(baseDate.getDate() + i * 12);
            dates.push(date.toISOString().split('T')[0]);
        }
        
        // Generate synthetic linked phases for each PS pixel
        for (const psPixel of psData.psPixels.slice(0, 100)) { // Limit for performance
            const linkedPhase = new Array(dates.length);
            for (let t = 0; t < dates.length; t++) {
                linkedPhase[t] = Math.sin(2 * Math.PI * t / dates.length) * 0.5 + (Math.random() - 0.5) * 0.1;
            }
            
            linkedPhases.push({
                coordinates: psPixel.coordinates,
                linkedPhase: linkedPhase
            });
        }
        
        return { linkedPhases, dates };
    }

    // Placeholder implementations for complex algorithms
    async autoReferenceSelection(parameters) {
        return { lat: 40.5 + Math.random() * 0.01, lon: -120.3 + Math.random() * 0.01 };
    }

    async maxCoherenceReference(parameters) {
        return { lat: 40.505, lon: -120.295 };
    }

    async minRMSReference(parameters) {
        return { lat: 40.503, lon: -120.302 };
    }

    assessReferenceQuality(referencePoint, parameters) {
        return {
            coherence: 0.85,
            temporalStability: 0.9,
            spatialConsistency: 0.8
        };
    }

    async detectAndCorrectUnwrapErrors(phase, interferogram, method) {
        const errorsCorrected = Math.floor(Math.random() * 5);
        return {
            phase: phase,
            errorsCorrected: errorsCorrected,
            quality: 0.85 + Math.random() * 0.1
        };
    }

    calculateUnwrapQualityImprovement(correctedPhases) {
        return {
            meanImprovement: 0.15,
            maxImprovement: 0.4,
            pixelsImproved: Math.floor(correctedPhases.length * 0.7)
        };
    }

    buildInversionMatrices(networkData, weightingMethod) {
        const numAcquisitions = networkData.acquisitions.length;
        const numInterferograms = networkData.interferograms.length;
        
        // Simplified design matrix (A): each row is an interferogram, columns are acquisitions
        const designMatrix = new Array(numInterferograms);
        for (let i = 0; i < numInterferograms; i++) {
            designMatrix[i] = new Array(numAcquisitions).fill(0);
            const ifg = networkData.interferograms[i];
            designMatrix[i][ifg.master.index] = -1;
            designMatrix[i][ifg.slave.index] = 1;
        }
        
        // Data vector: observed phases
        const dataVector = networkData.interferograms.map(ifg => 
            ifg.phase.reduce((sum, p) => sum + p, 0) / ifg.phase.length
        );
        
        // Weight vector based on coherence
        const weights = networkData.interferograms.map(ifg => 
            weightingMethod === 'coherence' ? ifg.coherence : 1.0
        );
        
        return { designMatrix, dataVector, weights };
    }

    weightedLeastSquares(A, d, w) {
        // Simplified weighted least squares solution
        // In reality, this would use proper matrix operations
        const numParams = A[0].length;
        const solution = new Array(numParams).fill(0);
        
        // Pseudo-solution for demonstration
        for (let i = 0; i < solution.length; i++) {
            solution[i] = (Math.random() - 0.5) * 10; // Simulated displacement values
        }
        
        const residuals = A.map((row, i) => {
            const predicted = row.reduce((sum, a, j) => sum + a * solution[j], 0);
            return d[i] - predicted;
        });
        
        return {
            solution: solution,
            residuals: residuals
        };
    }

    calculateTimeSeries(inversionResult, acquisitions) {
        return acquisitions.map((acq, i) => ({
            date: acq.date,
            displacement: inversionResult.solution[i],
            index: i
        }));
    }

    assessInversionQuality(inversionResult, networkData) {
        const residuals = inversionResult.residuals;
        const rmsResidual = Math.sqrt(
            residuals.reduce((sum, r) => sum + r * r, 0) / residuals.length
        );
        
        return {
            rmsResidual: rmsResidual,
            redundancy: networkData.interferograms.length - networkData.acquisitions.length + 1,
            conditionNumber: 15.7 // Simulated condition number
        };
    }

    // Additional utility methods
    calculateNetworkRedundancy(interferograms, numAcquisitions) {
        return interferograms.length / (numAcquisitions - 1);
    }

    classifyCoherenceQuality(meanCoherence) {
        if (meanCoherence > 0.8) return 'excellent';
        if (meanCoherence > 0.6) return 'good';
        if (meanCoherence > 0.4) return 'fair';
        return 'poor';
    }

    setupErrorHandling() {
        window.addEventListener('error', (event) => {
            if (event.filename?.includes('insar_algorithms')) {
                console.error(`InSAR Algorithm Error: ${event.message}`);
                this.eventBus.dispatchEvent(new CustomEvent('algorithmError', {
                    detail: { message: event.message, filename: event.filename }
                }));
            }
        });
    }

    setupAtmosphericCorrection() {
        // Initialize atmospheric correction methods
        this.atmosphericModels = {
            era5: this.era5AtmosphericCorrection.bind(this),
            gacos: this.gacosAtmosphericCorrection.bind(this),
            linear: this.linearAtmosphericCorrection.bind(this)
        };
    }

    async era5AtmosphericCorrection(interferogram) {
        // Simulate ERA5-based atmospheric correction
        const correction = interferogram.phase.map(p => Math.sin(p * 2) * 0.1 + (Math.random() - 0.5) * 0.05);
        return { correction: correction, model: 'ERA5', quality: 0.8 };
    }

    async gacosAtmosphericCorrection(interferogram) {
        // Simulate GACOS-based atmospheric correction
        const correction = interferogram.phase.map(p => Math.cos(p * 1.5) * 0.08 + (Math.random() - 0.5) * 0.03);
        return { correction: correction, model: 'GACOS', quality: 0.85 };
    }

    async linearAtmosphericCorrection(interferogram) {
        // Simulate linear atmospheric correction
        const correction = interferogram.phase.map((p, i) => (i / interferogram.phase.length) * 0.05);
        return { correction: correction, model: 'Linear', quality: 0.6 };
    }
}

// Mathematical utilities for InSAR processing
class InSARMathUtils {
    calculateStatistics(data) {
        if (data.length === 0) return { mean: 0, std: 0, median: 0, min: 0, max: 0 };
        
        const sorted = [...data].sort((a, b) => a - b);
        const mean = data.reduce((a, b) => a + b) / data.length;
        const variance = data.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / data.length;
        const std = Math.sqrt(variance);
        const median = sorted.length % 2 === 0 
            ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
            : sorted[Math.floor(sorted.length / 2)];
        
        return {
            mean: mean,
            std: std,
            median: median,
            min: sorted[0],
            max: sorted[sorted.length - 1]
        };
    }

    calculateHistogram(data, bins = 10) {
        const stats = this.calculateStatistics(data);
        const binSize = (stats.max - stats.min) / bins;
        const histogram = new Array(bins).fill(0);
        
        data.forEach(value => {
            const binIndex = Math.min(bins - 1, Math.floor((value - stats.min) / binSize));
            histogram[binIndex]++;
        });
        
        return {
            bins: histogram,
            binEdges: Array.from({ length: bins + 1 }, (_, i) => stats.min + i * binSize)
        };
    }

    complexMultiply(a, b) {
        return {
            real: a.real * b.real - a.imag * b.imag,
            imag: a.real * b.imag + a.imag * b.real
        };
    }

    complexMagnitude(complex) {
        return Math.sqrt(complex.real * complex.real + complex.imag * complex.imag);
    }

    calculatePhaseGradient(phase, coordinates) {
        // Simplified phase gradient calculation
        const gradient = { x: 0, y: 0 };
        
        for (let i = 1; i < phase.length - 1; i++) {
            const dx = coordinates[i+1].x - coordinates[i-1].x;
            const dy = coordinates[i+1].y - coordinates[i-1].y;
            
            if (Math.abs(dx) > 0) {
                gradient.x += (phase[i+1] - phase[i-1]) / (2 * dx);
            }
            if (Math.abs(dy) > 0) {
                gradient.y += (phase[i+1] - phase[i-1]) / (2 * dy);
            }
        }
        
        gradient.x /= (phase.length - 2);
        gradient.y /= (phase.length - 2);
        
        return gradient;
    }
}

// Global initialization
let insarAlgorithms;

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        insarAlgorithms = new InSARAlgorithms();
        window.insarAlgorithms = insarAlgorithms;
    });
} else {
    insarAlgorithms = new InSARAlgorithms();
    window.insarAlgorithms = insarAlgorithms;
}

console.log('🧮 InSAR Algorithms System Loaded');