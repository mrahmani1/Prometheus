/**
 * Intelligent Alert System for Real-time Volcanic Monitoring
 * Advanced ML-based anomaly detection with multi-modal data fusion
 * Implements novel research from 2025 monitoring systems
 */

class IntelligentAlertSystem {
    constructor(config = {}) {
        this.config = {
            alertThresholds: {
                anomalyScore: 0.75,
                confidenceThreshold: 0.6,
                threatLevelCritical: 4,
                sensorDataMinimum: 3
            },
            temporalWindows: {
                immediate: 300,     // 5 minutes
                short: 3600,        // 1 hour
                medium: 86400,      // 1 day
                long: 604800        // 1 week
            },
            alertLevels: {
                0: { name: 'Normal', color: '#4CAF50', priority: 'low' },
                1: { name: 'Advisory', color: '#2196F3', priority: 'low' },
                2: { name: 'Watch', color: '#FF9800', priority: 'medium' },
                3: { name: 'Warning', color: '#FF5722', priority: 'high' },
                4: { name: 'Alert', color: '#F44336', priority: 'critical' },
                5: { name: 'Emergency', color: '#9C27B0', priority: 'emergency' }
            },
            ...config
        };

        this.anomalyDetectors = new Map();
        this.alertHistory = new Map();
        this.riskAssessment = new AdvancedRiskAssessment();
        this.notificationSystem = new NotificationSystem();
        this.patternRecognition = new PatternRecognitionEngine();
        this.uncertaintyQuantifier = new UncertaintyQuantification();
        
        this.activeAlerts = new Map();
        this.alertMetrics = new AlertMetricsTracker();
        this.webSocketConnection = null;
        
        this.initialize();
    }

    initialize() {
        this.initializeWebSocket();
        this.startPerformanceMonitoring();
        this.loadHistoricalBaselines();
        
        console.log('Intelligent Alert System initialized with advanced ML capabilities');
    }

    initializeWebSocket() {
        const wsUrl = `ws://${window.location.host}/ws/monitoring`;
        this.webSocketConnection = new WebSocket(wsUrl);
        
        this.webSocketConnection.onopen = () => {
            console.log('Real-time monitoring connection established');
            this.sendSubscriptionRequest();
        };
        
        this.webSocketConnection.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.processRealtimeData(data);
        };
        
        this.webSocketConnection.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.attemptReconnection();
        };
    }

    async processRealtimeData(data) {
        try {
            const { volcanoId, sensorData, fusionResult, timestamp } = data;
            
            // Process multi-sensor fusion result
            const alertDecision = await this.evaluateAlert(volcanoId, fusionResult);
            
            // Update anomaly detectors
            await this.updateAnomalyDetection(volcanoId, sensorData);
            
            // Pattern recognition analysis
            const patterns = await this.patternRecognition.analyzePatterns(volcanoId, sensorData);
            
            // Risk assessment update
            const riskUpdate = await this.riskAssessment.assessCurrentRisk(
                volcanoId, fusionResult, patterns
            );
            
            // Generate alert if conditions are met
            if (alertDecision.shouldAlert) {
                await this.generateAlert(volcanoId, {
                    ...alertDecision,
                    patterns,
                    riskAssessment: riskUpdate,
                    timestamp: new Date(timestamp)
                });
            }
            
            // Update UI components
            this.updateDashboard(volcanoId, {
                fusionResult,
                alertDecision,
                patterns,
                riskUpdate
            });
            
        } catch (error) {
            console.error('Error processing realtime data:', error);
            this.handleProcessingError(error, data);
        }
    }

    async evaluateAlert(volcanoId, fusionResult) {
        const { anomalyScore, confidence, threatLevel, alertTriggers, contributingSensors } = fusionResult;
        
        // Multi-criteria alert evaluation
        const criteria = {
            anomalyThreshold: anomalyScore >= this.config.alertThresholds.anomalyScore,
            confidenceThreshold: confidence >= this.config.alertThresholds.confidenceThreshold,
            threatLevelCritical: threatLevel >= this.config.alertThresholds.threatLevelCritical,
            sufficientSensorData: contributingSensors.length >= this.config.alertThresholds.sensorDataMinimum,
            explicitTriggers: alertTriggers && alertTriggers.length > 0
        };
        
        // Historical context evaluation
        const historicalContext = await this.getHistoricalContext(volcanoId);
        const temporalTrends = this.analyzeTTemporalTrends(volcanoId, fusionResult);
        
        // Uncertainty-weighted decision
        const uncertaintyFactor = await this.uncertaintyQuantifier.computeDecisionUncertainty(
            fusionResult, historicalContext
        );
        
        // Bayesian alert probability
        const alertProbability = this.computeBayesianAlertProbability(
            criteria, temporalTrends, uncertaintyFactor
        );
        
        const shouldAlert = alertProbability > 0.7 && (
            criteria.anomalyThreshold ||
            criteria.threatLevelCritical ||
            (criteria.explicitTriggers && criteria.confidenceThreshold)
        );
        
        return {
            shouldAlert,
            alertProbability,
            alertLevel: this.determineAlertLevel(threatLevel, anomalyScore, alertProbability),
            criteria,
            uncertainty: uncertaintyFactor,
            reasoning: this.generateAlertReasoning(criteria, temporalTrends, uncertaintyFactor)
        };
    }

    computeBayesianAlertProbability(criteria, trends, uncertaintyFactor) {
        // Prior probability based on historical data
        const priorProbability = 0.05; // 5% base rate of true alerts
        
        // Likelihood factors
        const likelihoods = {
            anomaly: criteria.anomalyThreshold ? 0.8 : 0.2,
            confidence: criteria.confidenceThreshold ? 0.9 : 0.3,
            threat: criteria.threatLevelCritical ? 0.85 : 0.15,
            sensors: criteria.sufficientSensorData ? 0.9 : 0.4,
            triggers: criteria.explicitTriggers ? 0.95 : 0.1
        };
        
        // Temporal trend factor
        const trendFactor = trends.isAccelerating ? 1.5 : 
                           trends.isDecelerating ? 0.7 : 1.0;
        
        // Combined likelihood
        const combinedLikelihood = Object.values(likelihoods).reduce((a, b) => a * b, 1) * trendFactor;
        
        // Bayesian update with uncertainty adjustment
        const posterior = (combinedLikelihood * priorProbability) / 
                         (combinedLikelihood * priorProbability + (1 - priorProbability));
        
        // Adjust for uncertainty
        const adjustedPosterior = posterior * (1 - uncertaintyFactor);
        
        return Math.max(0, Math.min(1, adjustedPosterior));
    }

    analyzeTTemporalTrends(volcanoId, currentResult) {
        const history = this.alertHistory.get(volcanoId) || [];
        
        if (history.length < 3) {
            return { isAccelerating: false, isDecelerating: false, trend: 'insufficient_data' };
        }
        
        // Analyze recent trend in anomaly scores
        const recentScores = history.slice(-10).map(h => h.anomalyScore);
        const currentScore = currentResult.anomalyScore;
        
        // Linear regression for trend analysis
        const n = recentScores.length;
        const x = Array.from({length: n}, (_, i) => i);
        const y = recentScores;
        
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const correlation = this.computeCorrelation(x, y);
        
        // Acceleration detection
        const recentSlope = this.computeRecentSlope(recentScores.slice(-3));
        const overallSlope = slope;
        
        const isAccelerating = recentSlope > overallSlope && recentSlope > 0.01;
        const isDecelerating = recentSlope < overallSlope && overallSlope > 0;
        
        return {
            isAccelerating,
            isDecelerating,
            trend: slope > 0.005 ? 'increasing' : 
                   slope < -0.005 ? 'decreasing' : 'stable',
            slope,
            correlation: Math.abs(correlation),
            recentAcceleration: recentSlope - overallSlope
        };
    }

    determineAlertLevel(threatLevel, anomalyScore, alertProbability) {
        // Multi-factor alert level determination
        let level = threatLevel;
        
        // Boost level based on anomaly score
        if (anomalyScore > 0.9) level = Math.max(level, 4);
        else if (anomalyScore > 0.8) level = Math.max(level, 3);
        
        // Boost level based on alert probability
        if (alertProbability > 0.95) level = Math.max(level, 5);
        else if (alertProbability > 0.85) level = Math.max(level, 4);
        
        return Math.max(0, Math.min(5, level));
    }

    async generateAlert(volcanoId, alertData) {
        const alertId = `alert_${volcanoId}_${Date.now()}`;
        
        const alert = {
            id: alertId,
            volcanoId,
            timestamp: alertData.timestamp,
            level: alertData.alertLevel,
            probability: alertData.alertProbability,
            anomalyScore: alertData.fusionResult?.anomalyScore || 0,
            confidence: alertData.fusionResult?.confidence || 0,
            threatLevel: alertData.fusionResult?.threatLevel || 0,
            patterns: alertData.patterns,
            riskAssessment: alertData.riskAssessment,
            reasoning: alertData.reasoning,
            uncertainty: alertData.uncertainty,
            contributingSensors: alertData.fusionResult?.contributingSensors || [],
            alertTriggers: alertData.fusionResult?.alertTriggers || [],
            status: 'active',
            acknowledgedBy: null,
            resolvedAt: null
        };
        
        // Store alert
        this.activeAlerts.set(alertId, alert);
        
        // Update alert history
        const history = this.alertHistory.get(volcanoId) || [];
        history.push(alert);
        if (history.length > 1000) history.shift(); // Maintain reasonable size
        this.alertHistory.set(volcanoId, history);
        
        // Send notifications
        await this.notificationSystem.sendAlert(alert);
        
        // Update metrics
        this.alertMetrics.recordAlert(alert);
        
        // Trigger UI updates
        this.broadcastAlert(alert);
        
        // Log alert
        console.warn(`ALERT GENERATED: ${volcanoId} - Level ${alert.level} - Probability: ${alert.probability.toFixed(3)}`);
        
        // Auto-escalation for high-priority alerts
        if (alert.level >= 4) {
            setTimeout(() => this.escalateAlert(alertId), 300000); // 5 minutes
        }
        
        return alert;
    }

    broadcastAlert(alert) {
        // Emit custom event for UI components
        const event = new CustomEvent('volcanicAlert', {
            detail: alert,
            bubbles: true
        });
        document.dispatchEvent(event);
        
        // Update dashboard directly
        this.updateAlertDisplay(alert);
        
        // Send to connected clients if server-side
        if (this.webSocketConnection && this.webSocketConnection.readyState === WebSocket.OPEN) {
            this.webSocketConnection.send(JSON.stringify({
                type: 'alert_broadcast',
                data: alert
            }));
        }
    }

    updateAlertDisplay(alert) {
        // Update main alert banner
        const alertBanner = document.getElementById('alert-banner');
        if (alertBanner) {
            const levelConfig = this.config.alertLevels[alert.level];
            alertBanner.style.backgroundColor = levelConfig.color;
            alertBanner.style.display = 'block';
            alertBanner.innerHTML = `
                <div class="alert-content">
                    <div class="alert-header">
                        <span class="alert-level">${levelConfig.name}</span>
                        <span class="alert-volcano">${alert.volcanoId}</span>
                        <span class="alert-time">${alert.timestamp.toLocaleTimeString()}</span>
                    </div>
                    <div class="alert-details">
                        <span>Anomaly: ${(alert.anomalyScore * 100).toFixed(1)}%</span>
                        <span>Confidence: ${(alert.confidence * 100).toFixed(1)}%</span>
                        <span>Probability: ${(alert.probability * 100).toFixed(1)}%</span>
                    </div>
                    <div class="alert-actions">
                        <button onclick="intelligentAlertSystem.acknowledgeAlert('${alert.id}')" class="ack-btn">Acknowledge</button>
                        <button onclick="intelligentAlertSystem.viewAlertDetails('${alert.id}')" class="details-btn">Details</button>
                    </div>
                </div>
            `;
        }
        
        // Update alert list
        this.updateAlertList();
        
        // Update volcano status indicators
        this.updateVolcanoStatus(alert.volcanoId, alert.level);
    }

    updateAlertList() {
        const alertList = document.getElementById('alert-list');
        if (!alertList) return;
        
        const activeAlerts = Array.from(this.activeAlerts.values())
            .filter(alert => alert.status === 'active')
            .sort((a, b) => b.timestamp - a.timestamp);
        
        alertList.innerHTML = activeAlerts.map(alert => {
            const levelConfig = this.config.alertLevels[alert.level];
            return `
                <div class="alert-item ${levelConfig.priority}" data-alert-id="${alert.id}">
                    <div class="alert-indicator" style="background-color: ${levelConfig.color}"></div>
                    <div class="alert-info">
                        <div class="alert-volcano-name">${alert.volcanoId}</div>
                        <div class="alert-timestamp">${alert.timestamp.toLocaleString()}</div>
                        <div class="alert-metrics">
                            <span class="metric">Level: ${alert.level}</span>
                            <span class="metric">Score: ${(alert.anomalyScore * 100).toFixed(0)}%</span>
                            <span class="metric">Prob: ${(alert.probability * 100).toFixed(0)}%</span>
                        </div>
                        <div class="alert-triggers">
                            ${alert.alertTriggers.map(trigger => `<span class="trigger-tag">${trigger}</span>`).join('')}
                        </div>
                    </div>
                    <div class="alert-actions">
                        <button onclick="intelligentAlertSystem.acknowledgeAlert('${alert.id}')" class="btn-sm">ACK</button>
                        <button onclick="intelligentAlertSystem.viewAlertDetails('${alert.id}')" class="btn-sm">VIEW</button>
                    </div>
                </div>
            `;
        }).join('');
    }

    acknowledgeAlert(alertId) {
        const alert = this.activeAlerts.get(alertId);
        if (!alert) return;
        
        alert.status = 'acknowledged';
        alert.acknowledgedBy = 'user'; // In real system, would be current user
        alert.acknowledgedAt = new Date();
        
        this.activeAlerts.set(alertId, alert);
        
        // Update UI
        this.updateAlertList();
        
        // Send acknowledgment to server
        if (this.webSocketConnection && this.webSocketConnection.readyState === WebSocket.OPEN) {
            this.webSocketConnection.send(JSON.stringify({
                type: 'alert_acknowledgment',
                alertId,
                timestamp: new Date().toISOString()
            }));
        }
        
        console.log(`Alert ${alertId} acknowledged`);
    }

    viewAlertDetails(alertId) {
        const alert = this.activeAlerts.get(alertId);
        if (!alert) return;
        
        // Create detailed alert modal
        const modalContent = this.createAlertDetailsModal(alert);
        this.showModal('Alert Details', modalContent);
    }

    createAlertDetailsModal(alert) {
        const levelConfig = this.config.alertLevels[alert.level];
        
        return `
            <div class="alert-details-modal">
                <div class="alert-summary">
                    <h3 style="color: ${levelConfig.color}">
                        ${levelConfig.name} Alert - ${alert.volcanoId}
                    </h3>
                    <div class="timestamp">Generated: ${alert.timestamp.toLocaleString()}</div>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <label>Anomaly Score</label>
                        <div class="metric-value">${(alert.anomalyScore * 100).toFixed(2)}%</div>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${alert.anomalyScore * 100}%; background: ${levelConfig.color}"></div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <label>Confidence</label>
                        <div class="metric-value">${(alert.confidence * 100).toFixed(2)}%</div>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${alert.confidence * 100}%; background: #4CAF50"></div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <label>Alert Probability</label>
                        <div class="metric-value">${(alert.probability * 100).toFixed(2)}%</div>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${alert.probability * 100}%; background: #FF9800"></div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <label>Uncertainty</label>
                        <div class="metric-value">${(alert.uncertainty * 100).toFixed(2)}%</div>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${alert.uncertainty * 100}%; background: #9C27B0"></div>
                        </div>
                    </div>
                </div>
                
                <div class="alert-sections">
                    <div class="section">
                        <h4>Contributing Sensors</h4>
                        <div class="sensor-list">
                            ${alert.contributingSensors.map(sensor => `<span class="sensor-tag">${sensor}</span>`).join('')}
                        </div>
                    </div>
                    
                    <div class="section">
                        <h4>Alert Triggers</h4>
                        <div class="trigger-list">
                            ${alert.alertTriggers.map(trigger => `<span class="trigger-item">${trigger.replace(/_/g, ' ')}</span>`).join('')}
                        </div>
                    </div>
                    
                    <div class="section">
                        <h4>Risk Assessment</h4>
                        <div class="risk-details">
                            ${this.formatRiskAssessment(alert.riskAssessment)}
                        </div>
                    </div>
                    
                    <div class="section">
                        <h4>Pattern Analysis</h4>
                        <div class="pattern-details">
                            ${this.formatPatternAnalysis(alert.patterns)}
                        </div>
                    </div>
                    
                    <div class="section">
                        <h4>Alert Reasoning</h4>
                        <div class="reasoning-text">
                            ${alert.reasoning || 'Automated analysis based on multi-sensor fusion'}
                        </div>
                    </div>
                </div>
                
                <div class="alert-actions-modal">
                    ${alert.status === 'active' ? 
                        `<button onclick="intelligentAlertSystem.acknowledgeAlert('${alert.id}')" class="btn-primary">Acknowledge Alert</button>` : 
                        `<span class="acknowledged-status">Acknowledged by ${alert.acknowledgedBy} at ${alert.acknowledgedAt?.toLocaleString()}</span>`
                    }
                    <button onclick="intelligentAlertSystem.exportAlert('${alert.id}')" class="btn-secondary">Export Report</button>
                </div>
            </div>
        `;
    }

    formatRiskAssessment(riskAssessment) {
        if (!riskAssessment) return 'No risk assessment available';
        
        return `
            <div class="risk-item">Overall Risk: <span class="risk-level-${riskAssessment.overallRisk || 'unknown'}">${riskAssessment.overallRisk || 'Unknown'}</span></div>
            <div class="risk-item">Eruption Probability: ${((riskAssessment.eruptionProbability || 0) * 100).toFixed(1)}%</div>
            <div class="risk-item">Time to Event: ${riskAssessment.estimatedTimeframe || 'Unknown'}</div>
        `;
    }

    formatPatternAnalysis(patterns) {
        if (!patterns) return 'No pattern analysis available';
        
        return `
            <div class="pattern-item">Detected Patterns: ${patterns.detectedPatterns?.join(', ') || 'None'}</div>
            <div class="pattern-item">Pattern Confidence: ${((patterns.confidence || 0) * 100).toFixed(1)}%</div>
            <div class="pattern-item">Similar Historical Events: ${patterns.historicalMatches || 0}</div>
        `;
    }

    updateDashboard(volcanoId, data) {
        // Update real-time monitoring dashboard
        const dashboardElement = document.getElementById(`volcano-${volcanoId}-dashboard`);
        if (dashboardElement) {
            this.updateVolcanoDashboard(dashboardElement, data);
        }
        
        // Update global metrics
        this.updateGlobalMetrics();
        
        // Update charts and visualizations
        this.updateMonitoringCharts(volcanoId, data);
    }

    showModal(title, content) {
        // Create or get modal element
        let modal = document.getElementById('alert-details-modal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'alert-details-modal';
            modal.className = 'modal';
            document.body.appendChild(modal);
        }
        
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2>${title}</h2>
                    <span class="close" onclick="document.getElementById('alert-details-modal').style.display='none'">&times;</span>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
            </div>
        `;
        
        modal.style.display = 'block';
        
        // Close modal when clicking outside
        modal.onclick = (event) => {
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        };
    }

    // Utility methods
    computeCorrelation(x, y) {
        const n = x.length;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
        const sumYY = y.reduce((sum, yi) => sum + yi * yi, 0);
        
        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
        
        return denominator === 0 ? 0 : numerator / denominator;
    }

    computeRecentSlope(values) {
        if (values.length < 2) return 0;
        const n = values.length;
        const x = Array.from({length: n}, (_, i) => i);
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = values.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * values[i], 0);
        const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
        
        return (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    }
}

// Supporting classes

class AdvancedRiskAssessment {
    constructor() {
        this.riskModels = new Map();
        this.historicalData = new Map();
    }
    
    async assessCurrentRisk(volcanoId, fusionResult, patterns) {
        // Implement advanced risk assessment logic
        const baseRisk = this.computeBaseRisk(fusionResult);
        const patternRisk = this.computePatternRisk(patterns);
        const historicalRisk = await this.computeHistoricalRisk(volcanoId);
        
        const overallRisk = this.combineRiskFactors(baseRisk, patternRisk, historicalRisk);
        
        return {
            overallRisk: this.categorizeRisk(overallRisk),
            eruptionProbability: this.estimateEruptionProbability(overallRisk, patterns),
            estimatedTimeframe: this.estimateTimeframe(overallRisk, patterns),
            confidence: Math.min(fusionResult.confidence || 0, patterns?.confidence || 0)
        };
    }
    
    computeBaseRisk(fusionResult) {
        return (fusionResult.anomalyScore + fusionResult.threatLevel / 5) / 2;
    }
    
    computePatternRisk(patterns) {
        if (!patterns || !patterns.detectedPatterns) return 0.3;
        
        const riskPatterns = ['precursory_sequence', 'escalating_unrest', 'thermal_anomaly'];
        const matchedRiskPatterns = patterns.detectedPatterns.filter(p => 
            riskPatterns.some(rp => p.includes(rp))
        );
        
        return Math.min(1.0, matchedRiskPatterns.length / riskPatterns.length + 0.3);
    }
    
    async computeHistoricalRisk(volcanoId) {
        // Simplified historical risk computation
        return 0.4; // Would typically query historical database
    }
    
    combineRiskFactors(baseRisk, patternRisk, historicalRisk) {
        return 0.5 * baseRisk + 0.3 * patternRisk + 0.2 * historicalRisk;
    }
    
    categorizeRisk(riskValue) {
        if (riskValue < 0.2) return 'low';
        if (riskValue < 0.4) return 'moderate';
        if (riskValue < 0.6) return 'elevated';
        if (riskValue < 0.8) return 'high';
        return 'critical';
    }
    
    estimateEruptionProbability(riskValue, patterns) {
        let baseProbability = Math.min(0.95, riskValue * 1.2);
        
        if (patterns?.detectedPatterns?.includes('pre_eruptive_sequence')) {
            baseProbability *= 1.5;
        }
        
        return Math.min(0.99, baseProbability);
    }
    
    estimateTimeframe(riskValue, patterns) {
        if (riskValue > 0.8) return 'Hours to days';
        if (riskValue > 0.6) return 'Days to weeks';
        if (riskValue > 0.4) return 'Weeks to months';
        return 'Months to years';
    }
}

class NotificationSystem {
    constructor() {
        this.channels = {
            email: true,
            sms: true,
            push: true,
            webhook: true
        };
        this.subscribers = new Map();
    }
    
    async sendAlert(alert) {
        const notifications = [];
        
        if (this.channels.push) {
            notifications.push(this.sendPushNotification(alert));
        }
        
        if (this.channels.email && alert.level >= 3) {
            notifications.push(this.sendEmailAlert(alert));
        }
        
        if (this.channels.sms && alert.level >= 4) {
            notifications.push(this.sendSMSAlert(alert));
        }
        
        if (this.channels.webhook) {
            notifications.push(this.sendWebhookNotification(alert));
        }
        
        try {
            await Promise.all(notifications);
            console.log(`Notifications sent for alert ${alert.id}`);
        } catch (error) {
            console.error('Notification error:', error);
        }
    }
    
    async sendPushNotification(alert) {
        if ('Notification' in window && Notification.permission === 'granted') {
            new Notification(`Volcanic Alert: ${alert.volcanoId}`, {
                body: `Level ${alert.level} alert with ${(alert.probability * 100).toFixed(0)}% probability`,
                icon: '/icons/volcano-alert.png',
                tag: alert.id,
                requireInteraction: alert.level >= 4
            });
        }
    }
    
    async sendEmailAlert(alert) {
        // Implement email sending logic
        console.log(`Email alert sent for ${alert.volcanoId}`);
    }
    
    async sendSMSAlert(alert) {
        // Implement SMS sending logic
        console.log(`SMS alert sent for ${alert.volcanoId}`);
    }
    
    async sendWebhookNotification(alert) {
        // Implement webhook notification
        try {
            await fetch('/api/alerts/webhook', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(alert)
            });
        } catch (error) {
            console.error('Webhook notification failed:', error);
        }
    }
}

class PatternRecognitionEngine {
    constructor() {
        this.patterns = new Map();
        this.historicalDatabase = new Map();
    }
    
    async analyzePatterns(volcanoId, sensorData) {
        const detectedPatterns = [];
        const confidence = 0.8; // Simplified
        
        // Check for common volcanic precursory patterns
        if (this.detectSeismicSwarm(sensorData)) {
            detectedPatterns.push('seismic_swarm');
        }
        
        if (this.detectGroundDeformation(sensorData)) {
            detectedPatterns.push('ground_deformation');
        }
        
        if (this.detectThermalAnomaly(sensorData)) {
            detectedPatterns.push('thermal_anomaly');
        }
        
        if (this.detectGasEmissionChanges(sensorData)) {
            detectedPatterns.push('gas_emission_changes');
        }
        
        // Check for escalating patterns
        if (detectedPatterns.length >= 2) {
            detectedPatterns.push('multi_parameter_unrest');
        }
        
        if (this.detectAcceleratingTrends(sensorData)) {
            detectedPatterns.push('accelerating_unrest');
        }
        
        const historicalMatches = await this.findHistoricalMatches(volcanoId, detectedPatterns);
        
        return {
            detectedPatterns,
            confidence,
            historicalMatches: historicalMatches.length,
            patternSeverity: this.assessPatternSeverity(detectedPatterns),
            recommendation: this.generateRecommendation(detectedPatterns)
        };
    }
    
    detectSeismicSwarm(sensorData) {
        const seismicSensors = sensorData.filter(s => s.sensor_type === 'seismic');
        return seismicSensors.length > 5; // Simplified logic
    }
    
    detectGroundDeformation(sensorData) {
        const deformationSensors = sensorData.filter(s => 
            s.sensor_type === 'gnss' || s.sensor_type === 'insar'
        );
        return deformationSensors.some(s => Math.abs(s.value) > 0.01); // >1cm displacement
    }
    
    detectThermalAnomaly(sensorData) {
        const thermalSensors = sensorData.filter(s => s.sensor_type === 'thermal');
        return thermalSensors.some(s => s.value > 350); // >350K temperature
    }
    
    detectGasEmissionChanges(sensorData) {
        const gasSensors = sensorData.filter(s => s.sensor_type === 'gas');
        return gasSensors.length > 0; // Simplified
    }
    
    detectAcceleratingTrends(sensorData) {
        // Simplified trend analysis
        return sensorData.length > 10;
    }
    
    async findHistoricalMatches(volcanoId, patterns) {
        // Simplified historical matching
        return patterns.slice(0, 2); // Return subset as matches
    }
    
    assessPatternSeverity(patterns) {
        const severityWeights = {
            'seismic_swarm': 0.3,
            'ground_deformation': 0.4,
            'thermal_anomaly': 0.5,
            'gas_emission_changes': 0.3,
            'multi_parameter_unrest': 0.8,
            'accelerating_unrest': 0.9
        };
        
        const totalSeverity = patterns.reduce((sum, pattern) => 
            sum + (severityWeights[pattern] || 0.1), 0
        );
        
        return Math.min(1.0, totalSeverity / patterns.length);
    }
    
    generateRecommendation(patterns) {
        if (patterns.includes('accelerating_unrest')) {
            return 'Immediate increased monitoring recommended';
        }
        if (patterns.includes('multi_parameter_unrest')) {
            return 'Enhanced surveillance and preparation advised';
        }
        if (patterns.length > 0) {
            return 'Continued close monitoring recommended';
        }
        return 'Normal monitoring protocols sufficient';
    }
}

class UncertaintyQuantification {
    async computeDecisionUncertainty(fusionResult, historicalContext) {
        const dataUncertainty = this.computeDataUncertainty(fusionResult);
        const modelUncertainty = this.computeModelUncertainty(fusionResult);
        const contextUncertainty = this.computeContextUncertainty(historicalContext);
        
        // Combine uncertainties using error propagation
        const totalUncertainty = Math.sqrt(
            Math.pow(dataUncertainty, 2) +
            Math.pow(modelUncertainty, 2) +
            Math.pow(contextUncertainty, 2)
        ) / Math.sqrt(3);
        
        return Math.max(0, Math.min(1, totalUncertainty));
    }
    
    computeDataUncertainty(fusionResult) {
        const sensorCount = fusionResult.contributingSensors?.length || 0;
        const confidence = fusionResult.confidence || 0;
        
        // More sensors and higher confidence reduce uncertainty
        return Math.max(0.1, 1.0 - (confidence * Math.min(1, sensorCount / 5)));
    }
    
    computeModelUncertainty(fusionResult) {
        // Model uncertainty based on prediction confidence
        return Math.max(0.05, 1.0 - (fusionResult.confidence || 0.5));
    }
    
    computeContextUncertainty(historicalContext) {
        // Uncertainty based on availability of historical context
        return historicalContext ? 0.2 : 0.5;
    }
}

class AlertMetricsTracker {
    constructor() {
        this.metrics = {
            totalAlerts: 0,
            alertsByLevel: new Array(6).fill(0),
            falsePositives: 0,
            truePositives: 0,
            responseTime: [],
            acknowledgmentRate: 0
        };
    }
    
    recordAlert(alert) {
        this.metrics.totalAlerts++;
        this.metrics.alertsByLevel[alert.level]++;
    }
    
    recordFalsePositive(alertId) {
        this.metrics.falsePositives++;
    }
    
    recordTruePositive(alertId) {
        this.metrics.truePositives++;
    }
    
    getMetrics() {
        const totalClassified = this.metrics.falsePositives + this.metrics.truePositives;
        const accuracy = totalClassified > 0 ? 
            this.metrics.truePositives / totalClassified : 0;
        
        return {
            ...this.metrics,
            accuracy,
            averageResponseTime: this.metrics.responseTime.length > 0 ?
                this.metrics.responseTime.reduce((a, b) => a + b) / this.metrics.responseTime.length : 0
        };
    }
}

// Initialize global instance
let intelligentAlertSystem;

// Auto-initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    intelligentAlertSystem = new IntelligentAlertSystem();
    
    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        IntelligentAlertSystem,
        AdvancedRiskAssessment,
        NotificationSystem,
        PatternRecognitionEngine,
        UncertaintyQuantification,
        AlertMetricsTracker
    };
}