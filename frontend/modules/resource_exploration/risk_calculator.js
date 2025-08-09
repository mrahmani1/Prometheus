/**
 * Advanced Risk Calculator
 * Comprehensive risk assessment algorithms for multi-hazard analysis
 */

window.RISK_CALCULATOR = (function() {
    'use strict';

    // Risk assessment parameters and models
    const RISK_MODELS = {
        SEISMIC: {
            name: 'Seismic Risk Assessment',
            fragility_curves: {
                unreinforced_masonry: { slight: 0.1, moderate: 0.2, extensive: 0.4, complete: 0.8 },
                wood_frame: { slight: 0.15, moderate: 0.3, extensive: 0.6, complete: 1.2 },
                concrete: { slight: 0.2, moderate: 0.4, extensive: 0.8, complete: 1.5 },
                steel: { slight: 0.25, moderate: 0.5, extensive: 1.0, complete: 2.0 }
            },
            casualty_rates: { slight: 0.001, moderate: 0.01, extensive: 0.1, complete: 0.5 }
        },
        VOLCANIC: {
            name: 'Volcanic Risk Assessment',
            vei_impacts: {
                2: { radius: 5, casualty_rate: 0.05, damage_factor: 0.3 },
                3: { radius: 10, casualty_rate: 0.1, damage_factor: 0.5 },
                4: { radius: 20, casualty_rate: 0.2, damage_factor: 0.7 },
                5: { radius: 40, casualty_rate: 0.4, damage_factor: 0.9 },
                6: { radius: 80, casualty_rate: 0.6, damage_factor: 1.0 }
            },
            hazard_types: {
                lava_flows: { speed: 0.1, damage_prob: 1.0, casualty_rate: 0.01 },
                pyroclastic_flows: { speed: 100, damage_prob: 1.0, casualty_rate: 0.8 },
                ash_fall: { thickness_damage: { 10: 0.1, 50: 0.3, 200: 0.7, 500: 1.0 } },
                lahars: { speed: 20, damage_prob: 0.9, casualty_rate: 0.3 }
            }
        },
        TSUNAMI: {
            name: 'Tsunami Risk Assessment',
            inundation_damage: {
                1: { damage_ratio: 0.1, casualty_rate: 0.01 },
                2: { damage_ratio: 0.3, casualty_rate: 0.05 },
                5: { damage_ratio: 0.6, casualty_rate: 0.2 },
                10: { damage_ratio: 0.9, casualty_rate: 0.5 },
                20: { damage_ratio: 1.0, casualty_rate: 0.8 }
            }
        },
        LANDSLIDE: {
            name: 'Landslide Risk Assessment',
            susceptibility_factors: {
                slope: { low: 0.1, moderate: 0.3, high: 0.7, very_high: 1.0 },
                geology: { stable: 0.1, moderate: 0.4, unstable: 0.8 },
                precipitation: { dry: 0.1, normal: 0.3, wet: 0.6, extreme: 1.0 }
            }
        }
    };

    const VULNERABILITY_MODELS = {
        POPULATION: {
            age_factors: { under_18: 1.2, adult: 1.0, elderly: 1.5 },
            socioeconomic_factors: { low: 1.3, medium: 1.0, high: 0.7 },
            education_factors: { low: 1.2, medium: 1.0, high: 0.8 }
        },
        INFRASTRUCTURE: {
            building_codes: { none: 2.0, basic: 1.5, moderate: 1.0, advanced: 0.6 },
            construction_quality: { poor: 1.8, average: 1.0, good: 0.7, excellent: 0.5 },
            maintenance: { poor: 1.4, average: 1.0, good: 0.8 }
        },
        ECONOMIC: {
            sector_vulnerability: {
                agriculture: 1.5, manufacturing: 1.2, services: 1.0, 
                technology: 0.8, finance: 0.9, tourism: 1.8
            },
            recovery_factors: { slow: 2.0, moderate: 1.0, fast: 0.5 }
        }
    };

    const LOSS_FUNCTIONS = {
        STRUCTURAL: {
            residential: { cost_per_unit: 200000, replacement_threshold: 0.5 },
            commercial: { cost_per_unit: 500000, replacement_threshold: 0.4 },
            industrial: { cost_per_unit: 1000000, replacement_threshold: 0.3 },
            critical: { cost_per_unit: 2000000, replacement_threshold: 0.2 }
        },
        NON_STRUCTURAL: {
            contents: 0.3, // 30% of structural value
            business_interruption: 0.2, // 20% of structural value
            additional_living_expenses: 0.1 // 10% of structural value
        }
    };

    // Initialize the risk calculator
    function init() {
        console.log('⚠️ Initializing Advanced Risk Calculator...');
        
        setupRiskModels();
        initializeUncertaintyModels();
        
        console.log('✅ Risk Calculator initialized');
    }

    function setupRiskModels() {
        // Initialize Monte Carlo simulation parameters
        window.RISK_CALCULATOR.monte_carlo_params = {
            iterations: 10000,
            confidence_intervals: [0.05, 0.1, 0.16, 0.5, 0.84, 0.9, 0.95],
            random_seed: 12345
        };
        
        // Setup correlation matrices for multi-hazard interactions
        window.RISK_CALCULATOR.hazard_correlations = {
            seismic_landslide: 0.7,
            volcanic_lahar: 0.8,
            seismic_tsunami: 0.9,
            volcanic_ash_aviation: 0.6
        };
    }

    function initializeUncertaintyModels() {
        // Epistemic and aleatory uncertainty parameters
        window.RISK_CALCULATOR.uncertainty = {
            magnitude: { mean: 0, std: 0.3 },
            location: { mean: 0, std: 5 }, // km
            ground_motion: { mean: 0, std: 0.4 },
            fragility: { mean: 0, std: 0.2 },
            consequence: { mean: 0, std: 0.3 }
        };
    }

    // Comprehensive risk calculation function
    function calculateComprehensiveRisk(hazardParams, vulnerabilityParams, exposureParams) {
        console.log('🔍 Calculating comprehensive risk assessment...');
        
        // Initialize risk components
        const riskComponents = {
            seismic_risk: 0,
            volcanic_risk: 0,
            multi_hazard_risk: 0,
            total_risk: 0
        };
        
        // Calculate individual hazard risks
        if (hazardParams.seismic) {
            riskComponents.seismic_risk = calculateSeismicRisk(
                hazardParams.seismic, vulnerabilityParams, exposureParams
            );
        }
        
        if (hazardParams.volcanic) {
            riskComponents.volcanic_risk = calculateVolcanicRisk(
                hazardParams.volcanic, vulnerabilityParams, exposureParams
            );
        }
        
        if (hazardParams.tsunami) {
            riskComponents.tsunami_risk = calculateTsunamiRisk(
                hazardParams.tsunami, vulnerabilityParams, exposureParams
            );
        }
        
        if (hazardParams.landslide) {
            riskComponents.landslide_risk = calculateLandslideRisk(
                hazardParams.landslide, vulnerabilityParams, exposureParams
            );
        }
        
        // Calculate multi-hazard interactions
        riskComponents.multi_hazard_risk = calculateMultiHazardRisk(riskComponents);
        
        // Calculate total risk with uncertainty
        const totalRiskDistribution = performMonteCarloAnalysis(riskComponents);
        
        // Generate comprehensive risk metrics
        const riskMetrics = generateRiskMetrics(riskComponents, totalRiskDistribution);
        
        console.log('✅ Comprehensive risk assessment complete');
        return riskMetrics;
    }

    function calculateSeismicRisk(seismicParams, vulnerability, exposure) {
        const { magnitude, distance, site_class, fault_type } = seismicParams;
        
        // Ground motion prediction equation (simplified GMPE)
        const pga = calculatePGA(magnitude, distance, site_class, fault_type);
        
        // Building damage assessment
        const damageStates = calculateSeismicDamage(pga, vulnerability.building_stock);
        
        // Casualty estimation
        const casualties = calculateSeismicCasualties(damageStates, exposure.population);
        
        // Economic losses
        const economicLosses = calculateSeismicLosses(damageStates, exposure.building_inventory);
        
        return {
            pga: pga,
            damage_states: damageStates,
            casualties: casualties,
            economic_losses: economicLosses,
            risk_level: classifyRiskLevel(casualties.total + economicLosses.total / 1000000)
        };
    }

    function calculatePGA(magnitude, distance, siteClass, faultType) {
        // Simplified Boore-Atkinson GMPE
        const c1 = 0.757;
        const c2 = 0.922;
        const c3 = -0.0824;
        
        const logPGA = c1 + c2 * (magnitude - 6) + c3 * Math.log(Math.sqrt(distance * distance + 25));
        
        // Site amplification factors
        const siteFactors = { A: 0.8, B: 1.0, C: 1.2, D: 1.5, E: 2.0 };
        const siteFactor = siteFactors[siteClass] || 1.0;
        
        // Fault type factor
        const faultFactors = { strike_slip: 1.0, normal: 0.9, reverse: 1.1, thrust: 1.2 };
        const faultFactor = faultFactors[faultType] || 1.0;
        
        return Math.exp(logPGA) * siteFactor * faultFactor;
    }

    function calculateSeismicDamage(pga, buildingStock) {
        const fragilityCurves = RISK_MODELS.SEISMIC.fragility_curves[buildingStock];
        if (!fragilityCurves) {
            console.warn('Unknown building stock type:', buildingStock);
            return { none: 1.0, slight: 0, moderate: 0, extensive: 0, complete: 0 };
        }
        
        // Calculate damage probabilities using lognormal fragility curves
        const damageStates = {};
        const damageTypes = ['slight', 'moderate', 'extensive', 'complete'];
        
        damageTypes.forEach((damageType, index) => {
            const median = fragilityCurves[damageType];
            const beta = 0.6; // Lognormal standard deviation
            
            const phi = normalCDF(Math.log(pga / median) / beta);
            damageStates[damageType] = phi;
        });
        
        // Calculate complementary probabilities
        damageStates.none = 1 - damageStates.slight;
        damageStates.slight = damageStates.slight - damageStates.moderate;
        damageStates.moderate = damageStates.moderate - damageStates.extensive;
        damageStates.extensive = damageStates.extensive - damageStates.complete;
        
        return damageStates;
    }

    function calculateSeismicCasualties(damageStates, population) {
        const casualtyRates = RISK_MODELS.SEISMIC.casualty_rates;
        
        const casualties = {
            fatalities: 0,
            injuries: 0,
            displaced: 0,
            total: 0
        };
        
        Object.keys(damageStates).forEach(damageType => {
            if (casualtyRates[damageType]) {
                const rate = casualtyRates[damageType];
                const affected = damageStates[damageType] * population;
                
                casualties.fatalities += affected * rate * 0.1; // 10% fatality rate
                casualties.injuries += affected * rate * 0.3; // 30% injury rate
                casualties.displaced += affected * rate * 2.0; // 200% displacement rate
            }
        });
        
        casualties.total = casualties.fatalities + casualties.injuries;
        return casualties;
    }

    function calculateSeismicLosses(damageStates, buildingInventory) {
        const losses = {
            structural: 0,
            non_structural: 0,
            contents: 0,
            business_interruption: 0,
            total: 0
        };
        
        const lossRatios = { none: 0, slight: 0.02, moderate: 0.1, extensive: 0.5, complete: 1.0 };
        
        Object.keys(damageStates).forEach(damageType => {
            const probability = damageStates[damageType];
            const lossRatio = lossRatios[damageType];
            
            const structuralLoss = probability * lossRatio * buildingInventory.structural_value;
            losses.structural += structuralLoss;
            
            losses.contents += structuralLoss * LOSS_FUNCTIONS.NON_STRUCTURAL.contents;
            losses.business_interruption += structuralLoss * LOSS_FUNCTIONS.NON_STRUCTURAL.business_interruption;
        });
        
        losses.non_structural = losses.contents + losses.business_interruption;
        losses.total = losses.structural + losses.non_structural;
        
        return losses;
    }

    function calculateVolcanicRisk(volcanicParams, vulnerability, exposure) {
        const { vei, distance_from_vent, wind_direction, hazard_types } = volcanicParams;
        
        // VEI-based impact assessment
        const veiImpacts = RISK_MODELS.VOLCANIC.vei_impacts[vei] || RISK_MODELS.VOLCANIC.vei_impacts[4];
        
        // Calculate individual volcanic hazard risks
        const hazardRisks = {};
        
        if (hazard_types.includes('lava_flows')) {
            hazardRisks.lava_flows = calculateLavaFlowRisk(distance_from_vent, exposure);
        }
        
        if (hazard_types.includes('pyroclastic_flows')) {
            hazardRisks.pyroclastic_flows = calculatePyroclasticFlowRisk(vei, distance_from_vent, exposure);
        }
        
        if (hazard_types.includes('ash_fall')) {
            hazardRisks.ash_fall = calculateAshFallRisk(vei, distance_from_vent, wind_direction, exposure);
        }
        
        if (hazard_types.includes('lahars')) {
            hazardRisks.lahars = calculateLaharRisk(distance_from_vent, exposure);
        }
        
        // Aggregate volcanic risks
        const totalCasualties = Object.values(hazardRisks).reduce((sum, risk) => sum + (risk.casualties || 0), 0);
        const totalLosses = Object.values(hazardRisks).reduce((sum, risk) => sum + (risk.economic_losses || 0), 0);
        
        return {
            vei: vei,
            hazard_risks: hazardRisks,
            total_casualties: totalCasualties,
            total_losses: totalLosses,
            evacuation_radius: veiImpacts.radius,
            risk_level: classifyRiskLevel(totalCasualties + totalLosses / 1000000)
        };
    }

    function calculateLavaFlowRisk(distance, exposure) {
        const lavaHazard = RISK_MODELS.VOLCANIC.hazard_types.lava_flows;
        const arrivalTime = distance / lavaHazard.speed; // hours
        
        let casualties = 0;
        let economicLosses = 0;
        
        if (distance < 10) { // Within 10 km
            const evacuationSuccess = arrivalTime > 24 ? 0.9 : 0.5; // 90% success if >24h warning
            casualties = exposure.population * (1 - evacuationSuccess) * lavaHazard.casualty_rate;
            economicLosses = exposure.building_inventory.total_value * lavaHazard.damage_prob;
        }
        
        return { casualties, economic_losses: economicLosses, arrival_time: arrivalTime };
    }

    function calculatePyroclasticFlowRisk(vei, distance, exposure) {
        const pyroclasticHazard = RISK_MODELS.VOLCANIC.hazard_types.pyroclastic_flows;
        const maxDistance = Math.pow(2, vei) * 5; // VEI-dependent range
        
        let casualties = 0;
        let economicLosses = 0;
        
        if (distance < maxDistance) {
            const intensity = Math.max(0, 1 - distance / maxDistance);
            casualties = exposure.population * intensity * pyroclasticHazard.casualty_rate;
            economicLosses = exposure.building_inventory.total_value * intensity * pyroclasticHazard.damage_prob;
        }
        
        return { casualties, economic_losses: economicLosses, max_distance: maxDistance };
    }

    function calculateAshFallRisk(vei, distance, windDirection, exposure) {
        const ashHazard = RISK_MODELS.VOLCANIC.hazard_types.ash_fall;
        
        // Simplified ash thickness model
        let ashThickness = 0;
        const maxDistance = Math.pow(10, vei * 0.5) * 20;
        
        if (distance < maxDistance) {
            // Wind direction factor (simplified)
            const windFactor = windDirection === 'favorable' ? 2.0 : 0.5;
            ashThickness = (maxDistance - distance) / maxDistance * Math.pow(10, vei) * windFactor; // mm
        }
        
        // Damage based on ash thickness
        let damageRatio = 0;
        const thresholds = Object.keys(ashHazard.thickness_damage).map(Number).sort((a, b) => a - b);
        for (const threshold of thresholds) {
            if (ashThickness >= threshold) {
                damageRatio = ashHazard.thickness_damage[threshold];
            }
        }
        
        const casualties = exposure.population * damageRatio * 0.01; // Low casualty rate for ash fall
        const economicLosses = exposure.building_inventory.total_value * damageRatio;
        
        return { 
            casualties, 
            economic_losses: economicLosses, 
            ash_thickness: ashThickness,
            damage_ratio: damageRatio 
        };
    }

    function calculateLaharRisk(distance, exposure) {
        const laharHazard = RISK_MODELS.VOLCANIC.hazard_types.lahars;
        const maxDistance = 50; // km typical lahar runout
        
        let casualties = 0;
        let economicLosses = 0;
        
        if (distance < maxDistance) {
            const intensity = Math.max(0, 1 - distance / maxDistance);
            casualties = exposure.population * intensity * laharHazard.casualty_rate;
            economicLosses = exposure.building_inventory.total_value * intensity * laharHazard.damage_prob;
        }
        
        return { casualties, economic_losses: economicLosses };
    }

    function calculateTsunamiRisk(tsunamiParams, vulnerability, exposure) {
        const { wave_height, distance_from_coast, warning_time } = tsunamiParams;
        
        const inundationDamage = RISK_MODELS.TSUNAMI.inundation_damage;
        
        // Find appropriate damage category based on wave height
        let damageCategory = 1;
        const heights = Object.keys(inundationDamage).map(Number).sort((a, b) => b - a);
        for (const height of heights) {
            if (wave_height >= height) {
                damageCategory = height;
                break;
            }
        }
        
        const damage = inundationDamage[damageCategory];
        
        // Evacuation success based on warning time
        const evacuationSuccess = Math.min(0.95, Math.max(0.1, warning_time / 60)); // 1 hour = 95% success
        
        const casualties = exposure.population * (1 - evacuationSuccess) * damage.casualty_rate;
        const economicLosses = exposure.building_inventory.total_value * damage.damage_ratio;
        
        return {
            wave_height: wave_height,
            casualties: casualties,
            economic_losses: economicLosses,
            evacuation_success: evacuationSuccess,
            inundation_damage_ratio: damage.damage_ratio
        };
    }

    function calculateLandslideRisk(landslideParams, vulnerability, exposure) {
        const { slope, geology, precipitation, trigger_event } = landslideParams;
        
        const susceptibilityFactors = RISK_MODELS.LANDSLIDE.susceptibility_factors;
        
        // Calculate landslide susceptibility
        const slopeFactor = susceptibilityFactors.slope[slope] || 0.5;
        const geologyFactor = susceptibilityFactors.geology[geology] || 0.5;
        const precipitationFactor = susceptibilityFactors.precipitation[precipitation] || 0.3;
        
        // Trigger event amplification
        const triggerAmplification = trigger_event === 'earthquake' ? 2.0 : 
                                   trigger_event === 'heavy_rain' ? 1.5 : 1.0;
        
        const landslideProb = Math.min(1.0, slopeFactor * geologyFactor * precipitationFactor * triggerAmplification);
        
        // Impact assessment
        const affectedArea = 2.0; // km² typical landslide area
        const casualties = landslideProb * (exposure.population / 10) * 0.1; // 10% casualty rate in affected area
        const economicLosses = landslideProb * affectedArea * 5000000; // $5M/km² typical loss
        
        return {
            landslide_probability: landslideProb,
            casualties: casualties,
            economic_losses: economicLosses,
            affected_area: affectedArea
        };
    }

    function calculateMultiHazardRisk(riskComponents) {
        // Account for hazard interactions and cascading effects
        const correlations = window.RISK_CALCULATOR.hazard_correlations;
        
        let multiHazardMultiplier = 1.0;
        
        // Seismic-triggered landslides
        if (riskComponents.seismic_risk && riskComponents.landslide_risk) {
            multiHazardMultiplier += correlations.seismic_landslide * 0.3;
        }
        
        // Volcanic-triggered lahars
        if (riskComponents.volcanic_risk && riskComponents.volcanic_risk.hazard_risks.lahars) {
            multiHazardMultiplier += correlations.volcanic_lahar * 0.2;
        }
        
        // Seismic-triggered tsunamis
        if (riskComponents.seismic_risk && riskComponents.tsunami_risk) {
            multiHazardMultiplier += correlations.seismic_tsunami * 0.5;
        }
        
        return {
            multiplier: multiHazardMultiplier,
            cascading_effects: multiHazardMultiplier - 1.0,
            interaction_types: Object.keys(correlations)
        };
    }

    function performMonteCarloAnalysis(riskComponents) {
        const iterations = window.RISK_CALCULATOR.monte_carlo_params.iterations;
        const results = [];
        
        for (let i = 0; i < iterations; i++) {
            let totalRisk = 0;
            
            // Add uncertainty to each risk component
            Object.values(riskComponents).forEach(component => {
                if (typeof component === 'number') {
                    const uncertainty = 1 + (Math.random() - 0.5) * 0.4; // ±20% uncertainty
                    totalRisk += component * uncertainty;
                }
            });
            
            results.push(totalRisk);
        }
        
        results.sort((a, b) => a - b);
        return results;
    }

    function generateRiskMetrics(riskComponents, totalRiskDistribution) {
        const confidenceIntervals = window.RISK_CALCULATOR.monte_carlo_params.confidence_intervals;
        
        // Calculate percentiles
        const percentiles = {};
        confidenceIntervals.forEach(ci => {
            const index = Math.floor(ci * totalRiskDistribution.length);
            percentiles[`p${Math.round(ci * 100)}`] = totalRiskDistribution[index];
        });
        
        // Calculate summary statistics
        const mean = totalRiskDistribution.reduce((a, b) => a + b, 0) / totalRiskDistribution.length;
        const median = percentiles.p50;
        
        // Risk classification
        const overallRisk = classifyRiskLevel(median);
        
        // Calculate expected losses and casualties
        let expectedCasualties = 0;
        let expectedLosses = 0;
        
        Object.values(riskComponents).forEach(component => {
            if (component && typeof component === 'object') {
                expectedCasualties += component.casualties || component.total_casualties || 0;
                expectedLosses += component.economic_losses || component.total_losses || 0;
            }
        });
        
        // Estimate recovery time based on damage severity
        const recoveryTime = estimateRecoveryTime(expectedLosses, expectedCasualties);
        
        return {
            overall_risk: overallRisk,
            casualty_risk: classifyRiskLevel(expectedCasualties),
            economic_risk: classifyRiskLevel(expectedLosses / 1000000),
            expected_casualties: Math.round(expectedCasualties),
            economic_losses: Math.round(expectedLosses),
            recovery_time: recoveryTime,
            confidence_intervals: percentiles,
            risk_components: riskComponents,
            mean_risk: mean,
            median_risk: median,
            uncertainty_range: percentiles.p95 - percentiles.p5
        };
    }

    function classifyRiskLevel(value) {
        if (value < 1) return 'Very Low';
        if (value < 10) return 'Low';
        if (value < 100) return 'Moderate';
        if (value < 1000) return 'High';
        return 'Very High';
    }

    function estimateRecoveryTime(economicLosses, casualties) {
        // Recovery time estimation based on losses and casualties
        const lossMonths = Math.log10(economicLosses / 1000000 + 1) * 6; // Months based on economic losses
        const casualtyMonths = Math.log10(casualties + 1) * 3; // Months based on casualties
        
        return Math.ceil(Math.max(lossMonths, casualtyMonths));
    }

    function normalCDF(x) {
        // Standard normal cumulative distribution function approximation
        const a1 =  0.254829592;
        const a2 = -0.284496736;
        const a3 =  1.421413741;
        const a4 = -1.453152027;
        const a5 =  1.061405429;
        const p  =  0.3275911;
        
        const sign = x < 0 ? -1 : 1;
        x = Math.abs(x) / Math.sqrt(2.0);
        
        const t = 1.0 / (1.0 + p * x);
        const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
        
        return 0.5 * (1.0 + sign * y);
    }

    // Simplified risk calculation for quick assessments
    function calculateBasicRisk(vulnerabilityFactors) {
        const population = vulnerabilityFactors.population || { factor: 0.6, count: 25000 };
        const buildings = vulnerabilityFactors.buildings || { factor: 0.4, cost_per_unit: 300000 };
        const criticalFacilities = vulnerabilityFactors.critical_facilities || 5;
        
        // Basic risk calculation
        const casualties = population.count * population.factor * 0.05; // 5% casualty rate
        const economicLosses = buildings.cost_per_unit * 100 * buildings.factor; // Assume 100 buildings
        const recoveryTime = Math.ceil(economicLosses / 10000000); // 1 month per $10M
        
        return {
            overall_risk: classifyRiskLevel(casualties + economicLosses / 1000000),
            casualty_risk: classifyRiskLevel(casualties),
            economic_risk: classifyRiskLevel(economicLosses / 1000000),
            expected_casualties: Math.round(casualties),
            economic_losses: Math.round(economicLosses),
            recovery_time: recoveryTime,
            critical_facilities_at_risk: criticalFacilities
        };
    }

    // Risk-based decision support
    function generateRiskMitigationRecommendations(riskMetrics) {
        const recommendations = [];
        
        if (riskMetrics.overall_risk === 'Very High' || riskMetrics.overall_risk === 'High') {
            recommendations.push('Implement immediate evacuation procedures');
            recommendations.push('Activate emergency response teams');
            recommendations.push('Establish emergency shelters and medical facilities');
        }
        
        if (riskMetrics.casualty_risk === 'High') {
            recommendations.push('Improve early warning systems');
            recommendations.push('Conduct regular evacuation drills');
            recommendations.push('Strengthen emergency medical response');
        }
        
        if (riskMetrics.economic_risk === 'High') {
            recommendations.push('Enforce stricter building codes');
            recommendations.push('Implement insurance and financial protection measures');
            recommendations.push('Develop rapid recovery and reconstruction plans');
        }
        
        return recommendations;
    }

    // Public API
    return {
        init: init,
        calculateComprehensiveRisk: calculateComprehensiveRisk,
        calculateSeismicRisk: calculateSeismicRisk,
        calculateVolcanicRisk: calculateVolcanicRisk,
        calculateTsunamiRisk: calculateTsunamiRisk,
        calculateLandslideRisk: calculateLandslideRisk,
        calculateBasicRisk: calculateBasicRisk,
        generateRiskMitigationRecommendations: generateRiskMitigationRecommendations,
        performMonteCarloAnalysis: performMonteCarloAnalysis,
        classifyRiskLevel: classifyRiskLevel,
        RISK_MODELS: RISK_MODELS,
        VULNERABILITY_MODELS: VULNERABILITY_MODELS,
        LOSS_FUNCTIONS: LOSS_FUNCTIONS
    };

})();

// Auto-initialize
document.addEventListener('DOMContentLoaded', function() {
    window.RISK_CALCULATOR.init();
    console.log('⚠️ Advanced Risk Calculator loaded');
});