#!/usr/bin/env python3
"""
Advanced Geochemical Data Processing System
Comprehensive processing, validation, and analysis of geochemical data
"""

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List

# Visualization
import pandas as pd
import torch.nn.functional as F

# Scientific computing

# Geochemical libraries
try:
    pass

    PYROLITE_AVAILABLE = True
except ImportError:
    PYROLITE_AVAILABLE = False
    logging.warning("Pyrolite not available - using simplified geochemical processing")

logger = logging.getLogger(__name__)

@dataclass
class GeochemicalDataset:
    """Comprehensive geochemical dataset"""

    sample_ids: List[str]
    major_elements: pd.DataFrame  # wt% oxides
    trace_elements: pd.DataFrame  # ppm concentrations
    isotope_ratios: pd.DataFrame  # isotope ratios
    metadata: pd.DataFrame  # sample metadata
    analytical_methods: Dict[str, str]  # analytical techniques used
    reference_materials: Dict[str, Dict[str, float]]  # reference standards
    detection_limits: Dict[str, float]  # analytical detection limits
    uncertainties: pd.DataFrame  # analytical uncertainties
    quality_flags: pd.DataFrame  # data quality flags

@dataclass
class ProcessingParameters:
    """Parameters for geochemical data processing"""

    outlier_detection_method: str = "isolation_forest"
    outlier_threshold: float = 0.1
    missing_data_strategy: str = "iterative_imputation"
    normalization_method: str = "closure"
    transformation_method: str = "clr"  # centered log-ratio
    dimensionality_reduction: str = "pca"
    clustering_method: str = "hierarchical"
    reference_composition: str = "chondrite"

class AdvancedGeochemicalProcessor:
    """Advanced geochemical data processing system"""

    def __init__(self):
        self.reference_compositions = self._initialize_reference_compositions()
        self.element_properties = self._initialize_element_properties()
        self.analytical_methods = self._initialize_analytical_methods()
        self.quality_metrics = {}

    def _initialize_reference_compositions(self) -> Dict[str, Dict[str, float]]:
        """Initialize reference compositions for normalization"""

        return {
            "chondrite_mcdonough_sun": {
                # McDonough & Sun (1995) CI chondrite values
                "La": 0.237,
                "Ce": 0.613,
                "Pr": 0.095,
                "Nd": 0.467,
                "Sm": 0.153,
                "Eu": 0.058,
                "Gd": 0.204,
                "Tb": 0.037,
                "Dy": 0.246,
                "Ho": 0.0546,
                "Er": 0.160,
                "Tm": 0.0247,
                "Yb": 0.161,
                "Lu": 0.0246,
                "Rb": 2.32,
                "Ba": 2.41,
                "Th": 0.029,
                "U": 0.008,
                "Nb": 0.246,
                "Ta": 0.014,
                "K": 545,
                "La": 0.237,
                "Ce": 0.613,
                "Pb": 0.071,
                "Pr": 0.095,
                "Sr": 7.26,
                "Nd": 0.467,
                "Zr": 3.87,
                "H": 0.104,
                "Sm": 0.153,
                "Eu": 0.058,
                "Ti": 445,
                "Gd": 0.204,
                "Tb": 0.037,
                "Dy": 0.246,
                "Y": 1.57,
                "Ho": 0.0546,
                "Er": 0.160,
                "Tm": 0.0247,
                "Yb": 0.161,
                "Lu": 0.0246,
            },
            "primitive_mantle_mcdonough_sun": {
                # McDonough & Sun (1995) primitive mantle values
                "La": 0.648,
                "Ce": 1.675,
                "Pr": 0.254,
                "Nd": 1.25,
                "Sm": 0.406,
                "Eu": 0.154,
                "Gd": 0.544,
                "Tb": 0.099,
                "Dy": 0.674,
                "Ho": 0.149,
                "Er": 0.438,
                "Tm": 0.068,
                "Yb": 0.441,
                "Lu": 0.0675,
                "Rb": 0.635,
                "Ba": 6.989,
                "Th": 0.085,
                "U": 0.021,
                "Nb": 0.713,
                "Ta": 0.041,
                "K": 250,
                "Pb": 0.071,
                "Sr": 19.9,
                "Zr": 10.5,
                "H": 0.283,
                "Ti": 1205,
                "Y": 4.3,
            },
            "morb_sun_mcdonough": {
                # Sun & McDonough (1989) N-MORB values
                "La": 2.5,
                "Ce": 7.5,
                "Pr": 1.32,
                "Nd": 7.3,
                "Sm": 2.63,
                "Eu": 1.02,
                "Gd": 3.68,
                "Tb": 0.67,
                "Dy": 4.55,
                "Ho": 1.01,
                "Er": 2.97,
                "Tm": 0.456,
                "Yb": 3.05,
                "Lu": 0.455,
                "Rb": 0.56,
                "Ba": 6.3,
                "Th": 0.12,
                "U": 0.047,
                "Nb": 2.33,
                "Ta": 0.132,
                "K": 600,
                "Pb": 0.3,
                "Sr": 90,
                "Zr": 74,
                "H": 2.05,
                "Ti": 7600,
                "Y": 28,
            },
            "oib_sun_mcdonough": {
                # Sun & McDonough (1989) OIB values
                "La": 37,
                "Ce": 80,
                "Pr": 9.7,
                "Nd": 38.5,
                "Sm": 10,
                "Eu": 3,
                "Gd": 7.62,
                "Tb": 1.05,
                "Dy": 5.6,
                "Ho": 1.06,
                "Er": 2.62,
                "Tm": 0.35,
                "Yb": 2.16,
                "Lu": 0.3,
                "Rb": 31,
                "Ba": 350,
                "Th": 4,
                "U": 1.02,
                "Nb": 48,
                "Ta": 2.7,
                "K": 12000,
                "Pb": 3.2,
                "Sr": 660,
                "Zr": 280,
                "H": 7.8,
                "Ti": 17200,
                "Y": 29,
            },
        }

    def _initialize_element_properties(self) -> Dict[str, Dict[str, Any]]:
        """Initialize element properties database"""

        return {
            "La": {
                "atomic_number": 57,
                "ionic_radius": 1.032,
                "compatibility": "incompatible",
                "group": "LREE",
            },
            "Ce": {
                "atomic_number": 58,
                "ionic_radius": 1.010,
                "compatibility": "incompatible",
                "group": "LREE",
            },
            "Pr": {
                "atomic_number": 59,
                "ionic_radius": 0.990,
                "compatibility": "incompatible",
                "group": "LREE",
            },
            "Nd": {
                "atomic_number": 60,
                "ionic_radius": 0.983,
                "compatibility": "incompatible",
                "group": "LREE",
            },
            "Sm": {
                "atomic_number": 62,
                "ionic_radius": 0.958,
                "compatibility": "incompatible",
                "group": "MREE",
            },
            "Eu": {
                "atomic_number": 63,
                "ionic_radius": 0.947,
                "compatibility": "incompatible",
                "group": "MREE",
            },
            "Gd": {
                "atomic_number": 64,
                "ionic_radius": 0.938,
                "compatibility": "incompatible",
                "group": "MREE",
            },
            "Tb": {
                "atomic_number": 65,
                "ionic_radius": 0.923,
                "compatibility": "incompatible",
                "group": "HREE",
            },
            "Dy": {
                "atomic_number": 66,
                "ionic_radius": 0.912,
                "compatibility": "incompatible",
                "group": "HREE",
            },
            "Ho": {
                "atomic_number": 67,
                "ionic_radius": 0.901,
                "compatibility": "incompatible",
                "group": "HREE",
            },
            "Er": {
                "atomic_number": 68,
                "ionic_radius": 0.890,
                "compatibility": "incompatible",
                "group": "HREE",
            },
            "Tm": {
                "atomic_number": 69,
                "ionic_radius": 0.880,
                "compatibility": "incompatible",
                "group": "HREE",
            },
            "Yb": {
                "atomic_number": 70,
                "ionic_radius": 0.868,
                "compatibility": "incompatible",
                "group": "HREE",
            },
            "Lu": {
                "atomic_number": 71,
                "ionic_radius": 0.861,
                "compatibility": "incompatible",
                "group": "HREE",
            },
            "Rb": {
                "atomic_number": 37,
                "ionic_radius": 1.520,
                "compatibility": "incompatible",
                "group": "LILE",
            },
            "Ba": {
                "atomic_number": 56,
                "ionic_radius": 1.350,
                "compatibility": "incompatible",
                "group": "LILE",
            },
            "Th": {
                "atomic_number": 90,
                "ionic_radius": 0.940,
                "compatibility": "incompatible",
                "group": "HFSE",
            },
            "U": {
                "atomic_number": 92,
                "ionic_radius": 0.890,
                "compatibility": "incompatible",
                "group": "HFSE",
            },
            "Nb": {
                "atomic_number": 41,
                "ionic_radius": 0.640,
                "compatibility": "incompatible",
                "group": "HFSE",
            },
            "Ta": {
                "atomic_number": 73,
                "ionic_radius": 0.640,
                "compatibility": "incompatible",
                "group": "HFSE",
            },
            "Zr": {
                "atomic_number": 40,
                "ionic_radius": 0.720,
                "compatibility": "incompatible",
                "group": "HFSE",
            },
            "Hf": {
                "atomic_number": 72,
                "ionic_radius": 0.710,
                "compatibility": "incompatible",
                "group": "HFSE",
            },
            "Y": {
                "atomic_number": 39,
                "ionic_radius": 0.900,
                "compatibility": "incompatible",
                "group": "HREE",
            },
            "Sr": {
                "atomic_number": 38,
                "ionic_radius": 1.180,
                "compatibility": "incompatible",
                "group": "LILE",
            },
            "Pb": {
                "atomic_number": 82,
                "ionic_radius": 1.190,
                "compatibility": "incompatible",
                "group": "LILE",
            },
            "Ni": {
                "atomic_number": 28,
                "ionic_radius": 0.690,
                "compatibility": "compatible",
                "group": "transition",
            },
            "Cr": {
                "atomic_number": 24,
                "ionic_radius": 0.615,
                "compatibility": "compatible",
                "group": "transition",
            },
            "V": {
                "atomic_number": 23,
                "ionic_radius": 0.640,
                "compatibility": "compatible",
                "group": "transition",
            },
            "Sc": {
                "atomic_number": 21,
                "ionic_radius": 0.745,
                "compatibility": "compatible",
                "group": "transition",
            },
        }

    def _initialize_analytical_methods(self) -> Dict[str, Dict[str, Any]]:
        """Initialize analytical methods database"""

        return {
            "XRF": {
                "full_name": "X-ray Fluorescence Spectrometry",
                "suitable_elements": ["major_elements", "some_trace_elements"],
                "detection_limits": {
                    "SiO2": 0.01,
                    "Al2O3": 0.01,
                    "Fe2O3": 0.01,
                    "MgO": 0.01,
                },
                "precision": {
                    "major_elements": 0.5,
                    "trace_elements": 5.0,
                },  # % relative
                "matrix_effects": True,
                "sample_preparation": "fusion_bead_or_pressed_powder",
            },
            "ICP-MS": {
                "full_name": "Inductively Coupled Plasma Mass Spectrometry",
                "suitable_elements": ["trace_elements", "REE", "isotope_ratios"],
                "detection_limits": {"REE": 0.001, "HFSE": 0.01, "LILE": 0.1},  # ppm
                "precision": {
                    "trace_elements": 2.0,
                    "isotope_ratios": 0.1,
                },  # % relative
                "matrix_effects": True,
                "sample_preparation": "acid_digestion",
            },
            "ICP-OES": {
                "full_name": "Inductively Coupled Plasma Optical Emission Spectrometry",
                "suitable_elements": ["major_elements", "trace_elements"],
                "detection_limits": {
                    "major_elements": 0.001,
                    "trace_elements": 0.1,
                },  # wt% or ppm
                "precision": {
                    "major_elements": 1.0,
                    "trace_elements": 3.0,
                },  # % relative
                "matrix_effects": True,
                "sample_preparation": "acid_digestion",
            },
            "EPMA": {
                "full_name": "Electron Probe Microanalysis",
                "suitable_elements": ["major_elements", "minor_elements"],
                "detection_limits": {
                    "major_elements": 0.01,
                    "minor_elements": 0.02,
                },  # wt%
                "precision": {
                    "major_elements": 0.5,
                    "minor_elements": 2.0,
                },  # % relative
                "matrix_effects": True,
                "sample_preparation": "polished_section",
                "spatial_resolution": 1.0,  # micrometers
            },
            "LA-ICP-MS": {
                "full_name": "Laser Ablation Inductively Coupled Plasma Mass Spectrometry",
                "suitable_elements": ["trace_elements", "REE"],
                "detection_limits": {"trace_elements": 0.01, "REE": 0.001},  # ppm
                "precision": {"trace_elements": 5.0, "REE": 3.0},  # % relative
                "matrix_effects": True,
                "sample_preparation": "polished_section",
                "spatial_resolution": 25.0,  # micrometers
            },
            "TIMS": {
                "full_name": "Thermal Ionization Mass Spectrometry",
                "suitable_elements": ["isotope_ratios"],
                "detection_limits": {"Sr_isotopes": 1e-12, "Nd_isotopes": 1e-12},  # mol
                "precision": {"isotope_ratios": 0.001},  # % relative
                "matrix_effects": False,
                "sample_preparation": "chemical_separation",
            },
        }

    async def process_geochemical_dataset(
        self, dataset: GeochemicalDataset, parameters: ProcessingParameters
    ) -> Dict[str, Any]:
        """
        Comprehensive geochemical dataset processing
        """

        logger.info(
            f"Processing geochemical dataset with {len(dataset.sample_ids)} samples"
        )

        processing_results = {
            "dataset_summary": self._generate_dataset_summary(dataset),
            "processing_timestamp": datetime.now().isoformat(),
            "parameters": asdict(parameters),
        }

        # Data quality assessment
        quality_assessment = await self._assess_data_quality(dataset)
        processing_results["quality_assessment"] = quality_assessment

        # Outlier detection and removal
        outlier_analysis = await self._detect_outliers(dataset, parameters)
        processing_results["outlier_analysis"] = outlier_analysis

        # Missing data handling
        imputed_dataset = await self._handle_missing_data(dataset, parameters)
        processing_results["missing_data_handling"] = {
            "method": parameters.missing_data_strategy,
            "missing_data_summary": self._summarize_missing_data(dataset),
        }

        # Data normalization and transformation
        normalized_dataset = await self._normalize_and_transform(
            imputed_dataset, parameters
        )
        processing_results["normalization"] = {
            "method": parameters.normalization_method,
            "transformation": parameters.transformation_method,
        }

        # Compositional data analysis
        compositional_analysis = await self._perform_compositional_analysis(
            normalized_dataset
        )
        processing_results["compositional_analysis"] = compositional_analysis

        # Dimensionality reduction
        dimensionality_reduction = await self._perform_dimensionality_reduction(
            normalized_dataset, parameters
        )
        processing_results["dimensionality_reduction"] = dimensionality_reduction

        # Clustering analysis
        clustering_analysis = await self._perform_clustering_analysis(
            normalized_dataset, parameters
        )
        processing_results["clustering_analysis"] = clustering_analysis

        # Geochemical classification
        classification_results = await self._perform_geochemical_classification(
            normalized_dataset
        )
        processing_results["classification"] = classification_results

        # Statistical analysis
        statistical_analysis = await self._perform_statistical_analysis(
            normalized_dataset
        )
        processing_results["statistical_analysis"] = statistical_analysis

        # Visualization generation
        visualizations = await self._generate_visualizations(
            normalized_dataset, processing_results
        )
        processing_results["visualizations"] = visualizations

        return processing_results

    def _generate_dataset_summary(self, dataset: GeochemicalDataset) -> Dict[str, Any]:
        """Generate comprehensive dataset summary"""

        summary = {
            "total_samples": len(dataset.sample_ids),
            "major_elements": {
                "count": len(dataset.major_elements.columns),
                "elements": list(dataset.major_elements.columns),
                "completeness": (
                    dataset.major_elements.notna().sum() / len(dataset.major_elements)
                ).to_dict(),
            },
            "trace_elements": {
                "count": len(dataset.trace_elements.columns),
                "elements": list(dataset.trace_elements.columns),
                "completeness": (
                    dataset.trace_elements.notna().sum() / len(dataset.trace_elements)
                ).to_dict(),
            },
            "isotope_ratios": (
                {
                    "count": len(dataset.isotope_ratios.columns),
                    "ratios": list(dataset.isotope_ratios.columns),
                    "completeness": (
                        dataset.isotope_ratios.notna().sum()
                        / len(dataset.isotope_ratios)
                    ).to_dict(),
                }
                if not dataset.isotope_ratios.empty
                else {"count": 0}
            ),
            "analytical_methods": dataset.analytical_methods,
            "reference_materials": list(dataset.reference_materials.keys()),
        }

        return summary

    async def _assess_data_quality(self, dataset: GeochemicalDataset) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""

        quality_metrics = {}

        # Completeness assessment
        completeness = self._assess_completeness(dataset)
        quality_metrics["completeness"] = completeness

        # Consistency assessment
        consistency = self._assess_consistency(dataset)
        quality_metrics["consistency"] = consistency

        # Accuracy assessment (using reference materials)
        accuracy = self._assess_accuracy(dataset)
        quality_metrics["accuracy"] = accuracy

        # Precision assessment
        precision = self._assess_precision(dataset)
        quality_metrics["precision"] = precision

        # Detection limit assessment
        detection_limits = self._assess_detection_limits(dataset)
        quality_metrics["detection_limits"] = detection_limits

        # Analytical blank assessment
        blank_assessment = self._assess_analytical_blanks(dataset)
        quality_metrics["blank_assessment"] = blank_assessment

        # Overall quality score
        overall_score = self._calculate_overall_quality_score(quality_metrics)
        quality_metrics["overall_score"] = overall_score

        return quality_metrics

    def _assess_completeness(self, dataset: GeochemicalDataset) -> Dict[str, float]:
        """Assess data completeness"""

        completeness = {}

        # Major elements completeness
        major_completeness = dataset.major_elements.notna().sum() / len(
            dataset.major_elements
        )
        completeness["major_elements"] = major_completeness.mean()

        # Trace elements completeness
        trace_completeness = dataset.trace_elements.notna().sum() / len(
            dataset.trace_elements
        )
        completeness["trace_elements"] = trace_completeness.mean()

        # Isotope ratios completeness
        if not dataset.isotope_ratios.empty:
            isotope_completeness = dataset.isotope_ratios.notna().sum() / len(
                dataset.isotope_ratios
            )
            completeness["isotope_ratios"] = isotope_completeness.mean()

        # Sample-wise completeness
        all_data = pd.concat([dataset.major_elements, dataset.trace_elements], axis=1)
        sample_completeness = all_data.notna().sum(axis=1) / len(all_data.columns)
        completeness["sample_wise_mean"] = sample_completeness.mean()
        completeness["sample_wise_std"] = sample_completeness.std()

        return completeness

    def _assess_consistency(self, dataset: GeochemicalDataset) -> Dict[str, Any]:
        """Assess internal data consistency"""

        consistency = {}

        # Check for impossible values
        impossible_values = self._check_impossible_values(dataset)
        consistency["impossible_values"] = impossible_values

        # Check element correlations
        correlations = self._check_element_correlations(dataset)
        consistency["correlations"] = correlations

        # Check mass balance (major elements should sum to ~100%)
        mass_balance = self._check_mass_balance(dataset)
        consistency["mass_balance"] = mass_balance

        # Check charge balance (if available)
        if "charge_balance" in dataset.metadata.columns:
            charge_balance = dataset.metadata["charge_balance"].describe().to_dict()
            consistency["charge_balance"] = charge_balance

        return consistency

    def _check_impossible_values(self, dataset: GeochemicalDataset) -> Dict[str, int]:
        """Check for impossible geochemical values"""

        impossible_counts = {}

        # Major elements should be positive and reasonable
        for element in dataset.major_elements.columns:
            if element == "SiO2":
                impossible = (
                    (dataset.major_elements[element] < 30)
                    | (dataset.major_elements[element] > 80)
                ).sum()
            elif element in ["Al2O3", "Fe2O3", "FeO", "MgO", "CaO"]:
                impossible = (
                    (dataset.major_elements[element] < 0)
                    | (dataset.major_elements[element] > 50)
                ).sum()
            else:
                impossible = (dataset.major_elements[element] < 0).sum()

            impossible_counts[element] = impossible

        # Trace elements should be positive
        for element in dataset.trace_elements.columns:
            impossible = (dataset.trace_elements[element] < 0).sum()
            impossible_counts[element] = impossible

        return impossible_counts
