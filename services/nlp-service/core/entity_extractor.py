"""
Geological Entity Extraction
Named Entity Recognition specialized for geological and geophysical terms
"""

import re
import time
from typing import Dict, List, Set

import structlog

# Optional imports for NLP functionality
try:
    import spacy
    from spacy.matcher import Matcher

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

import numpy as np
import torch.nn.functional as F
from models.nlp_models import EntityExtractionResult, EntityType, GeologicalEntity

logger = structlog.get_logger()


class GeologicalEntityExtractor:
    """
    Extracts geological entities from natural language text using:
    - Geological term dictionaries
    - Pattern matching for measurements and coordinates
    - NLP models for general entity recognition
    """

    def __init__(self):
        self.logger = logger.bind(component="entity_extractor")

        # Initialize geological knowledge base
        self.geological_terms = self._initialize_geological_terms()

        # Initialize spaCy model if available
        self.nlp = None
        self.matcher = None

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.matcher = Matcher(self.nlp.vocab)
                self._initialize_patterns()
            except OSError:
                self.logger.warning(
                    "spaCy model not found, using dictionary-based extraction only"
                )
                self.nlp = None
        else:
            self.logger.warning(
                "spaCy not available, using dictionary-based extraction only"
            )

        self.logger.info("Geological entity extractor initialized successfully")

    def _initialize_geological_terms(self) -> Dict[EntityType, Set[str]]:
        """Initialize comprehensive geological term dictionary"""
        return {
            EntityType.LOCATION: {
                # Volcanic features
                "volcano",
                "volcanic",
                "caldera",
                "crater",
                "vent",
                "fissure",
                "dome",
                "cone",
                "stratovolcano",
                "shield volcano",
                "cinder cone",
                "composite volcano",
                "volcanic field",
                "volcanic arc",
                "hotspot",
                "seamount",
                # Tectonic features
                "fault",
                "rift",
                "ridge",
                "trench",
                "subduction zone",
                "transform fault",
                "spreading center",
                "convergent boundary",
                "divergent boundary",
                "plate boundary",
                "continental margin",
                "oceanic ridge",
                # Geological formations
                "basin",
                "plateau",
                "mountain",
                "valley",
                "canyon",
                "gorge",
                "continental shel",
                "abyssal plain",
                "continental slope",
                "sedimentary basin",
                "igneous province",
                "metamorphic core complex",
                # Specific locations (examples)
                "yellowstone",
                "mount st helens",
                "kilauea",
                "vesuvius",
                "etna",
                "san andreas",
                "cascadia",
                "ring of fire",
                "mid-atlantic ridge",
            },
            EntityType.PROCESS: {
                # Volcanic processes
                "eruption",
                "explosive eruption",
                "effusive eruption",
                "phreatomagmatic",
                "strombolian",
                "vulcanian",
                "plinian",
                "hawaiian",
                "surtseyan",
                "magma ascent",
                "degassing",
                "crystallization",
                "differentiation",
                # Seismic processes
                "earthquake",
                "seismic",
                "rupture",
                "slip",
                "stick-slip",
                "foreshock",
                "aftershock",
                "mainshock",
                "earthquake swarm",
                "induced seismicity",
                "triggered earthquakes",
                # Tectonic processes
                "subduction",
                "collision",
                "rifting",
                "spreading",
                "transform motion",
                "plate tectonics",
                "continental drift",
                "seafloor spreading",
                "orogeny",
                "extension",
                "compression",
                "transpression",
                # Surface processes
                "weathering",
                "erosion",
                "sedimentation",
                "mass wasting",
                "landslide",
                "debris flow",
                "rockfall",
                "avalanche",
                # Deep Earth processes
                "mantle convection",
                "core formation",
                "differentiation",
                "partial melting",
                "fractional crystallization",
                "assimilation",
            },
            EntityType.MATERIAL: {
                # Igneous materials
                "magma",
                "lava",
                "basalt",
                "andesite",
                "dacite",
                "rhyolite",
                "granite",
                "gabbro",
                "peridotite",
                "obsidian",
                "pumice",
                "scoria",
                "volcanic ash",
                "tephra",
                "pyroclastic",
                # Sedimentary materials
                "sediment",
                "sandstone",
                "limestone",
                "shale",
                "conglomerate",
                "mudstone",
                "siltstone",
                "breccia",
                "turbidite",
                # Metamorphic materials
                "gneiss",
                "schist",
                "slate",
                "marble",
                "quartzite",
                "amphibolite",
                "eclogite",
                "blueschist",
                "greenschist",
                # Minerals
                "quartz",
                "feldspar",
                "mica",
                "olivine",
                "pyroxene",
                "amphibole",
                "plagioclase",
                "orthoclase",
                "biotite",
                "calcite",
                "dolomite",
                "pyrite",
                "magnetite",
                # Fluids and gases
                "water",
                "co2",
                "so2",
                "h2s",
                "steam",
                "brine",
                "hydrothermal fluid",
                "volcanic gas",
                "groundwater",
            },
            EntityType.MEASUREMENT: {
                # Physical measurements
                "magnitude",
                "intensity",
                "depth",
                "temperature",
                "pressure",
                "velocity",
                "acceleration",
                "displacement",
                "strain",
                "stress",
                "density",
                "viscosity",
                "permeability",
                # Seismic measurements
                "p-wave",
                "s-wave",
                "surface wave",
                "body wave",
                "seismic velocity",
                "attenuation",
                "quality factor",
                "moment magnitude",
                "local magnitude",
                "richter scale",
                # Geochemical measurements
                "composition",
                "concentration",
                "isotope ratio",
                "trace element",
                "major element",
                "volatile content",
                "ph",
                "salinity",
                "alkalinity",
                "redox potential",
                # Geophysical measurements
                "gravity",
                "magnetic field",
                "electrical resistivity",
                "heat flow",
                "geothermal gradient",
                "seismic tomography",
            },
            EntityType.TIME_PERIOD: {
                # Geological time periods
                "cenozoic",
                "mesozoic",
                "paleozoic",
                "precambrian",
                "quaternary",
                "tertiary",
                "cretaceous",
                "jurassic",
                "triassic",
                "permian",
                "carboniferous",
                "devonian",
                "silurian",
                "ordovician",
                "cambrian",
                # Quaternary subdivisions
                "holocene",
                "pleistocene",
                "pliocene",
                "miocene",
                "oligocene",
                "eocene",
                "paleocene",
                # Time scales
                "geological time",
                "deep time",
                "recent",
                "historical",
                "prehistoric",
                "modern",
                "contemporary",
            },
            EntityType.INSTRUMENT: {
                # Seismic instruments
                "seismometer",
                "seismograph",
                "accelerometer",
                "broadband seismometer",
                "short period seismometer",
                "ocean bottom seismometer",
                "borehole seismometer",
                # Geodetic instruments
                "gps",
                "gnss",
                "insar",
                "tiltmeter",
                "strainmeter",
                "edm",
                "total station",
                "leveling",
                "vlbi",
                # Geochemical instruments
                "spectrometer",
                "mass spectrometer",
                "gas chromatograph",
                "x-ray fluorescence",
                "electron microprobe",
                "ion chromatograph",
                "multigas sensor",
                # Geophysical instruments
                "gravimeter",
                "magnetometer",
                "ground penetrating radar",
                "electrical resistivity",
                "electromagnetic",
                "mt",
                # Remote sensing
                "satellite",
                "radar",
                "lidar",
                "thermal camera",
                "hyperspectral",
                "multispectral",
                "aster",
                "landsat",
            },
            EntityType.PARAMETER: {
                # Model parameters
                "initial conditions",
                "boundary conditions",
                "mesh resolution",
                "time step",
                "convergence criteria",
                "iteration limit",
                # Physical parameters
                "young's modulus",
                "poisson's ratio",
                "bulk modulus",
                "shear modulus",
                "thermal conductivity",
                "heat capacity",
                "thermal expansion",
                "compressibility",
                # Fluid parameters
                "reynolds number",
                "prandtl number",
                "rayleigh number",
                "peclet number",
                "nusselt number",
                "grashof number",
            },
        }

    def _initialize_patterns(self):
        """Initialize spaCy patterns for geological entities"""
        if not self.nlp:
            return

        # Coordinate patterns
        coordinate_pattern = [
            {"TEXT": {"REGEX": r"\d+\.?\d*"}},
            {"TEXT": {"IN": ["°", "degrees", "deg"]}},
            {
                "TEXT": {"IN": ["N", "S", "E", "W", "North", "South", "East", "West"]},
                "OP": "?",
            },
        ]
        self.matcher.add("COORDINATE", [coordinate_pattern])

        # Measurement patterns
        measurement_pattern = [
            {"TEXT": {"REGEX": r"\d+\.?\d*"}},
            {
                "TEXT": {
                    "IN": [
                        "km",
                        "m",
                        "cm",
                        "mm",
                        "°C",
                        "K",
                        "Pa",
                        "MPa",
                        "GPa",
                        "kg/m³",
                    ]
                }
            },
        ]
        self.matcher.add("MEASUREMENT", [measurement_pattern])

        # Depth patterns
        depth_pattern = [
            {"LOWER": {"IN": ["depth", "at"]}},
            {"TEXT": {"REGEX": r"\d+\.?\d*"}},
            {"TEXT": {"IN": ["km", "m", "meters", "kilometers"]}},
        ]
        self.matcher.add("DEPTH", [depth_pattern])

        # Magnitude patterns
        magnitude_pattern = [
            {"LOWER": {"IN": ["magnitude", "m", "mw", "ml", "mb"]}},
            {"TEXT": {"REGEX": r"\d+\.?\d*"}},
        ]
        self.matcher.add("MAGNITUDE", [magnitude_pattern])

    def _extract_with_dictionary(self, text: str) -> List[GeologicalEntity]:
        """Extract entities using geological term dictionary"""
        entities = []
        text_lower = text.lower()

        for entity_type, terms in self.geological_terms.items():
            for term in terms:
                # Find all occurrences of the term
                pattern = r"\b" + re.escape(term.lower()) + r"\b"
                matches = re.finditer(pattern, text_lower)

                for match in matches:
                    start_pos = match.start()
                    end_pos = match.end()

                    # Calculate confidence based on term specificity
                    confidence = min(0.95, 0.6 + len(term) / 50)

                    entity = GeologicalEntity(
                        text=text[start_pos:end_pos],
                        entity_type=entity_type,
                        confidence=confidence,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        normalized_value=term,
                        metadata={"extraction_method": "dictionary"},
                    )
                    entities.append(entity)

        return entities

    def _extract_with_patterns(self, text: str) -> List[GeologicalEntity]:
        """Extract entities using spaCy patterns"""
        if not self.nlp:
            return []

        entities = []
        doc = self.nlp(text)
        matches = self.matcher(doc)

        for match_id, start, end in matches:
            span = doc[start:end]
            label = self.nlp.vocab.strings[match_id]

            # Determine entity type based on pattern label
            if label == "COORDINATE":
                entity_type = EntityType.LOCATION
            elif label in ["MEASUREMENT", "DEPTH", "MAGNITUDE"]:
                entity_type = EntityType.MEASUREMENT
            else:
                entity_type = EntityType.PARAMETER

            entity = GeologicalEntity(
                text=span.text,
                entity_type=entity_type,
                confidence=0.8,
                start_pos=span.start_char,
                end_pos=span.end_char,
                normalized_value=span.text.lower(),
                metadata={"extraction_method": "pattern", "pattern_type": label},
            )
            entities.append(entity)

        return entities

    def _extract_with_spacy_ner(self, text: str) -> List[GeologicalEntity]:
        """Extract entities using spaCy's built-in NER"""
        if not self.nlp:
            return []

        entities = []
        doc = self.nlp(text)

        for ent in doc.ents:
            # Map spaCy entity types to geological entity types
            if ent.label_ in ["GPE", "LOC"]:  # Geopolitical entity, Location
                entity_type = EntityType.LOCATION
            elif ent.label_ in ["QUANTITY", "CARDINAL"]:
                entity_type = EntityType.MEASUREMENT
            elif ent.label_ in ["DATE", "TIME"]:
                entity_type = EntityType.TIME_PERIOD
            else:
                continue  # Skip non-geological entities

            entity = GeologicalEntity(
                text=ent.text,
                entity_type=entity_type,
                confidence=0.7,  # Lower confidence for general NER
                start_pos=ent.start_char,
                end_pos=ent.end_char,
                normalized_value=ent.text.lower(),
                metadata={"extraction_method": "spacy_ner", "spacy_label": ent.label_},
            )
            entities.append(entity)

        return entities

    def _merge_overlapping_entities(
        self, entities: List[GeologicalEntity]
    ) -> List[GeologicalEntity]:
        """Merge overlapping entities, keeping the one with higher confidence"""
        if not entities:
            return entities

        # Sort entities by start position
        sorted_entities = sorted(entities, key=lambda e: e.start_pos)
        merged_entities = []

        current_entity = sorted_entities[0]

        for next_entity in sorted_entities[1:]:
            # Check for overlap
            if next_entity.start_pos < current_entity.end_pos:
                # Overlapping entities - keep the one with higher confidence
                if next_entity.confidence > current_entity.confidence:
                    current_entity = next_entity
                # If same confidence, prefer longer entity
                elif next_entity.confidence == current_entity.confidence and (
                    next_entity.end_pos - next_entity.start_pos
                ) > (current_entity.end_pos - current_entity.start_pos):
                    current_entity = next_entity
            else:
                # No overlap - add current entity and move to next
                merged_entities.append(current_entity)
                current_entity = next_entity

        # Add the last entity
        merged_entities.append(current_entity)

        return merged_entities

    async def extract_entities(self, text: str) -> EntityExtractionResult:
        """
        Extract geological entities from natural language text

        Args:
            text: Input text to extract entities from

        Returns:
            EntityExtractionResult with extracted geological entities
        """
        start_time = time.time()

        try:
            self.logger.info("Extracting geological entities", text_length=len(text))

            all_entities = []

            # Extract using dictionary matching
            dict_entities = self._extract_with_dictionary(text)
            all_entities.extend(dict_entities)

            # Extract using pattern matching
            pattern_entities = self._extract_with_patterns(text)
            all_entities.extend(pattern_entities)

            # Extract using spaCy NER
            spacy_entities = self._extract_with_spacy_ner(text)
            all_entities.extend(spacy_entities)

            # Merge overlapping entities
            merged_entities = self._merge_overlapping_entities(all_entities)

            # Sort by position
            final_entities = sorted(merged_entities, key=lambda e: e.start_pos)

            processing_time = (time.time() - start_time) * 1000

            self.logger.info(
                "Entity extraction completed",
                total_entities=len(final_entities),
                processing_time_ms=processing_time,
            )

            return EntityExtractionResult(
                entities=final_entities,
                processing_time_ms=processing_time,
                total_entities=len(final_entities),
            )

        except Exception as e:
            self.logger.error("Entity extraction failed", error=str(e))
            processing_time = (time.time() - start_time) * 1000

            return EntityExtractionResult(
                entities=[], processing_time_ms=processing_time, total_entities=0
            )
