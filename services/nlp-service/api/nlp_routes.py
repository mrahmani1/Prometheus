"""
NLP API routes for geological query processing
"""

from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException
import structlog
from models.nlp_models import (
    ConversationalResponse, 
    EntityExtractionResult,
    GeologicalQueryRequest, 
    GeologicalQueryResponse,
    IntentClassificationResult, 
    NLPQuery,
    QueryTranslationResult
)

logger = structlog.get_logger()
router = APIRouter()

def get_nlp_components():
    """Dependency to get NLP components from global state"""
    from main import nlp_components

    return nlp_components

@router.post("/classify-intent", response_model=IntentClassificationResult)
async def classify_intent(
    query: NLPQuery, components: Dict = Depends(get_nlp_components)
):
    """
    Classify the intent of a geological query
    """
    try:
        logger.info("Processing intent classification", query=query.text)

        intent_classifier = components["intent_classifier"]
        result = await intent_classifier.classify_intent(query.text)

        logger.info(
            "Intent classification completed",
            intent=result.intent,
            confidence=result.confidence,
        )
        return result

    except Exception as e:
        logger.error("Intent classification failed", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Intent classification failed: {str(e)}"
        )

@router.post("/extract-entities", response_model=EntityExtractionResult)
async def extract_entities(
    query: NLPQuery, components: Dict = Depends(get_nlp_components)
):
    """
    Extract geological entities from natural language query
    """
    try:
        logger.info("Processing entity extraction", query=query.text)

        entity_extractor = components["entity_extractor"]
        result = await entity_extractor.extract_entities(query.text)

        logger.info("Entity extraction completed", entities_count=len(result.entities))
        return result

    except Exception as e:
        logger.error("Entity extraction failed", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Entity extraction failed: {str(e)}"
        )

@router.post("/translate-query", response_model=QueryTranslationResult)
async def translate_query(
    query: NLPQuery, components: Dict = Depends(get_nlp_components)
):
    """
    Translate natural language query to system parameters
    """
    try:
        logger.info("Processing query translation", query=query.text)

        query_translator = components["query_translator"]
        result = await query_translator.translate_query(query.text)

        logger.info(
            "Query translation completed", parameters_count=len(result.parameters)
        )
        return result

    except Exception as e:
        logger.error("Query translation failed", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Query translation failed: {str(e)}"
        )

@router.post("/generate-response", response_model=ConversationalResponse)
async def generate_response(
    request: GeologicalQueryRequest, components: Dict = Depends(get_nlp_components)
):
    """
    Generate conversational response for geological explanations
    """
    try:
        logger.info("Processing response generation", query=request.query)

        response_generator = components["response_generator"]
        result = await response_generator.generate_response(
            query=request.query,
            context=request.context,
            geological_data=request.geological_data,
        )

        logger.info(
            "Response generation completed", response_length=len(result.response)
        )
        return result

    except Exception as e:
        logger.error("Response generation failed", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Response generation failed: {str(e)}"
        )

@router.post("/process-geological-query", response_model=GeologicalQueryResponse)
async def process_geological_query(
    request: GeologicalQueryRequest, components: Dict = Depends(get_nlp_components)
):
    """
    Complete end-to-end processing of geological query
    """
    try:
        logger.info("Processing complete geological query", query=request.query)

        # Step 1: Classify intent
        intent_classifier = components["intent_classifier"]
        intent_result = await intent_classifier.classify_intent(request.query)

        # Step 2: Extract entities
        entity_extractor = components["entity_extractor"]
        entity_result = await entity_extractor.extract_entities(request.query)

        # Step 3: Translate to system parameters
        query_translator = components["query_translator"]
        translation_result = await query_translator.translate_query(request.query)

        # Step 4: Generate conversational response
        response_generator = components["response_generator"]
        response_result = await response_generator.generate_response(
            query=request.query,
            context=request.context,
            geological_data=request.geological_data,
        )

        # Combine all results
        complete_response = GeologicalQueryResponse(
            query=request.query,
            intent=intent_result,
            entities=entity_result,
            translation=translation_result,
            response=response_result,
            processing_time=datetime.utcnow(),
            correlation_id=request.correlation_id,
        )

        logger.info(
            "Complete geological query processing finished",
            intent=intent_result.intent,
            entities_count=len(entity_result.entities),
        )

        return complete_response

    except Exception as e:
        logger.error("Complete geological query processing failed", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Query processing failed: {str(e)}"
        )

@router.get("/geological-terms")
async def get_geological_terms():
    """
    Get list of supported geological terms and entities
    """
    try:
        # This would typically come from a geological ontology or knowledge base
        geological_terms = {
            "locations": [
                "volcano",
                "fault",
                "basin",
                "ridge",
                "trench",
                "plate",
                "continent",
            ],
            "processes": [
                "eruption",
                "earthquake",
                "subduction",
                "rifting",
                "collision",
                "weathering",
            ],
            "materials": [
                "magma",
                "lava",
                "basalt",
                "granite",
                "sediment",
                "mineral",
                "crystal",
            ],
            "measurements": [
                "magnitude",
                "depth",
                "temperature",
                "pressure",
                "velocity",
                "displacement",
            ],
            "time_periods": [
                "holocene",
                "pleistocene",
                "miocene",
                "cretaceous",
                "jurassic",
                "permian",
            ],
            "instruments": [
                "seismometer",
                "gps",
                "insar",
                "spectrometer",
                "gravimeter",
                "magnetometer",
            ],
        }

        return {"geological_terms": geological_terms}

    except Exception as e:
        logger.error("Failed to retrieve geological terms", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve geological terms: {str(e)}"
        )

@router.get("/supported-intents")
async def get_supported_intents():
    """
    Get list of supported query intents
    """
    try:
        supported_intents = {
            "simulation_request": "Request for geological simulation",
            "data_analysis": "Request for data analysis",
            "monitoring_query": "Query about real-time monitoring",
            "hazard_assessment": "Request for hazard assessment",
            "resource_prediction": "Query about resource prediction",
            "explanation_request": "Request for geological explanation",
            "comparison_query": "Request to compare geological features",
            "historical_query": "Query about historical geological events",
        }

        return {"supported_intents": supported_intents}

    except Exception as e:
        logger.error("Failed to retrieve supported intents", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve supported intents: {str(e)}"
        )
