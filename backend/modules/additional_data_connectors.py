"""
Additional Data Connectors Module
Comprehensive integration with Wikidata, GeoNames, and Connected Papers APIs
"""

import requests
import json
import asyncio
import aiohttp
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from urllib.parse import urlencode, quote
import re

# Configure logging
logger = logging.getLogger(__name__)

class WikidataConnector:
    """Wikidata SPARQL connector for geological knowledge extraction"""
    
    def __init__(self, rate_limit: float = 0.5):  # Conservative rate limit for Wikidata
        self.endpoint = "https://query.wikidata.org/sparql"
        self.rate_limiter_interval = 1.0 / rate_limit
        self.last_request = 0
        self.user_agent = "GeologicalDataConnector/1.0 (geological-research@example.com)"
        
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        elapsed = current_time - self.last_request
        if elapsed < self.rate_limiter_interval:
            await asyncio.sleep(self.rate_limiter_interval - elapsed)
        self.last_request = time.time()
    
    async def query_geological_entities(self, 
                                      entity_type: str = "volcano",
                                      country: str = None,
                                      limit: int = 100) -> List[Dict[str, Any]]:
        """Query geological entities from Wikidata"""
        
        # Define SPARQL query templates
        queries = {
            "volcano": f"""
            SELECT DISTINCT ?volcano ?volcanoLabel ?country ?countryLabel 
                   ?coords ?elevation ?lastEruption ?volcanoType ?volcanoTypeLabel WHERE {{
              ?volcano wdt:P31/wdt:P279* wd:Q8072 .  # Instance of volcano
              OPTIONAL {{ ?volcano wdt:P17 ?country . }}
              OPTIONAL {{ ?volcano wdt:P625 ?coords . }}
              OPTIONAL {{ ?volcano wdt:P2044 ?elevation . }}
              OPTIONAL {{ ?volcano wdt:P1345 ?lastEruption . }}
              OPTIONAL {{ ?volcano wdt:P279 ?volcanoType . }}
              {f"?volcano wdt:P17 wd:{self._get_country_qid(country)} ." if country else ""}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" . }}
            }}
            LIMIT {limit}
            """,
            
            "earthquake": f"""
            SELECT DISTINCT ?earthquake ?earthquakeLabel ?location ?locationLabel 
                   ?coords ?magnitude ?date WHERE {{
              ?earthquake wdt:P31/wdt:P279* wd:Q8068 .  # Instance of earthquake
              OPTIONAL {{ ?earthquake wdt:P276 ?location . }}
              OPTIONAL {{ ?earthquake wdt:P625 ?coords . }}
              OPTIONAL {{ ?earthquake wdt:P2048 ?magnitude . }}
              OPTIONAL {{ ?earthquake wdt:P585 ?date . }}
              {f"?location wdt:P17 wd:{self._get_country_qid(country)} ." if country else ""}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" . }}
            }}
            LIMIT {limit}
            """,
            
            "geological_formation": f"""
            SELECT DISTINCT ?formation ?formationLabel ?location ?locationLabel 
                   ?coords ?geologicalAge ?ageLabel WHERE {{
              ?formation wdt:P31/wdt:P279* wd:Q1970725 .  # Instance of geological formation
              OPTIONAL {{ ?formation wdt:P276 ?location . }}
              OPTIONAL {{ ?formation wdt:P625 ?coords . }}
              OPTIONAL {{ ?formation wdt:P2348 ?geologicalAge . }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" . }}
            }}
            LIMIT {limit}
            """,
            
            "mineral": f"""
            SELECT DISTINCT ?mineral ?mineralLabel ?chemicalFormula ?crystalSystem 
                   ?hardness ?density ?color WHERE {{
              ?mineral wdt:P31/wdt:P279* wd:Q7946 .  # Instance of mineral
              OPTIONAL {{ ?mineral wdt:P274 ?chemicalFormula . }}
              OPTIONAL {{ ?mineral wdt:P4855 ?crystalSystem . }}
              OPTIONAL {{ ?mineral wdt:P2035 ?hardness . }}
              OPTIONAL {{ ?mineral wdt:P2054 ?density . }}
              OPTIONAL {{ ?mineral wdt:P462 ?color . }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" . }}
            }}
            LIMIT {limit}
            """
        }
        
        if entity_type not in queries:
            logger.error(f"Unknown entity type: {entity_type}")
            return []
        
        try:
            await self._rate_limit()
            
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'application/sparql-results+json'
            }
            
            async with aiohttp.ClientSession() as session:
                params = {
                    'query': queries[entity_type],
                    'format': 'json'
                }
                
                async with session.get(self.endpoint, params=params, headers=headers, timeout=30) as response:
                    if response.status != 200:
                        logger.error(f"Wikidata query failed with status {response.status}")
                        return []
                    
                    data = await response.json()
                    results = []
                    
                    for binding in data.get('results', {}).get('bindings', []):
                        result = {}
                        
                        for key, value in binding.items():
                            if key.endswith('Label'):
                                result[key.replace('Label', '_label')] = value.get('value')
                            else:
                                result[key] = value.get('value')
                        
                        # Parse coordinates if present
                        if 'coords' in result:
                            coords_str = result['coords']
                            if coords_str and coords_str.startswith('Point('):
                                # Extract lat/lon from "Point(longitude latitude)"
                                coords_part = coords_str.replace('Point(', '').replace(')', '')
                                try:
                                    lon, lat = map(float, coords_part.split())
                                    result['latitude'] = lat
                                    result['longitude'] = lon
                                except ValueError:
                                    pass
                        
                        results.append(result)
                    
                    logger.info(f"Retrieved {len(results)} {entity_type} records from Wikidata")
                    return results
                    
        except Exception as e:
            logger.error(f"Error querying Wikidata for {entity_type}: {e}")
            return []
    
    def _get_country_qid(self, country_name: str) -> str:
        """Get Wikidata QID for country name - simplified mapping"""
        country_qids = {
            'United States': 'Q30',
            'Canada': 'Q16',
            'Italy': 'Q38',
            'Japan': 'Q17',
            'Indonesia': 'Q252',
            'Philippines': 'Q928',
            'Chile': 'Q298',
            'Iceland': 'Q189',
            'New Zealand': 'Q664',
            'Greece': 'Q41',
            'Turkey': 'Q43',
            'Mexico': 'Q96',
            'Peru': 'Q419',
            'Ecuador': 'Q736',
            'Guatemala': 'Q774',
            'Costa Rica': 'Q800',
            'Nicaragua': 'Q811',
            'El Salvador': 'Q792',
            'Colombia': 'Q739'
        }
        return country_qids.get(country_name, '')

class GeoNamesConnector:
    """GeoNames.org geographical database connector"""
    
    def __init__(self, username: str = None, rate_limit: float = 1.0):
        self.base_url = "http://api.geonames.org"
        self.username = username or "demo"  # Demo user has limited access
        self.rate_limiter_interval = 1.0 / rate_limit
        self.last_request = 0
        
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        elapsed = current_time - self.last_request
        if elapsed < self.rate_limiter_interval:
            await asyncio.sleep(self.rate_limiter_interval - elapsed)
        self.last_request = time.time()
    
    async def search_geographical_features(self, 
                                         query: str,
                                         feature_class: str = None,
                                         feature_code: str = None,
                                         country_code: str = None,
                                         bbox: Tuple[float, float, float, float] = None,
                                         max_rows: int = 50) -> List[Dict[str, Any]]:
        """Search for geographical features"""
        
        params = {
            'q': query,
            'username': self.username,
            'type': 'json',
            'maxRows': max_rows,
            'style': 'full'
        }
        
        if feature_class:
            params['featureClass'] = feature_class
        if feature_code:
            params['featureCode'] = feature_code
        if country_code:
            params['country'] = country_code
        if bbox:
            params['north'] = bbox[3]
            params['south'] = bbox[1] 
            params['east'] = bbox[2]
            params['west'] = bbox[0]
        
        try:
            await self._rate_limit()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/searchJSON", params=params, timeout=30) as response:
                    if response.status != 200:
                        logger.error(f"GeoNames query failed with status {response.status}")
                        return []
                    
                    data = await response.json()
                    
                    if 'geonames' not in data:
                        if 'status' in data:
                            logger.warning(f"GeoNames API error: {data['status']}")
                        return []
                    
                    features = []
                    for item in data['geonames']:
                        feature = {
                            'geoname_id': item.get('geonameId'),
                            'name': item.get('name'),
                            'ascii_name': item.get('asciiName'),
                            'latitude': float(item.get('lat', 0)),
                            'longitude': float(item.get('lng', 0)),
                            'feature_class': item.get('fcl'),
                            'feature_code': item.get('fcode'),
                            'country_code': item.get('countryCode'),
                            'country_name': item.get('countryName'),
                            'admin1_code': item.get('adminCode1'),
                            'admin1_name': item.get('adminName1'),
                            'population': item.get('population', 0),
                            'elevation': item.get('elevation'),
                            'dem': item.get('dem'),  # Digital Elevation Model
                            'timezone': item.get('timezone', {}).get('timeZoneId'),
                            'bbox': {
                                'north': item.get('bbox', {}).get('north'),
                                'south': item.get('bbox', {}).get('south'),
                                'east': item.get('bbox', {}).get('east'),
                                'west': item.get('bbox', {}).get('west')
                            } if item.get('bbox') else None
                        }
                        features.append(feature)
                    
                    logger.info(f"Retrieved {len(features)} geographical features from GeoNames")
                    return features
                    
        except Exception as e:
            logger.error(f"Error searching GeoNames: {e}")
            return []
    
    async def get_geological_features(self, 
                                    country_code: str = None,
                                    bbox: Tuple[float, float, float, float] = None,
                                    max_rows: int = 100) -> List[Dict[str, Any]]:
        """Get geological features like volcanoes, caves, etc."""
        
        geological_feature_codes = [
            'VLC',   # Volcano
            'CAVE',  # Cave
            'GRGE',  # Gorge
            'GAP',   # Gap
            'PASS',  # Pass
            'RDGE',  # Ridge
            'SPUR',  # Spur
            'VAL',   # Valley
            'CNYN',  # Canyon
            'PLTU'   # Plateau
        ]
        
        all_features = []
        
        for feature_code in geological_feature_codes:
            features = await self.search_geographical_features(
                query='',
                feature_code=feature_code,
                country_code=country_code,
                bbox=bbox,
                max_rows=max_rows // len(geological_feature_codes)
            )
            all_features.extend(features)
        
        return all_features

class ConnectedPapersConnector:
    """Connected Papers API connector for academic literature network analysis"""
    
    def __init__(self, api_key: str = None, rate_limit: float = 1.0):
        self.base_url = "https://api.connectedpapers.com"
        self.api_key = api_key
        self.rate_limiter_interval = 1.0 / rate_limit
        self.last_request = 0
        self.headers = {
            'User-Agent': 'GeologicalResearchConnector/1.0',
            'Accept': 'application/json'
        }
        if self.api_key:
            self.headers['Authorization'] = f'Bearer {self.api_key}'
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        elapsed = current_time - self.last_request
        if elapsed < self.rate_limiter_interval:
            await asyncio.sleep(self.rate_limiter_interval - elapsed)
        self.last_request = time.time()
    
    async def get_paper_graph(self, paper_id: str, fresh: bool = False) -> Dict[str, Any]:
        """Get the citation graph for a paper"""
        
        try:
            await self._rate_limit()
            
            params = {}
            if fresh:
                params['fresh'] = 'true'
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/graph/{paper_id}"
                async with session.get(url, headers=self.headers, params=params, timeout=30) as response:
                    if response.status == 401:
                        logger.error("Connected Papers API authentication failed")
                        return {}
                    elif response.status == 404:
                        logger.warning(f"Paper {paper_id} not found in Connected Papers")
                        return {}
                    elif response.status != 200:
                        logger.error(f"Connected Papers API error: {response.status}")
                        return {}
                    
                    data = await response.json()
                    
                    # Process the graph data
                    graph_data = {
                        'main_paper': data.get('main_paper', {}),
                        'prior_works': data.get('prior_works', []),
                        'derivative_works': data.get('derivative_works', []),
                        'similar_papers': data.get('similar_papers', []),
                        'graph_metadata': {
                            'total_papers': len(data.get('prior_works', [])) + len(data.get('derivative_works', [])) + len(data.get('similar_papers', [])),
                            'retrieved_at': datetime.now().isoformat()
                        }
                    }
                    
                    logger.info(f"Retrieved citation graph for paper {paper_id} with {graph_data['graph_metadata']['total_papers']} connected papers")
                    return graph_data
                    
        except Exception as e:
            logger.error(f"Error getting paper graph from Connected Papers: {e}")
            return {}
    
    async def search_papers(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for papers using Connected Papers API"""
        
        try:
            await self._rate_limit()
            
            params = {
                'q': query,
                'limit': limit
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/search"
                async with session.get(url, headers=self.headers, params=params, timeout=30) as response:
                    if response.status != 200:
                        logger.error(f"Connected Papers search failed with status {response.status}")
                        return []
                    
                    data = await response.json()
                    papers = []
                    
                    for paper in data.get('results', []):
                        processed_paper = {
                            'id': paper.get('id'),
                            'title': paper.get('title'),
                            'authors': paper.get('authors', []),
                            'abstract': paper.get('abstract'),
                            'year': paper.get('year'),
                            'doi': paper.get('doi'),
                            'arxiv_id': paper.get('arxiv_id'),
                            'url': paper.get('url'),
                            'citation_count': paper.get('citation_count', 0),
                            'influence_score': paper.get('influence_score', 0),
                            'keywords': paper.get('keywords', []),
                            'venue': paper.get('venue'),
                            'venue_type': paper.get('venue_type'),
                            'fields_of_study': paper.get('fields_of_study', [])
                        }
                        papers.append(processed_paper)
                    
                    logger.info(f"Found {len(papers)} papers matching query: {query}")
                    return papers
                    
        except Exception as e:
            logger.error(f"Error searching Connected Papers: {e}")
            return []
    
    async def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific paper"""
        
        try:
            await self._rate_limit()
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/paper/{paper_id}"
                async with session.get(url, headers=self.headers, timeout=30) as response:
                    if response.status != 200:
                        logger.error(f"Connected Papers paper details failed with status {response.status}")
                        return {}
                    
                    data = await response.json()
                    
                    # Extract and process paper details
                    paper_details = {
                        'id': data.get('id'),
                        'title': data.get('title'),
                        'abstract': data.get('abstract'),
                        'authors': data.get('authors', []),
                        'year': data.get('year'),
                        'venue': data.get('venue'),
                        'doi': data.get('doi'),
                        'arxiv_id': data.get('arxiv_id'),
                        'url': data.get('url'),
                        'pdf_url': data.get('pdf_url'),
                        'citation_count': data.get('citation_count', 0),
                        'reference_count': data.get('reference_count', 0),
                        'influence_score': data.get('influence_score', 0),
                        'fields_of_study': data.get('fields_of_study', []),
                        'keywords': data.get('keywords', []),
                        'topics': data.get('topics', []),
                        'tldr': data.get('tldr'),
                        'is_open_access': data.get('is_open_access', False),
                        'publication_types': data.get('publication_types', []),
                        's2_paper_id': data.get('s2_paper_id'),
                        'semantic_scholar_url': data.get('semantic_scholar_url'),
                        'mag_id': data.get('mag_id'),
                        'dblp_id': data.get('dblp_id')
                    }
                    
                    return paper_details
                    
        except Exception as e:
            logger.error(f"Error getting paper details from Connected Papers: {e}")
            return {}
    
    async def get_trending_papers(self, field: str = None, time_period: str = "month") -> List[Dict[str, Any]]:
        """Get trending papers in geological sciences"""
        
        geological_fields = [
            "geology", "geophysics", "volcanology", "seismology", 
            "tectonics", "geochemistry", "petrology", "mineralogy",
            "earthquake", "volcano", "deformation"
        ]
        
        field_to_use = field if field else "geology"
        
        try:
            await self._rate_limit()
            
            params = {
                'field': field_to_use,
                'period': time_period,  # "week", "month", "year"
                'limit': 50
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/trending"
                async with session.get(url, headers=self.headers, params=params, timeout=30) as response:
                    if response.status != 200:
                        logger.error(f"Connected Papers trending papers failed with status {response.status}")
                        return []
                    
                    data = await response.json()
                    trending_papers = []
                    
                    for paper in data.get('papers', []):
                        trending_paper = {
                            'id': paper.get('id'),
                            'title': paper.get('title'),
                            'authors': paper.get('authors', []),
                            'year': paper.get('year'),
                            'venue': paper.get('venue'),
                            'citation_count': paper.get('citation_count', 0),
                            'recent_citation_count': paper.get('recent_citation_count', 0),
                            'influence_score': paper.get('influence_score', 0),
                            'trending_score': paper.get('trending_score', 0),
                            'fields_of_study': paper.get('fields_of_study', []),
                            'doi': paper.get('doi'),
                            'url': paper.get('url'),
                            'abstract': paper.get('abstract')
                        }
                        trending_papers.append(trending_paper)
                    
                    logger.info(f"Retrieved {len(trending_papers)} trending papers in {field_to_use}")
                    return trending_papers
                    
        except Exception as e:
            logger.error(f"Error getting trending papers from Connected Papers: {e}")
            return []
    
    async def analyze_research_network(self, paper_ids: List[str]) -> Dict[str, Any]:
        """Analyze the research network formed by a collection of papers"""
        
        network_analysis = {
            'papers': [],
            'connections': [],
            'clusters': [],
            'influential_papers': [],
            'research_fronts': [],
            'collaboration_patterns': {}
        }
        
        try:
            # Get detailed information for each paper
            for paper_id in paper_ids[:10]:  # Limit to avoid rate limits
                paper_details = await self.get_paper_details(paper_id)
                if paper_details:
                    network_analysis['papers'].append(paper_details)
                
                # Get citation graph for network analysis
                graph_data = await self.get_paper_graph(paper_id)
                if graph_data:
                    # Analyze connections
                    connections = []
                    for connected_paper in graph_data.get('prior_works', [])[:5]:  # Limit connections
                        connection = {
                            'from': paper_id,
                            'to': connected_paper.get('id'),
                            'type': 'cites',
                            'weight': connected_paper.get('citation_count', 1)
                        }
                        connections.append(connection)
                    
                    for connected_paper in graph_data.get('derivative_works', [])[:5]:
                        connection = {
                            'from': connected_paper.get('id'),
                            'to': paper_id,
                            'type': 'cited_by',
                            'weight': connected_paper.get('citation_count', 1)
                        }
                        connections.append(connection)
                    
                    network_analysis['connections'].extend(connections)
            
            # Identify influential papers (high citation count and influence score)
            network_analysis['influential_papers'] = sorted(
                network_analysis['papers'],
                key=lambda x: (x.get('citation_count', 0) * x.get('influence_score', 0)),
                reverse=True
            )[:5]
            
            # Basic collaboration analysis
            author_collaborations = {}
            for paper in network_analysis['papers']:
                authors = paper.get('authors', [])
                for i, author in enumerate(authors):
                    if author not in author_collaborations:
                        author_collaborations[author] = {'papers': 0, 'collaborators': set()}
                    author_collaborations[author]['papers'] += 1
                    author_collaborations[author]['collaborators'].update(
                        [other_author for j, other_author in enumerate(authors) if i != j]
                    )
            
            # Convert sets to lists for JSON serialization
            for author, data in author_collaborations.items():
                data['collaborators'] = list(data['collaborators'])
                data['collaboration_score'] = len(data['collaborators']) * data['papers']
            
            network_analysis['collaboration_patterns'] = dict(
                sorted(author_collaborations.items(), 
                      key=lambda x: x[1]['collaboration_score'], 
                      reverse=True)[:10]
            )
            
            return network_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing research network: {e}")
            return network_analysis

# Integration manager for all additional connectors
class AdditionalDataIntegrationManager:
    """Manager for Wikidata, GeoNames, and Connected Papers integration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize connectors
        self.wikidata = WikidataConnector(
            rate_limit=self.config.get('wikidata', {}).get('rate_limit', 0.5)
        )
        
        self.geonames = GeoNamesConnector(
            username=self.config.get('geonames', {}).get('username'),
            rate_limit=self.config.get('geonames', {}).get('rate_limit', 1.0)
        )
        
        self.connected_papers = ConnectedPapersConnector(
            api_key=self.config.get('connected_papers', {}).get('api_key'),
            rate_limit=self.config.get('connected_papers', {}).get('rate_limit', 1.0)
        )
        
        logger.info("Additional data integration manager initialized")
    
    async def collect_comprehensive_geological_data(self, 
                                                  region: str = None,
                                                  bbox: Tuple[float, float, float, float] = None,
                                                  research_topic: str = None) -> Dict[str, Any]:
        """Collect comprehensive geological data from all sources"""
        
        comprehensive_data = {
            'wikidata_entities': {},
            'geonames_features': [],
            'connected_papers': {},
            'integration_metadata': {
                'region': region,
                'bbox': bbox,
                'research_topic': research_topic,
                'collected_at': datetime.now().isoformat()
            }
        }
        
        try:
            # Collect Wikidata geological entities
            geological_entity_types = ['volcano', 'earthquake', 'geological_formation', 'mineral']
            
            for entity_type in geological_entity_types:
                entities = await self.wikidata.query_geological_entities(
                    entity_type=entity_type,
                    country=region,
                    limit=50
                )
                comprehensive_data['wikidata_entities'][entity_type] = entities
            
            # Collect GeoNames geographical features
            if bbox:
                geonames_features = await self.geonames.get_geological_features(
                    bbox=bbox,
                    max_rows=100
                )
                comprehensive_data['geonames_features'] = geonames_features
            
            # Collect Connected Papers research data
            if research_topic:
                papers = await self.connected_papers.search_papers(
                    query=f"geology {research_topic}",
                    limit=20
                )
                comprehensive_data['connected_papers']['search_results'] = papers
                
                # Get trending papers
                trending = await self.connected_papers.get_trending_papers(
                    field="geology",
                    time_period="month"
                )
                comprehensive_data['connected_papers']['trending'] = trending[:10]
            
            logger.info(f"Collected comprehensive geological data: "
                       f"{sum(len(entities) for entities in comprehensive_data['wikidata_entities'].values())} Wikidata entities, "
                       f"{len(comprehensive_data['geonames_features'])} GeoNames features, "
                       f"{len(comprehensive_data['connected_papers'].get('search_results', []))} research papers")
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Error collecting comprehensive geological data: {e}")
            return comprehensive_data

# Example usage
if __name__ == "__main__":
    async def test_connectors():
        """Test all additional data connectors"""
        
        # Test Wikidata connector
        print("Testing Wikidata connector...")
        wikidata = WikidataConnector()
        volcanoes = await wikidata.query_geological_entities("volcano", limit=5)
        print(f"Found {len(volcanoes)} volcanoes from Wikidata")
        
        # Test GeoNames connector  
        print("Testing GeoNames connector...")
        geonames = GeoNamesConnector(username="demo")
        features = await geonames.search_geographical_features("volcano", max_rows=5)
        print(f"Found {len(features)} geographical features from GeoNames")
        
        # Test Connected Papers connector
        print("Testing Connected Papers connector...")
        connected_papers = ConnectedPapersConnector()  # Requires API key for full functionality
        papers = await connected_papers.search_papers("volcano geology", limit=5)
        print(f"Found {len(papers)} papers from Connected Papers")
        
        # Test integration manager
        print("Testing integration manager...")
        config = {
            'wikidata': {'rate_limit': 0.5},
            'geonames': {'username': 'demo', 'rate_limit': 1.0},
            'connected_papers': {'rate_limit': 1.0}
        }
        
        manager = AdditionalDataIntegrationManager(config)
        comprehensive_data = await manager.collect_comprehensive_geological_data(
            region="Italy",
            bbox=(12.0, 36.0, 20.0, 47.0),  # Italy bounding box
            research_topic="volcanic eruption"
        )
        
        print(f"Comprehensive data collection completed:")
        print(f"- Wikidata entities: {sum(len(entities) for entities in comprehensive_data['wikidata_entities'].values())}")
        print(f"- GeoNames features: {len(comprehensive_data['geonames_features'])}")
        print(f"- Research papers: {len(comprehensive_data['connected_papers'].get('search_results', []))}")
    
    # Run the test
    asyncio.run(test_connectors())