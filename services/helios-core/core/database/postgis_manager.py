"""
PostgreSQL with PostGIS Manager for Prometheus AI Platform

This module provides advanced spatial-temporal database operations for geological data,
including complex queries, materialized view management, and geological calculations.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import asyncpg
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GeologicalLocation:
    """Geological location with spatial coordinates"""

    id: Optional[int]
    name: str
    latitude: float
    longitude: float
    elevation: Optional[float] = None
    depth: Optional[float] = None
    coordinate_system: str = "WGS84"


@dataclass
class SeismicEvent:
    """Seismic event data structure"""

    id: Optional[int]
    event_time: datetime
    location_id: int
    magnitude: float
    depth: float
    event_type: str
    confidence: float = 1.0
    waveform_data: Optional[bytes] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DeformationMeasurement:
    """Ground deformation measurement"""

    id: Optional[int]
    measurement_time: datetime
    location_id: int
    displacement_x: float
    displacement_y: float
    displacement_z: float
    uncertainty_x: Optional[float] = None
    uncertainty_y: Optional[float] = None
    uncertainty_z: Optional[float] = None
    measurement_type: str = "GPS"
    reference_frame: str = "ITRF2014"


@dataclass
class SpatialTemporalQuery:
    """Parameters for spatial-temporal queries"""

    center_lat: float
    center_lon: float
    spatial_radius: float  # meters
    time_center: datetime
    time_window: timedelta
    event_types: List[str] = None


class PostGISManager:
    """Advanced PostgreSQL with PostGIS manager for geological data"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string, min_size=5, max_size=20, command_timeout=60
            )
            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")

    async def create_geological_location(self, location: GeologicalLocation) -> int:
        """Create a new geological location with spatial indexing"""
        async with self.pool.acquire() as conn:
            query = """
            INSERT INTO geological.locations
            (name, latitude, longitude, elevation, depth, coordinate_system, geom, geom_3d)
            VALUES ($1, $2, $3, $4, $5, $6,
                    ST_SetSRID(ST_MakePoint($3, $2), 4326),
                    ST_SetSRID(ST_MakePoint($3, $2, COALESCE($4, 0)), 4326))
            RETURNING id
            """
            location_id = await conn.fetchval(
                query,
                location.name,
                location.latitude,
                location.longitude,
                location.elevation,
                location.depth,
                location.coordinate_system,
            )

            # Update UTM coordinates for advanced spatial analysis
            await conn.execute(
                """
                UPDATE geological.locations
                SET geom_utm = ST_Transform(geom, 9001)
                WHERE id = $1
            """,
                location_id,
            )

            logger.info(
                f"Created geological location {location.name} with ID {location_id}"
            )
            return location_id

    async def insert_seismic_event(self, event: SeismicEvent) -> int:
        """Insert seismic event with spatial-temporal indexing"""
        async with self.pool.acquire() as conn:
            query = """
            INSERT INTO geological.seismic_events
            (event_time, location_id, magnitude, depth, event_type, confidence,
             waveform_data, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
            """
            event_id = await conn.fetchval(
                query,
                event.event_time,
                event.location_id,
                event.magnitude,
                event.depth,
                event.event_type,
                event.confidence,
                event.waveform_data,
                json.dumps(event.metadata) if event.metadata else None,
            )

            logger.info(f"Inserted seismic event with ID {event_id}")
            return event_id

    async def insert_deformation_measurement(
        self, measurement: DeformationMeasurement
    ) -> int:
        """Insert deformation measurement with spatial-temporal indexing"""
        async with self.pool.acquire() as conn:
            query = """
            INSERT INTO geological.deformation_measurements
            (measurement_time, location_id, displacement_x, displacement_y, displacement_z,
             uncertainty_x, uncertainty_y, uncertainty_z, measurement_type, reference_frame)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
            """
            measurement_id = await conn.fetchval(
                query,
                measurement.measurement_time,
                measurement.location_id,
                measurement.displacement_x,
                measurement.displacement_y,
                measurement.displacement_z,
                measurement.uncertainty_x,
                measurement.uncertainty_y,
                measurement.uncertainty_z,
                measurement.measurement_type,
                measurement.reference_frame,
            )

            logger.info(f"Inserted deformation measurement with ID {measurement_id}")
            return measurement_id

    async def find_events_in_spacetime(
        self, query_params: SpatialTemporalQuery
    ) -> List[Dict[str, Any]]:
        """Find geological events within spatial-temporal window using advanced indexing"""
        async with self.pool.acquire() as conn:
            center_point = f"POINT({query_params.center_lon} {query_params.center_lat})"
            event_types = query_params.event_types or ["earthquake", "volcanic"]

            sql_query = """
            SELECT * FROM geological.find_events_in_spacetime(
                ST_SetSRID(ST_GeomFromText($1), 4326),
                $2, $3, $4, $5
            )
            """

            rows = await conn.fetch(
                sql_query,
                center_point,
                query_params.spatial_radius,
                query_params.time_center,
                query_params.time_window,
                event_types,
            )

            return [dict(row) for row in rows]

    async def calculate_geological_distance(
        self, loc1_id: int, loc2_id: int, distance_type: str = "euclidean"
    ) -> Optional[float]:
        """Calculate distance between geological locations using stored procedure"""
        async with self.pool.acquire() as conn:
            distance = await conn.fetchval(
                "SELECT geological.calculate_distance($1, $2, $3)",
                loc1_id,
                loc2_id,
                distance_type,
            )
            return distance

    async def calculate_seismic_moment(
        self, magnitude: float, magnitude_type: str = "Mw"
    ) -> float:
        """Calculate seismic moment from magnitude using geological calculations"""
        async with self.pool.acquire() as conn:
            moment = await conn.fetchval(
                "SELECT geological.calculate_seismic_moment($1, $2)",
                magnitude,
                magnitude_type,
            )
            return moment

    async def estimate_vei(
        self, eruption_volume: float, eruption_height: Optional[float] = None
    ) -> int:
        """Estimate Volcanic Explosivity Index using stored procedure"""
        async with self.pool.acquire() as conn:
            vei = await conn.fetchval(
                "SELECT geological.estimate_vei($1, $2)",
                eruption_volume,
                eruption_height,
            )
            return vei

    async def calculate_hazard_probability(
        self,
        latitude: float,
        longitude: float,
        hazard_type: str,
        time_horizon: timedelta,
    ) -> float:
        """Calculate geological hazard probability for a location"""
        async with self.pool.acquire() as conn:
            point = f"POINT({longitude} {latitude})"
            probability = await conn.fetchval(
                """
                SELECT geological.calculate_hazard_probability(
                    ST_SetSRID(ST_GeomFromText($1), 4326), $2, $3
                )
            """,
                point,
                hazard_type,
                time_horizon,
            )
            return probability

    async def get_seismic_activity_summary(
        self, days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Get seismic activity summary using materialized view"""
        async with self.pool.acquire() as conn:
            query = """
            SELECT
                activity_date,
                event_type,
                event_count,
                avg_magnitude,
                max_magnitude,
                min_magnitude,
                magnitude_stddev,
                avg_depth,
                ST_AsText(centroid_location) as centroid_wkt,
                ST_AsText(activity_envelope) as envelope_wkt
            FROM geological.mv_seismic_activity_summary
            WHERE activity_date >= $1
            ORDER BY activity_date DESC, event_type
            """

            rows = await conn.fetch(query, datetime.now() - timedelta(days=days_back))
            return [dict(row) for row in rows]

    async def get_deformation_analysis(
        self, location_id: Optional[int] = None, months_back: int = 12
    ) -> List[Dict[str, Any]]:
        """Get deformation analysis using materialized view"""
        async with self.pool.acquire() as conn:
            base_query = """
            SELECT
                location_id,
                location_name,
                ST_AsText(geom) as location_wkt,
                analysis_month,
                measurement_type,
                measurement_count,
                avg_total_displacement,
                max_total_displacement,
                avg_displacement_x,
                avg_displacement_y,
                avg_displacement_z,
                stddev_displacement_x,
                stddev_displacement_y,
                stddev_displacement_z
            FROM geological.mv_deformation_analysis
            WHERE analysis_month >= $1
            """

            params = [datetime.now() - timedelta(days=months_back * 30)]

            if location_id:
                base_query += " AND location_id = $2"
                params.append(location_id)

            base_query += " ORDER BY analysis_month DESC, location_name"

            rows = await conn.fetch(base_query, *params)
            return [dict(row) for row in rows]

    async def get_thermal_anomalies(
        self, anomaly_level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get thermal anomalies using materialized view"""
        async with self.pool.acquire() as conn:
            base_query = """
            SELECT
                location_id,
                location_name,
                ST_AsText(geom) as location_wkt,
                analysis_week,
                measurement_method,
                measurement_count,
                avg_temperature,
                max_temperature,
                min_temperature,
                temperature_stddev,
                anomaly_level
            FROM geological.mv_thermal_anomalies
            WHERE analysis_week >= $1
            """

            params = [datetime.now() - timedelta(days=90)]

            if anomaly_level:
                base_query += " AND anomaly_level = $2"
                params.append(anomaly_level)

            base_query += " ORDER BY analysis_week DESC, anomaly_level DESC"

            rows = await conn.fetch(base_query, *params)
            return [dict(row) for row in rows]

    async def get_geological_correlation(
        self, location_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get multi-parameter geological correlation using materialized view"""
        async with self.pool.acquire() as conn:
            base_query = """
            SELECT
                location_id,
                location_name,
                ST_AsText(geom) as location_wkt,
                elevation,
                depth,
                analysis_month,
                seismic_event_count,
                avg_seismic_magnitude,
                deformation_measurement_count,
                avg_total_deformation,
                thermal_measurement_count,
                avg_temperature,
                geochemical_sample_count
            FROM geological.mv_geological_correlation
            WHERE analysis_month >= $1
            """

            params = [datetime.now() - timedelta(days=365)]

            if location_id:
                base_query += " AND location_id = $2"
                params.append(location_id)

            base_query += " ORDER BY analysis_month DESC, location_name"

            rows = await conn.fetch(base_query, *params)
            return [dict(row) for row in rows]

    async def get_spatiotemporal_seismic_analysis(
        self, days_back: int = 365
    ) -> List[Dict[str, Any]]:
        """Get complex spatial-temporal seismic analysis"""
        async with self.pool.acquire() as conn:
            query = """
            SELECT
                id,
                event_time,
                magnitude,
                depth,
                event_type,
                location_name,
                ST_AsText(geom) as location_wkt,
                spatial_cluster_id,
                temporal_cluster_start,
                distance_to_previous,
                hours_since_previous,
                magnitude_diff
            FROM geological.v_seismic_spatiotemporal_analysis
            WHERE event_time >= $1
            ORDER BY event_time DESC
            """

            rows = await conn.fetch(query, datetime.now() - timedelta(days=days_back))
            return [dict(row) for row in rows]

    async def get_deformation_velocity_analysis(
        self, location_id: Optional[int] = None, days_back: int = 365
    ) -> List[Dict[str, Any]]:
        """Get deformation velocity analysis"""
        async with self.pool.acquire() as conn:
            base_query = """
            SELECT
                location_id,
                location_name,
                ST_AsText(geom) as location_wkt,
                time1,
                time2,
                days_between,
                velocity_x_mm_day,
                velocity_y_mm_day,
                velocity_z_mm_day,
                total_velocity_mm_day
            FROM geological.v_deformation_velocity_analysis
            WHERE time1 >= $1
            """

            params = [datetime.now() - timedelta(days=days_back)]

            if location_id:
                base_query += " AND location_id = $2"
                params.append(location_id)

            base_query += " ORDER BY time1 DESC"

            rows = await conn.fetch(base_query, *params)
            return [dict(row) for row in rows]

    async def refresh_materialized_views(self):
        """Refresh all materialized views for updated analysis"""
        async with self.pool.acquire() as conn:
            await conn.execute("SELECT geological.refresh_all_materialized_views()")
            logger.info("All geological materialized views refreshed")

    async def execute_spatial_query(
        self, query: str, params: List[Any] = None
    ) -> List[Dict[str, Any]]:
        """Execute custom spatial query with parameters"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *(params or []))
            return [dict(row) for row in rows]

    async def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics and performance metrics"""
        async with self.pool.acquire() as conn:
            stats = {}

            # Table row counts
            tables = [
                "geological.locations",
                "geological.seismic_events",
                "geological.deformation_measurements",
                "geological.thermal_measurements",
                "geological.geochemical_samples",
            ]

            for table in tables:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = count

            # Index usage statistics
            index_stats = await conn.fetch(
                """
                SELECT
                    schemaname,
                    tablename,
                    indexname,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes
                WHERE schemaname IN ('geological', 'monitoring')
                ORDER BY idx_tup_read DESC
                LIMIT 10
            """
            )

            stats["top_indexes"] = [dict(row) for row in index_stats]

            # Materialized view freshness
            mv_stats = await conn.fetch(
                """
                SELECT
                    schemaname,
                    matviewname,
                    ispopulated
                FROM pg_matviews
                WHERE schemaname = 'geological'
            """
            )

            stats["materialized_views"] = [dict(row) for row in mv_stats]

            return stats


# Example usage and testing functions
async def example_usage():
    """Example usage of PostGIS manager"""
    manager = PostGISManager(
        "postgresql://prometheus:password@localhost:5432/prometheus"
    )

    try:
        await manager.initialize()

        # Create a geological location
        location = GeologicalLocation(
            id=None,
            name="Mount St. Helens",
            latitude=46.1912,
            longitude=-122.1944,
            elevation=2549.0,
            depth=None,
        )
        location_id = await manager.create_geological_location(location)

        # Insert a seismic event
        event = SeismicEvent(
            id=None,
            event_time=datetime.now(),
            location_id=location_id,
            magnitude=3.2,
            depth=5000.0,
            event_type="volcanic",
            confidence=0.95,
            metadata={"source": "USGS", "quality": "A"},
        )
        await manager.insert_seismic_event(event)

        # Perform spatial-temporal query
        query_params = SpatialTemporalQuery(
            center_lat=46.1912,
            center_lon=-122.1944,
            spatial_radius=50000,  # 50km
            time_center=datetime.now(),
            time_window=timedelta(days=30),
            event_types=["volcanic", "earthquake"],
        )

        events = await manager.find_events_in_spacetime(query_params)
        print(f"Found {len(events)} events in spatial-temporal window")

        # Get analysis results
        seismic_summary = await manager.get_seismic_activity_summary(days_back=30)
        thermal_anomalies = await manager.get_thermal_anomalies(
            anomaly_level="HIGH_ANOMALY"
        )

        print(f"Seismic activity summary: {len(seismic_summary)} entries")
        print(f"High thermal anomalies: {len(thermal_anomalies)} entries")

        # Refresh materialized views
        await manager.refresh_materialized_views()

        # Get database statistics
        stats = await manager.get_database_statistics()
        print(f"Database statistics: {stats}")

    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(example_usage())
