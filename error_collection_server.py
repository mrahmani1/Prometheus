#!/usr/bin/env python3
"""
Prometheus Error Collection Server
Collects and stores error reports from remote monitoring agents
"""

import json
import os
import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
import threading
import time
from urllib.parse import urlparse

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ErrorReport:
    id: str
    type: str
    message: str
    timestamp: str
    url: str
    user_agent: str
    session_id: str
    user_id: str
    severity: str
    stack: Optional[str] = None
    filename: Optional[str] = None
    lineno: Optional[int] = None
    colno: Optional[int] = None
    page_info: Optional[Dict] = None
    browser_info: Optional[Dict] = None
    screen_info: Optional[Dict] = None
    viewport_info: Optional[Dict] = None
    additional_data: Optional[Dict] = None

class ErrorDatabase:
    """SQLite database manager for error storage"""
    
    def __init__(self, db_path: str = "prometheus_errors.db"):
        self.db_path = db_path
        self.init_database()
        self.lock = threading.Lock()
    
    def init_database(self):
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    url TEXT NOT NULL,
                    user_agent TEXT,
                    session_id TEXT,
                    user_id TEXT,
                    severity TEXT DEFAULT 'low',
                    stack TEXT,
                    filename TEXT,
                    lineno INTEGER,
                    colno INTEGER,
                    page_info TEXT,
                    browser_info TEXT,
                    screen_info TEXT,
                    viewport_info TEXT,
                    additional_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    start_time TEXT,
                    last_activity TEXT,
                    error_count INTEGER DEFAULT 0,
                    user_agent TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_errors INTEGER,
                    active_sessions INTEGER,
                    error_rate REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON errors(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_severity ON errors(severity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON errors(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON errors(type)")
            
            conn.commit()
    
    def insert_error(self, error: ErrorReport) -> bool:
        """Insert an error record into the database"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO errors (
                            id, type, message, timestamp, url, user_agent, session_id, user_id,
                            severity, stack, filename, lineno, colno, page_info, browser_info,
                            screen_info, viewport_info, additional_data
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        error.id, error.type, error.message, error.timestamp, error.url,
                        error.user_agent, error.session_id, error.user_id, error.severity,
                        error.stack, error.filename, error.lineno, error.colno,
                        json.dumps(error.page_info) if error.page_info else None,
                        json.dumps(error.browser_info) if error.browser_info else None,
                        json.dumps(error.screen_info) if error.screen_info else None,
                        json.dumps(error.viewport_info) if error.viewport_info else None,
                        json.dumps(error.additional_data) if error.additional_data else None
                    ))
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to insert error: {e}")
            return False
    
    def get_errors(self, limit: int = 100, offset: int = 0, filters: Dict = None) -> List[Dict]:
        """Retrieve errors from the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = "SELECT * FROM errors"
                params = []
                
                if filters:
                    conditions = []
                    if filters.get('severity'):
                        conditions.append("severity = ?")
                        params.append(filters['severity'])
                    if filters.get('type'):
                        conditions.append("type LIKE ?")
                        params.append(f"%{filters['type']}%")
                    if filters.get('session_id'):
                        conditions.append("session_id = ?")
                        params.append(filters['session_id'])
                    if filters.get('since'):
                        conditions.append("timestamp >= ?")
                        params.append(filters['since'])
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor = conn.execute(query, params)
                errors = []
                
                for row in cursor.fetchall():
                    error_dict = dict(row)
                    # Parse JSON fields
                    for field in ['page_info', 'browser_info', 'screen_info', 'viewport_info', 'additional_data']:
                        if error_dict[field]:
                            try:
                                error_dict[field] = json.loads(error_dict[field])
                            except json.JSONDecodeError:
                                error_dict[field] = None
                    errors.append(error_dict)
                
                return errors
                
        except Exception as e:
            logger.error(f"Failed to retrieve errors: {e}")
            return []
    
    def update_session(self, session_id: str, user_id: str = None, user_agent: str = None):
        """Update or create session information"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    # Check if session exists
                    cursor = conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
                    exists = cursor.fetchone()
                    
                    if exists:
                        conn.execute("""
                            UPDATE sessions 
                            SET last_activity = CURRENT_TIMESTAMP, error_count = error_count + 1
                            WHERE session_id = ?
                        """, (session_id,))
                    else:
                        conn.execute("""
                            INSERT INTO sessions (session_id, user_id, start_time, last_activity, user_agent, error_count)
                            VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, 1)
                        """, (session_id, user_id, user_agent))
                    
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total errors
                total_errors = conn.execute("SELECT COUNT(*) FROM errors").fetchone()[0]
                
                # Active sessions (last 1 hour)
                active_sessions = conn.execute("""
                    SELECT COUNT(*) FROM sessions 
                    WHERE last_activity >= datetime('now', '-1 hour')
                """).fetchone()[0]
                
                # Error rate (errors per minute in last hour)
                recent_errors = conn.execute("""
                    SELECT COUNT(*) FROM errors 
                    WHERE timestamp >= datetime('now', '-1 hour')
                """).fetchone()[0]
                error_rate = recent_errors / 60.0  # errors per minute
                
                # Severity breakdown
                severity_stats = {}
                cursor = conn.execute("""
                    SELECT severity, COUNT(*) as count 
                    FROM errors 
                    WHERE timestamp >= datetime('now', '-24 hours')
                    GROUP BY severity
                """)
                for row in cursor.fetchall():
                    severity_stats[row[0]] = row[1]
                
                return {
                    'totalErrors': total_errors,
                    'activeUsers': active_sessions,
                    'errorRate': round(error_rate, 2),
                    'uptime': 0,  # This would need to be tracked separately
                    'severityStats': severity_stats,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

class PrometheusErrorServer:
    """Main server class for handling error collection"""
    
    def __init__(self, host='0.0.0.0', port=8080, db_path="prometheus_errors.db"):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.db = ErrorDatabase(db_path)
        
        # Configure CORS
        CORS(self.app, origins=['*'])
        
        # Setup routes
        self.setup_routes()
        
        # Start background tasks
        self.start_background_tasks()
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/api/errors', methods=['POST'])
        def collect_errors():
            """Collect error reports from clients"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400
                
                errors = data.get('errors', [])
                metadata = data.get('metadata', {})
                
                processed_count = 0
                for error_data in errors:
                    try:
                        # Generate unique ID if not provided
                        error_id = error_data.get('id', f"{metadata.get('sessionId', 'unknown')}_{int(time.time() * 1000)}")
                        
                        error = ErrorReport(
                            id=error_id,
                            type=error_data.get('type', 'unknown'),
                            message=error_data.get('message', ''),
                            timestamp=error_data.get('timestamp', datetime.datetime.now().isoformat()),
                            url=error_data.get('url', ''),
                            user_agent=metadata.get('userAgent', ''),
                            session_id=metadata.get('sessionId', ''),
                            user_id=metadata.get('userId', 'anonymous'),
                            severity=error_data.get('severity', 'low'),
                            stack=error_data.get('stack'),
                            filename=error_data.get('filename'),
                            lineno=error_data.get('lineno'),
                            colno=error_data.get('colno'),
                            page_info=error_data.get('page'),
                            browser_info=error_data.get('browser'),
                            screen_info=error_data.get('screen'),
                            viewport_info=error_data.get('viewport'),
                            additional_data=error_data.get('context')
                        )
                        
                        if self.db.insert_error(error):
                            processed_count += 1
                            
                            # Update session info
                            self.db.update_session(
                                error.session_id,
                                error.user_id,
                                error.user_agent
                            )
                    
                    except Exception as e:
                        logger.error(f"Failed to process error: {e}")
                
                logger.info(f"Processed {processed_count}/{len(errors)} errors")
                return jsonify({
                    'status': 'success',
                    'processed': processed_count,
                    'total': len(errors)
                })
                
            except Exception as e:
                logger.error(f"Error in collect_errors: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/errors', methods=['GET'])
        def get_errors():
            """Retrieve error reports"""
            try:
                limit = min(int(request.args.get('limit', 100)), 1000)  # Max 1000
                offset = int(request.args.get('offset', 0))
                
                filters = {}
                if request.args.get('severity'):
                    filters['severity'] = request.args.get('severity')
                if request.args.get('type'):
                    filters['type'] = request.args.get('type')
                if request.args.get('session_id'):
                    filters['session_id'] = request.args.get('session_id')
                if request.args.get('since'):
                    filters['since'] = request.args.get('since')
                
                errors = self.db.get_errors(limit, offset, filters)
                stats = self.db.get_stats()
                
                return jsonify({
                    'errors': errors,
                    'stats': stats,
                    'count': len(errors),
                    'limit': limit,
                    'offset': offset
                })
                
            except Exception as e:
                logger.error(f"Error in get_errors: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/stats', methods=['GET'])
        def get_stats():
            """Get system statistics"""
            try:
                stats = self.db.get_stats()
                return jsonify(stats)
            except Exception as e:
                logger.error(f"Error in get_stats: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.datetime.now().isoformat(),
                'version': '1.0.0'
            })
        
        @self.app.route('/')
        def dashboard():
            """Serve the debug dashboard"""
            try:
                with open('debug_dashboard.html', 'r') as f:
                    return f.read()
            except FileNotFoundError:
                return jsonify({'error': 'Dashboard not found'}), 404
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error'}), 500
    
    def start_background_tasks(self):
        """Start background maintenance tasks"""
        
        def cleanup_old_errors():
            """Periodically clean up old error records"""
            while True:
                try:
                    # Clean up errors older than 30 days
                    with sqlite3.connect(self.db.db_path) as conn:
                        deleted = conn.execute("""
                            DELETE FROM errors 
                            WHERE timestamp < datetime('now', '-30 days')
                        """).rowcount
                        
                        if deleted > 0:
                            logger.info(f"Cleaned up {deleted} old error records")
                        
                        conn.commit()
                        
                        # Also clean up old sessions
                        deleted_sessions = conn.execute("""
                            DELETE FROM sessions 
                            WHERE last_activity < datetime('now', '-7 days')
                        """).rowcount
                        
                        if deleted_sessions > 0:
                            logger.info(f"Cleaned up {deleted_sessions} old sessions")
                            
                        conn.commit()
                        
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
                
                # Wait 1 hour before next cleanup
                time.sleep(3600)
        
        # Start cleanup thread
        cleanup_thread = threading.Thread(target=cleanup_old_errors, daemon=True)
        cleanup_thread.start()
        
        logger.info("Background tasks started")
    
    def run(self, debug=False):
        """Start the Flask server"""
        logger.info(f"Starting Prometheus Error Collection Server on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prometheus Error Collection Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--db-path', default='prometheus_errors.db', help='Database path')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    server = PrometheusErrorServer(
        host=args.host,
        port=args.port,
        db_path=args.db_path
    )
    
    server.run(debug=args.debug)

if __name__ == '__main__':
    main()