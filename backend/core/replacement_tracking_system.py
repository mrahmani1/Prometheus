"""
Placeholder Replacement Tracking System
Tracks progress of placeholder replacements and maintains replacement history
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging

from .placeholder_detection_engine import PlaceholderInstance, Priority, PlaceholderType

logger = logging.getLogger(__name__)


class ReplacementStatus(Enum):
    """Status of placeholder replacement"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DEFERRED = "deferred"


@dataclass
class ReplacementTask:
    """Represents a placeholder replacement task"""
    task_id: str
    placeholder_instance: PlaceholderInstance
    assigned_developer: Optional[str] = None
    status: ReplacementStatus = ReplacementStatus.PENDING
    start_date: Optional[datetime] = None
    completion_date: Optional[datetime] = None
    estimated_hours: int = 0
    actual_hours: int = 0
    replacement_method: Optional[str] = None
    validation_status: Optional[str] = None
    notes: List[str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.notes is None:
            self.notes = []
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ReplacementProgress:
    """Progress tracking for replacement efforts"""
    total_placeholders: int
    completed_replacements: int
    in_progress_replacements: int
    failed_replacements: int
    completion_percentage: float
    estimated_completion_date: Optional[datetime]
    velocity_per_week: float
    current_sprint_progress: Dict[str, Any]


class ReplacementTrackingSystem:
    """System for tracking placeholder replacement progress"""
    
    def __init__(self, db_path: str = "backend/core/replacement_tracking.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS replacement_tasks (
                task_id TEXT PRIMARY KEY,
                module_path TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                placeholder_type TEXT NOT NULL,
                description TEXT NOT NULL,
                priority TEXT NOT NULL,
                assigned_developer TEXT,
                status TEXT NOT NULL,
                start_date TEXT,
                completion_date TEXT,
                estimated_hours INTEGER,
                actual_hours INTEGER,
                replacement_method TEXT,
                validation_status TEXT,
                created_date TEXT NOT NULL,
                updated_date TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS replacement_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                note TEXT NOT NULL,
                created_date TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES replacement_tasks (task_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS replacement_dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                dependency_task_id TEXT NOT NULL,
                created_date TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES replacement_tasks (task_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS progress_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date TEXT NOT NULL,
                total_placeholders INTEGER NOT NULL,
                completed_replacements INTEGER NOT NULL,
                in_progress_replacements INTEGER NOT NULL,
                failed_replacements INTEGER NOT NULL,
                completion_percentage REAL NOT NULL,
                velocity_per_week REAL NOT NULL,
                snapshot_data TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sprint_planning (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sprint_name TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                planned_tasks TEXT NOT NULL,
                actual_completion TEXT,
                sprint_velocity REAL,
                created_date TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_replacement_tasks(self, placeholders: List[PlaceholderInstance]) -> List[ReplacementTask]:
        """Create replacement tasks from detected placeholders"""
        tasks = []
        
        for placeholder in placeholders:
            task_id = self._generate_task_id(placeholder)
            
            task = ReplacementTask(
                task_id=task_id,
                placeholder_instance=placeholder,
                estimated_hours=placeholder.impact_assessment.get('estimated_effort_hours', 8),
                replacement_method=self._suggest_replacement_method(placeholder),
                dependencies=self._identify_dependencies(placeholder, placeholders)
            )
            
            tasks.append(task)
            self._save_task_to_db(task)
        
        logger.info(f"Created {len(tasks)} replacement tasks")
        return tasks
    
    def _generate_task_id(self, placeholder: PlaceholderInstance) -> str:
        """Generate unique task ID for placeholder"""
        module_name = Path(placeholder.module_path).stem
        return f"{module_name}_{placeholder.line_number}_{placeholder.placeholder_type.value}"
    
    def _suggest_replacement_method(self, placeholder: PlaceholderInstance) -> str:
        """Suggest replacement method based on placeholder type"""
        method_map = {
            PlaceholderType.MOCK_DATA: "integrate_real_data_source",
            PlaceholderType.SYNTHETIC_DATA: "replace_with_authentic_data",
            PlaceholderType.SIMPLIFIED_ALGORITHM: "implement_production_algorithm",
            PlaceholderType.TODO_IMPLEMENTATION: "complete_implementation",
            PlaceholderType.EMPTY_IMPLEMENTATION: "implement_full_functionality",
            PlaceholderType.HARDCODED_VALUES: "parameterize_and_configure",
            PlaceholderType.RANDOM_DATA_GENERATION: "connect_to_real_data_stream",
            PlaceholderType.PLACEHOLDER_RETURN: "implement_proper_logic",
            PlaceholderType.INCOMPLETE_INTERFACE: "complete_interface_implementation",
            PlaceholderType.SIMULATION_CODE: "replace_with_real_data_processing"
        }
        
        return method_map.get(placeholder.placeholder_type, "manual_review_required")
    
    def _identify_dependencies(self, placeholder: PlaceholderInstance, 
                             all_placeholders: List[PlaceholderInstance]) -> List[str]:
        """Identify dependencies between replacement tasks"""
        dependencies = []
        
        # Check for module-level dependencies
        current_module = placeholder.module_path
        
        for other in all_placeholders:
            if other.module_path == current_module and other != placeholder:
                # Same module dependencies
                if (other.placeholder_type == PlaceholderType.MOCK_DATA and 
                    placeholder.placeholder_type != PlaceholderType.MOCK_DATA):
                    # Data source should be replaced first
                    dependencies.append(self._generate_task_id(other))
        
        return dependencies
    
    def _save_task_to_db(self, task: ReplacementTask):
        """Save replacement task to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT OR REPLACE INTO replacement_tasks 
            (task_id, module_path, file_path, line_number, placeholder_type, description,
             priority, assigned_developer, status, start_date, completion_date,
             estimated_hours, actual_hours, replacement_method, validation_status,
             created_date, updated_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.task_id,
            task.placeholder_instance.module_path,
            task.placeholder_instance.file_path,
            task.placeholder_instance.line_number,
            task.placeholder_instance.placeholder_type.value,
            task.placeholder_instance.description,
            task.placeholder_instance.replacement_priority.value,
            task.assigned_developer,
            task.status.value,
            task.start_date.isoformat() if task.start_date else None,
            task.completion_date.isoformat() if task.completion_date else None,
            task.estimated_hours,
            task.actual_hours,
            task.replacement_method,
            task.validation_status,
            now,
            now
        ))
        
        # Save notes
        for note in task.notes:
            cursor.execute("""
                INSERT INTO replacement_notes (task_id, note, created_date)
                VALUES (?, ?, ?)
            """, (task.task_id, note, now))
        
        # Save dependencies
        for dep_id in task.dependencies:
            cursor.execute("""
                INSERT INTO replacement_dependencies (task_id, dependency_task_id, created_date)
                VALUES (?, ?, ?)
            """, (task.task_id, dep_id, now))
        
        conn.commit()
        conn.close()
    
    def update_task_status(self, task_id: str, status: ReplacementStatus, 
                          developer: Optional[str] = None, notes: Optional[str] = None,
                          actual_hours: Optional[int] = None) -> bool:
        """Update task status and related information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now()
        
        # Get current task
        cursor.execute("SELECT * FROM replacement_tasks WHERE task_id = ?", (task_id,))
        task_row = cursor.fetchone()
        
        if not task_row:
            logger.error(f"Task {task_id} not found")
            conn.close()
            return False
        
        # Update task
        update_fields = ["status = ?", "updated_date = ?"]
        update_values = [status.value, now.isoformat()]
        
        if developer:
            update_fields.append("assigned_developer = ?")
            update_values.append(developer)
        
        if actual_hours is not None:
            update_fields.append("actual_hours = ?")
            update_values.append(actual_hours)
        
        if status == ReplacementStatus.IN_PROGRESS and not task_row[9]:  # start_date
            update_fields.append("start_date = ?")
            update_values.append(now.isoformat())
        elif status == ReplacementStatus.COMPLETED:
            update_fields.append("completion_date = ?")
            update_values.append(now.isoformat())
        
        update_values.append(task_id)
        
        cursor.execute(f"""
            UPDATE replacement_tasks 
            SET {', '.join(update_fields)}
            WHERE task_id = ?
        """, update_values)
        
        # Add note if provided
        if notes:
            cursor.execute("""
                INSERT INTO replacement_notes (task_id, note, created_date)
                VALUES (?, ?, ?)
            """, (task_id, notes, now.isoformat()))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated task {task_id} to status {status.value}")
        return True
    
    def get_replacement_progress(self) -> ReplacementProgress:
        """Get current replacement progress"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get overall statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
            FROM replacement_tasks
        """)
        
        stats = cursor.fetchone()
        total, completed, in_progress, failed = stats
        
        completion_percentage = (completed / total * 100) if total > 0 else 0
        
        # Calculate velocity (completions per week)
        cursor.execute("""
            SELECT COUNT(*) 
            FROM replacement_tasks 
            WHERE status = 'completed' 
            AND completion_date >= date('now', '-7 days')
        """)
        
        weekly_completions = cursor.fetchone()[0]
        velocity_per_week = weekly_completions
        
        # Estimate completion date
        remaining_tasks = total - completed
        estimated_weeks = remaining_tasks / max(velocity_per_week, 1)
        estimated_completion = datetime.now() + timedelta(weeks=estimated_weeks)
        
        # Get current sprint progress
        current_sprint = self._get_current_sprint_progress(cursor)
        
        conn.close()
        
        return ReplacementProgress(
            total_placeholders=total,
            completed_replacements=completed,
            in_progress_replacements=in_progress,
            failed_replacements=failed,
            completion_percentage=completion_percentage,
            estimated_completion_date=estimated_completion if remaining_tasks > 0 else None,
            velocity_per_week=velocity_per_week,
            current_sprint_progress=current_sprint
        )
    
    def _get_current_sprint_progress(self, cursor) -> Dict[str, Any]:
        """Get current sprint progress information"""
        # Get latest sprint
        cursor.execute("""
            SELECT sprint_name, start_date, end_date, planned_tasks
            FROM sprint_planning 
            ORDER BY start_date DESC 
            LIMIT 1
        """)
        
        sprint_row = cursor.fetchone()
        if not sprint_row:
            return {"message": "No active sprint"}
        
        sprint_name, start_date, end_date, planned_tasks_json = sprint_row
        planned_tasks = json.loads(planned_tasks_json)
        
        # Get completion status for planned tasks
        task_ids = [task['task_id'] for task in planned_tasks]
        placeholders = ', '.join(['?' for _ in task_ids])
        
        cursor.execute(f"""
            SELECT status, COUNT(*) 
            FROM replacement_tasks 
            WHERE task_id IN ({placeholders})
            GROUP BY status
        """, task_ids)
        
        status_counts = dict(cursor.fetchall())
        
        return {
            'sprint_name': sprint_name,
            'start_date': start_date,
            'end_date': end_date,
            'planned_tasks_count': len(planned_tasks),
            'completed_count': status_counts.get('completed', 0),
            'in_progress_count': status_counts.get('in_progress', 0),
            'sprint_completion_percentage': (status_counts.get('completed', 0) / len(planned_tasks) * 100) if planned_tasks else 0
        }
    
    def create_sprint(self, sprint_name: str, start_date: datetime, end_date: datetime,
                     task_priorities: List[Priority] = None) -> Dict[str, Any]:
        """Create a new sprint with selected tasks"""
        if task_priorities is None:
            task_priorities = [Priority.CRITICAL, Priority.HIGH]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get available tasks for sprint
        priority_placeholders = ', '.join([f"'{p.value}'" for p in task_priorities])
        
        cursor.execute(f"""
            SELECT task_id, description, priority, estimated_hours
            FROM replacement_tasks 
            WHERE status = 'pending' 
            AND priority IN ({priority_placeholders})
            ORDER BY 
                CASE priority 
                    WHEN 'critical' THEN 1 
                    WHEN 'high' THEN 2 
                    WHEN 'medium' THEN 3 
                    WHEN 'low' THEN 4 
                END,
                estimated_hours DESC
        """)
        
        available_tasks = cursor.fetchall()
        
        # Select tasks for sprint (assuming 40 hours per week capacity)
        sprint_duration_weeks = (end_date - start_date).days / 7
        sprint_capacity = int(sprint_duration_weeks * 40)  # 40 hours per week
        
        selected_tasks = []
        total_hours = 0
        
        for task_id, description, priority, estimated_hours in available_tasks:
            if total_hours + estimated_hours <= sprint_capacity:
                selected_tasks.append({
                    'task_id': task_id,
                    'description': description,
                    'priority': priority,
                    'estimated_hours': estimated_hours
                })
                total_hours += estimated_hours
        
        # Save sprint
        cursor.execute("""
            INSERT INTO sprint_planning 
            (sprint_name, start_date, end_date, planned_tasks, created_date)
            VALUES (?, ?, ?, ?, ?)
        """, (
            sprint_name,
            start_date.isoformat(),
            end_date.isoformat(),
            json.dumps(selected_tasks),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Created sprint '{sprint_name}' with {len(selected_tasks)} tasks ({total_hours} hours)")
        
        return {
            'sprint_name': sprint_name,
            'start_date': start_date,
            'end_date': end_date,
            'selected_tasks': selected_tasks,
            'total_estimated_hours': total_hours,
            'capacity_utilization': (total_hours / sprint_capacity * 100) if sprint_capacity > 0 else 0
        }
    
    def get_task_details(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific task"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get task details
        cursor.execute("SELECT * FROM replacement_tasks WHERE task_id = ?", (task_id,))
        task_row = cursor.fetchone()
        
        if not task_row:
            conn.close()
            return None
        
        # Get notes
        cursor.execute("""
            SELECT note, created_date 
            FROM replacement_notes 
            WHERE task_id = ? 
            ORDER BY created_date DESC
        """, (task_id,))
        notes = cursor.fetchall()
        
        # Get dependencies
        cursor.execute("""
            SELECT dependency_task_id 
            FROM replacement_dependencies 
            WHERE task_id = ?
        """, (task_id,))
        dependencies = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        # Convert to dictionary
        columns = [
            'task_id', 'module_path', 'file_path', 'line_number', 'placeholder_type',
            'description', 'priority', 'assigned_developer', 'status', 'start_date',
            'completion_date', 'estimated_hours', 'actual_hours', 'replacement_method',
            'validation_status', 'created_date', 'updated_date'
        ]
        
        task_dict = dict(zip(columns, task_row))
        task_dict['notes'] = [{'note': note, 'created_date': date} for note, date in notes]
        task_dict['dependencies'] = dependencies
        
        return task_dict
    
    def get_tasks_by_status(self, status: ReplacementStatus) -> List[Dict[str, Any]]:
        """Get all tasks with specific status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT task_id, module_path, description, priority, assigned_developer,
                   estimated_hours, actual_hours, created_date, updated_date
            FROM replacement_tasks 
            WHERE status = ?
            ORDER BY priority, created_date
        """, (status.value,))
        
        tasks = []
        columns = ['task_id', 'module_path', 'description', 'priority', 'assigned_developer',
                  'estimated_hours', 'actual_hours', 'created_date', 'updated_date']
        
        for row in cursor.fetchall():
            tasks.append(dict(zip(columns, row)))
        
        conn.close()
        return tasks
    
    def get_developer_workload(self, developer: str) -> Dict[str, Any]:
        """Get workload information for a specific developer"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                status,
                COUNT(*) as task_count,
                SUM(estimated_hours) as estimated_hours,
                SUM(actual_hours) as actual_hours
            FROM replacement_tasks 
            WHERE assigned_developer = ?
            GROUP BY status
        """, (developer,))
        
        workload = {}
        for status, count, est_hours, act_hours in cursor.fetchall():
            workload[status] = {
                'task_count': count,
                'estimated_hours': est_hours or 0,
                'actual_hours': act_hours or 0
            }
        
        conn.close()
        return workload
    
    def generate_progress_report(self) -> Dict[str, Any]:
        """Generate comprehensive progress report"""
        progress = self.get_replacement_progress()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get progress by module
        cursor.execute("""
            SELECT 
                module_path,
                COUNT(*) as total_tasks,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_tasks,
                AVG(CASE WHEN status = 'completed' THEN actual_hours ELSE NULL END) as avg_hours
            FROM replacement_tasks
            GROUP BY module_path
            ORDER BY total_tasks DESC
        """)
        
        module_progress = []
        for module, total, completed, avg_hours in cursor.fetchall():
            module_progress.append({
                'module': Path(module).name,
                'total_tasks': total,
                'completed_tasks': completed,
                'completion_percentage': (completed / total * 100) if total > 0 else 0,
                'average_hours_per_task': round(avg_hours or 0, 1)
            })
        
        # Get progress by priority
        cursor.execute("""
            SELECT 
                priority,
                COUNT(*) as total_tasks,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_tasks
            FROM replacement_tasks
            GROUP BY priority
        """)
        
        priority_progress = {}
        for priority, total, completed in cursor.fetchall():
            priority_progress[priority] = {
                'total_tasks': total,
                'completed_tasks': completed,
                'completion_percentage': (completed / total * 100) if total > 0 else 0
            }
        
        conn.close()
        
        return {
            'overall_progress': asdict(progress),
            'module_breakdown': module_progress,
            'priority_breakdown': priority_progress,
            'report_generated': datetime.now().isoformat()
        }
    
    def save_progress_snapshot(self):
        """Save current progress as a snapshot for historical tracking"""
        progress = self.get_replacement_progress()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO progress_snapshots 
            (snapshot_date, total_placeholders, completed_replacements, 
             in_progress_replacements, failed_replacements, completion_percentage,
             velocity_per_week, snapshot_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            progress.total_placeholders,
            progress.completed_replacements,
            progress.in_progress_replacements,
            progress.failed_replacements,
            progress.completion_percentage,
            progress.velocity_per_week,
            json.dumps(asdict(progress), default=str)
        ))
        
        conn.commit()
        conn.close()
        
        logger.info("Progress snapshot saved")