"""Database backup utilities."""
import os
import shutil
import json
from datetime import datetime
from typing import Optional, Dict, Any
from config import Config


class BackupManager:
    """
    Manage database backups.
    
    Supports:
    - SQLite file backups
    - JSON export for PostgreSQL
    - Scheduled backup capability
    """
    
    def __init__(self, backup_dir: str = "./backups"):
        self.backup_dir = backup_dir
        self._ensure_backup_dir()
    
    def _ensure_backup_dir(self):
        """Ensure backup directory exists."""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
    
    def backup_sqlite(self, db_path: str = None) -> Optional[str]:
        """
        Create a backup of SQLite database.
        
        Args:
            db_path: Path to SQLite database file
            
        Returns:
            Path to backup file, or None if failed
        """
        if db_path is None:
            # Extract path from DATABASE_URL
            db_url = Config.DATABASE_URL
            if db_url.startswith("sqlite:///"):
                db_path = db_url.replace("sqlite:///", "")
            else:
                print("Not using SQLite database")
                return None
        
        if not os.path.exists(db_path):
            print(f"Database file not found: {db_path}")
            return None
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"backup_{timestamp}.db"
        backup_path = os.path.join(self.backup_dir, backup_filename)
        
        try:
            shutil.copy2(db_path, backup_path)
            print(f"Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"Backup failed: {e}")
            return None
    
    def export_to_json(self, db) -> Optional[str]:
        """
        Export database to JSON format (works with any database).
        
        Args:
            db: Database instance
            
        Returns:
            Path to JSON export file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"export_{timestamp}.json"
        export_path = os.path.join(self.backup_dir, export_filename)
        
        try:
            # Export users (without passwords)
            from database import User, PromptTemplate, EvaluationResultDB
            
            session = db.SessionLocal()
            
            # Get all users (excluding sensitive data)
            users = session.query(User).all()
            users_data = [
                {
                    "id": u.id,
                    "email": u.email,
                    "username": u.username,
                    "created_at": u.created_at.isoformat() if u.created_at else None,
                    "email_verified": u.email_verified
                }
                for u in users
            ]
            
            # Get all templates
            templates = session.query(PromptTemplate).all()
            templates_data = [t.to_dict() for t in templates]
            
            # Get all evaluation results
            results = session.query(EvaluationResultDB).limit(10000).all()
            results_data = []
            for r in results:
                results_data.append({
                    "id": r.id,
                    "user_id": r.user_id,
                    "prompt": r.prompt[:500] if r.prompt else None,  # Truncate for space
                    "model_name": r.model_name,
                    "prompt_version": r.prompt_version,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "is_hallucination": r.is_hallucination,
                    "overall_hallucination_score": r.overall_hallucination_score
                })
            
            session.close()
            
            # Create export data
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "database_type": "postgresql" if db.is_postgresql() else "sqlite",
                "stats": {
                    "users": len(users_data),
                    "templates": len(templates_data),
                    "evaluations": len(results_data)
                },
                "users": users_data,
                "templates": templates_data,
                "evaluations": results_data
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Export created: {export_path}")
            return export_path
            
        except Exception as e:
            print(f"Export failed: {e}")
            return None
    
    def list_backups(self) -> list:
        """List all available backups."""
        backups = []
        
        if not os.path.exists(self.backup_dir):
            return backups
        
        for filename in os.listdir(self.backup_dir):
            filepath = os.path.join(self.backup_dir, filename)
            if os.path.isfile(filepath):
                stat = os.stat(filepath)
                backups.append({
                    "filename": filename,
                    "path": filepath,
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x['created'], reverse=True)
        return backups
    
    def restore_sqlite(self, backup_path: str, db_path: str = None) -> bool:
        """
        Restore SQLite database from backup.
        
        Args:
            backup_path: Path to backup file
            db_path: Target database path
            
        Returns:
            True if restore successful
        """
        if db_path is None:
            db_url = Config.DATABASE_URL
            if db_url.startswith("sqlite:///"):
                db_path = db_url.replace("sqlite:///", "")
            else:
                print("Not using SQLite database")
                return False
        
        if not os.path.exists(backup_path):
            print(f"Backup file not found: {backup_path}")
            return False
        
        try:
            # Create backup of current database before restore
            if os.path.exists(db_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pre_restore_backup = os.path.join(
                    self.backup_dir, 
                    f"pre_restore_{timestamp}.db"
                )
                shutil.copy2(db_path, pre_restore_backup)
            
            # Restore from backup
            shutil.copy2(backup_path, db_path)
            print(f"Database restored from: {backup_path}")
            return True
            
        except Exception as e:
            print(f"Restore failed: {e}")
            return False
    
    def cleanup_old_backups(self, keep_count: int = 10):
        """
        Remove old backups, keeping the most recent ones.
        
        Args:
            keep_count: Number of backups to keep
        """
        backups = self.list_backups()
        
        if len(backups) <= keep_count:
            return
        
        # Remove oldest backups
        for backup in backups[keep_count:]:
            try:
                os.remove(backup['path'])
                print(f"Removed old backup: {backup['filename']}")
            except Exception as e:
                print(f"Failed to remove {backup['filename']}: {e}")


def create_backup(db) -> Dict[str, Any]:
    """
    Create a backup of the database.
    
    Args:
        db: Database instance
        
    Returns:
        Backup information
    """
    manager = BackupManager()
    
    if db.is_postgresql():
        # For PostgreSQL, export to JSON
        export_path = manager.export_to_json(db)
        return {
            "type": "json_export",
            "path": export_path,
            "success": export_path is not None
        }
    else:
        # For SQLite, copy the file
        backup_path = manager.backup_sqlite()
        return {
            "type": "sqlite_backup",
            "path": backup_path,
            "success": backup_path is not None
        }


def get_backup_manager() -> BackupManager:
    """Get BackupManager instance."""
    return BackupManager()

