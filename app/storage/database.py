"""
Local SQLite storage manager and image archiver for DefectSense Edge.
Zero cloud dependence: all logs, defect events, and images are kept locally.
"""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from app.vision.backend_base import InspectionResult

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Thread-safe SQLite inspection log and defect metrics database.
    """

    def __init__(
        self,
        db_path: str | Path = "data/defectsense.db",
        image_dir: str | Path = "data/inspections",
        save_all_images: bool = False,
        thumbnail_size: tuple[int, int] = (160, 160),
    ):
        self.db_path = Path(db_path)
        self.image_dir = Path(image_dir)
        self.save_all_images = save_all_images
        self.thumbnail_size = thumbnail_size

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        (self.image_dir / "full").mkdir(parents=True, exist_ok=True)
        (self.image_dir / "thumbnails").mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    part_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    frame_index INTEGER,
                    verdict TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    severity_score REAL NOT NULL,
                    defect_count INTEGER NOT NULL,
                    defects_json TEXT NOT NULL,
                    breakdown_json TEXT NOT NULL,
                    inference_ms REAL NOT NULL,
                    backend TEXT NOT NULL,
                    image_path TEXT,
                    thumbnail_path TEXT,
                    advice_json TEXT,
                    fallback_used INTEGER DEFAULT 0
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS defect_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection_id INTEGER,
                    part_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    area_pct REAL NOT NULL,
                    is_edge INTEGER NOT NULL,
                    bbox_json TEXT NOT NULL,
                    FOREIGN KEY(inspection_id) REFERENCES inspections(id) ON DELETE CASCADE
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_inspections_timestamp ON inspections(timestamp);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_inspections_verdict ON inspections(verdict);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_defect_events_class ON defect_events(class_name);
            """)
            conn.commit()

    def save_inspection(
        self,
        result: InspectionResult,
        frame: Optional[np.ndarray] = None,
        advice: Optional[Dict[str, Any]] = None,
        fallback_used: bool = False,
    ) -> int:
        """
        Record inspection, optionally saving annotated images/thumbnails for REVIEW and REJECT parts.
        """
        image_rel_path: Optional[str] = None
        thumb_rel_path: Optional[str] = None

        should_save_image = self.save_all_images or (result.verdict in ("REVIEW", "REJECT"))
        if should_save_image and frame is not None:
            safe_part = result.part_id.replace(":", "_").replace("/", "_")
            fname = f"{int(result.timestamp)}_{safe_part}_{result.verdict}.jpg"

            full_path = self.image_dir / "full" / fname
            thumb_path = self.image_dir / "thumbnails" / fname

            # Write full annotated or raw frame
            img_to_save = result.annotated_image if result.annotated_image is not None else frame
            cv2.imwrite(str(full_path), img_to_save)
            image_rel_path = f"full/{fname}"

            # Write thumbnail
            thumb = cv2.resize(img_to_save, self.thumbnail_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(thumb_path), thumb)
            thumb_rel_path = f"thumbnails/{fname}"

        defects_json = json.dumps([d.to_dict() for d in result.detections])
        breakdown_json = json.dumps(result.score_breakdown)
        advice_json = json.dumps(advice) if advice else None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO inspections (
                    part_id, timestamp, frame_index, verdict, severity, severity_score,
                    defect_count, defects_json, breakdown_json, inference_ms, backend,
                    image_path, thumbnail_path, advice_json, fallback_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.part_id,
                    result.timestamp,
                    result.frame_index,
                    result.verdict,
                    result.severity,
                    result.severity_score,
                    len(result.detections),
                    defects_json,
                    breakdown_json,
                    result.inference_ms,
                    result.backend,
                    image_rel_path,
                    thumb_rel_path,
                    advice_json,
                    1 if fallback_used else 0,
                ),
            )
            inspection_id = cursor.lastrowid or 0

            # Insert defect details
            for det in result.detections:
                cursor.execute(
                    """
                    INSERT INTO defect_events (
                        inspection_id, part_id, timestamp, class_name, confidence, area_pct, is_edge, bbox_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        inspection_id,
                        result.part_id,
                        result.timestamp,
                        det.class_name,
                        det.confidence,
                        det.area_pct,
                        1 if det.is_edge else 0,
                        json.dumps(det.bbox),
                    ),
                )
            conn.commit()

        return inspection_id

    def update_advice(self, inspection_id: int, advice: Dict[str, Any], fallback_used: bool = False) -> None:
        """Update advice asynchronously when Llama worker produces recommendation."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE inspections
                SET advice_json = ?, fallback_used = ?
                WHERE id = ?
                """,
                (json.dumps(advice), 1 if fallback_used else 0, inspection_id),
            )
            conn.commit()

    def get_recent_inspections(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent inspections for the UI feed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM inspections
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                row_dict = dict(r)
                row_dict["defects"] = json.loads(row_dict["defects_json"])
                row_dict["breakdown"] = json.loads(row_dict["breakdown_json"])
                row_dict["advice"] = json.loads(row_dict["advice_json"]) if row_dict["advice_json"] else None
                results.append(row_dict)
            return results

    def get_trend_context(self, n_recent: int = 10) -> Dict[str, Any]:
        """
        Build a concise trend context for the Llama advisor
        (e.g., number of recent defects, defect types seen over the last N parts).
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT verdict, severity, defects_json FROM inspections
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (n_recent,),
            )
            rows = cursor.fetchall()

        total = len(rows)
        rejects = sum(1 for r in rows if r["verdict"] == "REJECT")
        reviews = sum(1 for r in rows if r["verdict"] == "REVIEW")
        passes = sum(1 for r in rows if r["verdict"] == "PASS")

        recent_classes: Dict[str, int] = {}
        for r in rows:
            try:
                defects = json.loads(r["defects_json"])
                for d in defects:
                    c = d.get("class_name", "unknown")
                    recent_classes[c] = recent_classes.get(c, 0) + 1
            except Exception:
                pass

        return {
            "window_size": total,
            "recent_passes": passes,
            "recent_reviews": reviews,
            "recent_rejects": rejects,
            "reject_rate_pct": round((rejects / total * 100.0), 1) if total > 0 else 0.0,
            "top_recent_defect_classes": recent_classes,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics for dashboard charts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), verdict FROM inspections GROUP BY verdict")
            verdicts = {row[1]: row[0] for row in cursor.fetchall()}

            cursor.execute("SELECT COUNT(*), severity FROM inspections GROUP BY severity")
            severities = {row[1]: row[0] for row in cursor.fetchall()}

            cursor.execute("SELECT COUNT(*), class_name FROM defect_events GROUP BY class_name")
            defect_classes = {row[1]: row[0] for row in cursor.fetchall()}

            cursor.execute("SELECT COUNT(*), AVG(inference_ms) FROM inspections")
            row = cursor.fetchone()
            total_inspections = row[0] if row else 0
            avg_latency = round(row[1] or 0.0, 2) if row else 0.0

        total_rejects = verdicts.get("REJECT", 0)
        reject_rate = (total_rejects / total_inspections * 100.0) if total_inspections > 0 else 0.0

        return {
            "total_inspections": total_inspections,
            "verdicts": verdicts,
            "severities": severities,
            "defect_classes": defect_classes,
            "avg_latency_ms": avg_latency,
            "reject_rate_pct": round(reject_rate, 2),
        }

    def close(self) -> None:
        """Explicitly release any open sqlite resources."""
        pass

    def export_csv(self, out_path: str | Path) -> str:
        """Export inspection records to CSV file."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn, open(out, "w", newline="", encoding="utf-8") as f:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, part_id, datetime(timestamp, 'unixepoch', 'localtime') as inspection_time,
                       verdict, severity, severity_score, defect_count, inference_ms, backend,
                       fallback_used, advice_json
                FROM inspections
                ORDER BY id ASC
            """)
            writer = csv.writer(f)
            writer.writerow([
                "ID", "Part ID", "Inspection Time", "Verdict", "Severity",
                "Severity Score", "Defect Count", "Inference (ms)", "Backend",
                "Fallback Used", "Advice"
            ])
            for row in cursor.fetchall():
                writer.writerow(list(row))
        return str(out)
