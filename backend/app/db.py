import aiosqlite
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path("backend/tiger_detection.db")


async def init_db():
    """Initializes the SQLite database schema if not present."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                image_url TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                detected INTEGER NOT NULL,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                detection_confidence REAL,
                predicted_id TEXT NOT NULL,
                tiger_name TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                is_known INTEGER NOT NULL,
                reference_image TEXT,
                processing_time_ms REAL NOT NULL
            )
            """
        )
        await db.commit()


async def insert_detection(record: dict) -> int:
    """Inserts a detection event into the database and returns the record ID."""
    await init_db()
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """
            INSERT INTO detections (
                timestamp, filename, image_url, width, height,
                detected, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                detection_confidence, predicted_id, tiger_name,
                similarity_score, is_known, reference_image, processing_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["timestamp"],
                record["filename"],
                record["image_url"],
                record["width"],
                record["height"],
                1 if record["detected"] else 0,
                record.get("bbox_x1"),
                record.get("bbox_y1"),
                record.get("bbox_x2"),
                record.get("bbox_y2"),
                record.get("detection_confidence", 0.0),
                record["predicted_id"],
                record["tiger_name"],
                record["similarity_score"],
                1 if record["is_known"] else 0,
                record.get("reference_image"),
                record["processing_time_ms"],
            ),
        )
        await db.commit()
        return cursor.lastrowid


async def get_history(limit: int = 50, offset: int = 0) -> List[Dict]:
    """Retrieves chronological detection history."""
    await init_db()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT * FROM detections
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def get_detection_by_id(record_id: int) -> Optional[Dict]:
    """Retrieves a single detection record by primary key."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM detections WHERE id = ?", (record_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None
