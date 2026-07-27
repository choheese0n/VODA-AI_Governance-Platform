import json
import os
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path(
    os.getenv(
        "VODA_DB_PATH",
        "voda_monitoring.db",
    )
)


def _database_path(
    db_path: str | Path | None = None,
) -> Path:
    path = (
        Path(db_path)
        if db_path is not None
        else DEFAULT_DB_PATH
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def _connect(
    db_path: str | Path | None = None,
) -> sqlite3.Connection:
    connection = sqlite3.connect(
        _database_path(db_path),
        timeout=10,
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database(
    db_path: str | Path | None = None,
) -> None:
    with _connect(db_path) as connection:
        connection.execute(
            "PRAGMA journal_mode=WAL"
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                user_text TEXT NOT NULL,
                ai_text TEXT NOT NULL,
                grade TEXT NOT NULL,
                sycophancy INTEGER NOT NULL
                    CHECK (sycophancy IN (0, 1)),
                bias_level INTEGER NOT NULL
                    CHECK (bias_level BETWEEN 0 AND 3),
                severity INTEGER NOT NULL
                    CHECK (severity BETWEEN 0 AND 2),
                dominant_category TEXT,
                review_status TEXT NOT NULL
                    DEFAULT 'pending',
                payload_json TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_incidents_created_at
            ON incidents(created_at DESC)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_incidents_grade
            ON incidents(grade)
            """
        )


def save_incident(
    record: dict[str, Any],
    db_path: str | Path | None = None,
) -> None:
    required_fields = {
        "id",
        "created_at",
        "user_text",
        "ai_text",
        "grade",
        "sycophancy",
        "bias_level",
        "severity",
    }

    missing = sorted(
        required_fields.difference(record)
    )

    if missing:
        raise ValueError(
            "저장할 분석 기록에 필수 필드가 없습니다: "
            + ", ".join(missing)
        )

    initialize_database(db_path)

    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
    )

    values = (
        str(record["id"]),
        str(record["created_at"]),
        str(record["user_text"]),
        str(record["ai_text"]),
        str(record["grade"]),
        int(record["sycophancy"]),
        int(record["bias_level"]),
        int(record["severity"]),
        record.get("dominant_category"),
        str(
            record.get(
                "review_status",
                "pending",
            )
        ),
        payload,
    )

    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO incidents (
                id,
                created_at,
                user_text,
                ai_text,
                grade,
                sycophancy,
                bias_level,
                severity,
                dominant_category,
                review_status,
                payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(id) DO UPDATE SET
                created_at = excluded.created_at,
                user_text = excluded.user_text,
                ai_text = excluded.ai_text,
                grade = excluded.grade,
                sycophancy = excluded.sycophancy,
                bias_level = excluded.bias_level,
                severity = excluded.severity,
                dominant_category =
                    excluded.dominant_category,
                review_status =
                    excluded.review_status,
                payload_json =
                    excluded.payload_json
            """,
            values,
        )


def list_incidents(
    limit: int = 200,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    initialize_database(db_path)

    safe_limit = max(
        1,
        min(
            int(limit),
            1000,
        ),
    )

    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                payload_json,
                review_status
            FROM incidents
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    records = []

    for row in reversed(rows):
        record = json.loads(
            row["payload_json"]
        )

        record["review_status"] = (
            row["review_status"]
        )

        records.append(record)

    return records


def get_incident(
    incident_id: str,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    initialize_database(db_path)

    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT
                payload_json,
                review_status
            FROM incidents
            WHERE id = ?
            """,
            (incident_id,),
        ).fetchone()

    if row is None:
        return None

    record = json.loads(
        row["payload_json"]
    )

    record["review_status"] = (
        row["review_status"]
    )

    return record


initialize_database()