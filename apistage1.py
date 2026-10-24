import logging
import os

import psycopg
from fastapi import FastAPI, HTTPException

app = FastAPI(title="NOVA API — Stage 1")
logger = logging.getLogger(__name__)

# Same local database connection approach as simulator
DATABASE_URL = os.getenv("DATABASE_URL", "dbname=nova")

@app.get("/health")
def health():
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_database();")
                database_name = cur.fetchone()[0]

                cur.execute("""
                    SELECT id, name
                    FROM missions
                    WHERE active = TRUE;
                """)
                mission = cur.fetchone()
        return {
            "status": "ok",
            "database": database_name,
            "database_connected": True,
            "active_mission_id": mission[0] if mission else None,
            "active_mission_name": mission[1] if mission else None,
        }
    except psycopg.Error as e:
        # Full details go to the Terminal; the browser gets a short message.
        logger.error("Database connection failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to the nova database. Is PostgreSQL running?",
        )