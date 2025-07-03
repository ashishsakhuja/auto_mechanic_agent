#!/usr/bin/env python3
import csv
import sys
from pathlib import Path
import duckdb

# Adjust these paths to match your project layout:
ROOT     = Path(__file__).resolve().parent
CSV_PATH = ROOT / "charm_manifest.csv"
DB_PATH  = ROOT / "knowledge" / "manuals.duckdb"

def setup_database(csv_path: Path, db_path: Path):
    # ensure output folder exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # check CSV exists
    if not csv_path.exists():
        print(f"Error: cannot find {csv_path}", file=sys.stderr)
        sys.exit(1)

    # connect to DuckDB
    conn = duckdb.connect(str(db_path))

    # drop & recreate table
    conn.execute("DROP TABLE IF EXISTS manifest")
    conn.execute("""
        CREATE TABLE manifest (
            make        TEXT,
            model       TEXT,
            year        TEXT,
            bundle_url  TEXT
        );
    """)

    # insert rows
    inserted = 0
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                conn.execute(
                    "INSERT INTO manifest VALUES (?, ?, ?, ?)",
                    [row["make"], row["model"], row["year"], row["bundle_url"]]
                )
                inserted += 1
            except Exception as e:
                print(f"[WARN] could not insert {row}: {e}", file=sys.stderr)

    conn.commit()
    conn.close()
    print(f"Inserted {inserted} rows into {db_path!r}")

if __name__ == "__main__":
    setup_database(CSV_PATH, DB_PATH)
