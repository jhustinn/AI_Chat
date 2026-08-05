"""
DB Exporter - Export PostgreSQL schema + sample data ke JSON
Untuk RAG knowledge base
"""

import psycopg2
import json
import os
from datetime import datetime, date, time
from decimal import Decimal

OUTPUT_DIR = r"E:\App\SWA\AI\data\db_knowledge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DB_CONFIG = {
    "host": "103.83.98.156",
    "port": "9497",
    "dbname": "dev_richz",
    "user": "svc_pg3xp",
    "password": "PgP4ss_9494!",
    "options": "-c search_path=global"
}


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, time):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        return super().default(obj)


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def export_schema(cur):
    """Export semua table schema"""
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'global' 
        ORDER BY table_name
    """)
    tables = [t[0] for t in cur.fetchall()]
    
    schema = {}
    for table in tables:
        cur.execute("""
            SELECT column_name, data_type, is_nullable, 
                   column_default
            FROM information_schema.columns 
            WHERE table_schema = 'global' AND table_name = %s
            ORDER BY ordinal_position
        """, (table,))
        
        columns = []
        for col in cur.fetchall():
            columns.append({
                "name": col[0],
                "type": col[1],
                "nullable": col[2] == "YES",
                "default": col[3]
            })
        
        # Get row count
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        row_count = cur.fetchone()[0]
        
        schema[table] = {
            "columns": columns,
            "row_count": row_count
        }
    
    return schema


def export_sample_data(cur, table, limit=5):
    """Export sample data dari table"""
    try:
        cur.execute(f"SELECT * FROM {table} LIMIT {limit}")
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        
        data = []
        for row in rows:
            data.append(dict(zip(columns, row)))
        
        return data
    except Exception as e:
        return []


def export_all():
    """Export semua schema + sample data"""
    conn = get_connection()
    cur = conn.cursor()
    
    print("Exporting schema...")
    schema = export_schema(cur)
    
    # Save schema
    with open(os.path.join(OUTPUT_DIR, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, cls=DateTimeEncoder, ensure_ascii=False, indent=2)
    
    print(f"Exported {len(schema)} tables")
    
    # Export sample data untuk tables yang punya data
    print("Exporting sample data...")
    sample_data = {}
    for table_name, info in schema.items():
        if info["row_count"] > 0:
            data = export_sample_data(cur, table_name, limit=5)
            if data:
                sample_data[table_name] = {
                    "row_count": info["row_count"],
                    "sample": data
                }
    
    with open(os.path.join(OUTPUT_DIR, "sample_data.json"), "w", encoding="utf-8") as f:
        json.dump(sample_data, f, cls=DateTimeEncoder, ensure_ascii=False, indent=2)
    
    print(f"Exported sample data for {len(sample_data)} tables")
    
    # Create knowledge base document untuk RAG
    print("Creating knowledge base document...")
    knowledge = create_knowledge_document(schema, sample_data)
    
    with open(os.path.join(OUTPUT_DIR, "knowledge.txt"), "w", encoding="utf-8") as f:
        f.write(knowledge)
    
    print(f"Knowledge document: {len(knowledge)} chars")
    
    cur.close()
    conn.close()
    
    return schema, sample_data


def create_knowledge_document(schema, sample_data):
    """Buat knowledge base document untuk RAG"""
    doc = []
    doc.append("=== DATABASE SCHEMA: dev_richz (PostgreSQL) ===")
    doc.append(f"Schema: global")
    doc.append(f"Total Tables: {len(schema)}")
    doc.append("")
    
    for table_name, info in sorted(schema.items()):
        if info["row_count"] == 0:
            continue
        
        doc.append(f"\n--- Table: {table_name} ({info['row_count']:,} rows) ---")
        doc.append("Columns:")
        
        for col in info["columns"]:
            nullable = "NULL" if col["nullable"] else "NOT NULL"
            doc.append(f"  - {col['name']}: {col['type']} ({nullable})")
        
        # Add sample data
        if table_name in sample_data:
            doc.append("Sample data:")
            for row in sample_data[table_name]["sample"][:3]:
                doc.append(f"  {row}")
    
    return "\n".join(doc)


if __name__ == "__main__":
    schema, sample_data = export_all()
    print("\nDone! Files saved to:", OUTPUT_DIR)
