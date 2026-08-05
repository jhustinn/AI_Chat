"""
DB Engine - Execute SQL queries secara safe (READ ONLY)
Untuk text-to-SQL functionality
"""

import psycopg2
import json
import re
from datetime import datetime, date, time
from decimal import Decimal
from typing import Optional

DB_CONFIG = {
    "host": "103.83.98.156",
    "port": "9497",
    "dbname": "dev_richz",
    "user": "svc_pg3xp",
    "password": "PgP4ss_9494!",
    "options": "-c search_path=global"
}

# Forbidden keywords (write operations)
FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "GRANT", "REVOKE", "EXECUTE", "EXEC",
    "CALL", "DO", "SET", "RESET", "SHOW", "BEGIN", "COMMIT",
    "ROLLBACK", "SAVEPOINT", "LOCK", "UNLOCK", "COMMENT"
]


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


def validate_query(sql: str) -> tuple[bool, str]:
    """Validate bahwa query hanya SELECT"""
    sql_upper = sql.upper().strip()
    
    # Check harus mulai dengan SELECT
    if not sql_upper.startswith("SELECT"):
        return False, "Query harus dimulai dengan SELECT"
    
    # Check forbidden keywords
    for keyword in FORBIDDEN_KEYWORDS:
        # Word boundary check
        pattern = r'\b' + keyword + r'\b'
        if re.search(pattern, sql_upper):
            return False, f"Keyword '{keyword}' tidak diizinkan"
    
    return True, "OK"


def execute_query(sql: str, max_rows: int = 100) -> dict:
    """Execute SELECT query dan return hasil"""
    # Validate
    is_valid, message = validate_query(sql)
    if not is_valid:
        return {"error": message, "success": False}
    
    # Add LIMIT jika tidak ada
    sql_upper = sql.strip().upper()
    if "LIMIT" not in sql_upper:
        sql = sql.rstrip().rstrip(";") + f" LIMIT {max_rows}"
    
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Execute dengan timeout
        cur.execute("SET statement_timeout = '10000'")  # 10 detik
        cur.execute(sql)
        
        # Get results
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        
        # Convert to dict
        result = []
        for row in rows:
            result.append(dict(zip(columns, row)))
        
        cur.close()
        
        return {
            "success": True,
            "columns": columns,
            "rows": result,
            "row_count": len(result),
            "sql": sql
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "success": False,
            "sql": sql
        }
    finally:
        if conn:
            conn.close()


def get_schema_context(tables: list[str] = None) -> str:
    """Get schema context untuk RAG"""
    schema_path = r"E:\App\SWA\AI\data\db_knowledge\schema.json"
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    
    context = []
    for table_name, info in schema.items():
        if tables and table_name not in tables:
            continue
        if info["row_count"] == 0:
            continue
        
        context.append(f"\nTable: {table_name} ({info['row_count']:,} rows)")
        context.append("Columns:")
        for col in info["columns"]:
            nullable = "NULL" if col["nullable"] else "NOT NULL"
            context.append(f"  - {col['name']}: {col['type']} ({nullable})")
    
    return "\n".join(context)


if __name__ == "__main__":
    # Test
    print("Testing DB Engine...")
    
    # Test valid query
    result = execute_query("SELECT COUNT(*) as total FROM global_member")
    print(f"Count members: {result}")
    
    # Test invalid query
    result = execute_query("DELETE FROM global_member")
    print(f"Delete attempt: {result}")
