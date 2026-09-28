import json
import os
import re


def save_metadata(filename: str, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_metadata(filename: str):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None


def validate_sql_safety(sql_query: str) -> tuple[bool, str]:
    """Validates SQL query for dangerous operations. Returns (is_safe, error_message)."""
    DANGEROUS_KEYWORDS = [
        'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE',
        'GRANT', 'REVOKE', 'EXEC', 'EXECUTE', 'CALL', 'SET', 'LOCK', 'UNLOCK'
    ]
    sql_upper = sql_query.upper()

    for keyword in DANGEROUS_KEYWORDS:
        pattern = r'\b' + keyword + r'\b'
        if re.search(pattern, sql_upper):
            return False, f"Dangerous SQL keyword '{keyword}' detected. Only SELECT queries are allowed."

    sql_stripped = sql_query.strip().upper()
    if not (sql_stripped.startswith('SELECT') or sql_stripped.startswith('WITH')):
        return False, "Only SELECT queries are allowed."

    # Reject unbound positional parameters ($1, $2, ...) – asyncpg requires
    # them to be passed as arguments but the execution layer calls fetch(sql)
    # without any bound values.
    if re.search(r'\$\d+', sql_query):
        return False, "SQL must not contain parameter placeholders ($1, $2, …) — use literal values only."

    return True, ""


async def discover_schema_from_db(pool, schema_file: str, relationships_file: str):
    """Discover schema from PostgreSQL and cache to files. Returns (schema, relationships)."""
    target_schemas = ["finance"]

    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema = ANY($1)
            AND table_type = 'BASE TABLE'
            ORDER BY table_schema, table_name;
        """, target_schemas)

        tables = [f"{row['table_schema']}.{row['table_name']}" for row in rows]

        schema_metadata = {}
        relationships = []

        for table_key in tables:
            schema, table = table_key.split('.')

            cols = await conn.fetch("""
                SELECT column_name, data_type, is_nullable, column_default, character_maximum_length
                FROM information_schema.columns
                WHERE table_schema = $1 AND table_name = $2
                ORDER BY ordinal_position;
            """, schema, table)

            columns = {}
            for col in cols:
                columns[col['column_name']] = {
                    "type": col['data_type'],
                    "nullable": col['is_nullable'] == "YES",
                    "default": col['column_default'],
                    "max_length": col['character_maximum_length']
                }

            pks = await conn.fetch("""
                SELECT a.attname
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                JOIN pg_class c ON i.indrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = $1 AND c.relname = $2 AND i.indisprimary;
            """, schema, table)
            primary_keys = [row['attname'] for row in pks]

            fks = await conn.fetch("""
                SELECT
                    kcu.column_name,
                    ccu.table_schema AS foreign_table_schema,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = $1
                AND tc.table_name = $2;
            """, schema, table)

            foreign_keys = []
            for fk in fks:
                foreign_table_key = f"{fk['foreign_table_schema']}.{fk['foreign_table_name']}"
                fk_data = {
                    "column": fk['column_name'],
                    "references_table": foreign_table_key,
                    "references_column": fk['foreign_column_name']
                }
                foreign_keys.append(fk_data)
                relationships.append({
                    "from_table": table_key,
                    "to_table": foreign_table_key,
                    "from_column": fk['column_name'],
                    "to_column": fk['foreign_column_name'],
                    "relationship_type": "many-to-one"
                })

            schema_metadata[table_key] = {
                "columns": columns,
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys
            }

    manual_relationships = [
        {
            "from_table": "finance.sales_invoice",
            "to_table": "finance.customer_payment",
            "from_column": "document_number",
            "to_column": "applying_transaction",
            "relationship_type": "logical"
        }
    ]
    relationships.extend(manual_relationships)

    save_metadata(schema_file, schema_metadata)
    save_metadata(relationships_file, relationships)

    return schema_metadata, relationships
