import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlfluff
from pipeline.sql_generator import generate_sql

import re

def sanitize_sql(sql: str) -> str:
    sql = sql.replace("≥", ">=").replace("≤", "<=")
    return sql

def validate_and_fix_sql(sql: str, user_query: str, retrieved_chunks: dict, max_retries: int = 2) -> tuple[bool, str]:
    if sql in ("UNSAFE_QUERY_DETECTED", "INSUFFICIENT_CONTEXT"):
        return (False, sql)
        
    keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE"]
    if any(keyword in sql.upper() for keyword in keywords):
        return (False, "UNSAFE_QUERY_DETECTED")
        
    current_sql = sanitize_sql(sql)
    
    # Load environment variables for DB connection
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(env_path)
    conn_str = os.getenv("DB_CONNECTION_STRING")
    
    from pipeline.db_connector import get_connection, close_connection
    
    for attempt in range(max_retries + 1):
        try:
            # 1. Syntax check using SQLFluff
            parsed = sqlfluff.parse(current_sql, dialect="tsql")
            
            # 2. Database Dry-Run (Schema & Column validation)
            if conn_str and conn_str != "your_sql_server_connection_string":
                connection = get_connection(conn_str)
                try:
                    cursor = connection.cursor()
                    # SET NOEXEC ON compiles the query to check objects, but doesn't run it
                    cursor.execute("SET NOEXEC ON")
                    cursor.execute(current_sql)
                except Exception as db_err:
                    raise Exception(f"Database schema error: {db_err}")
                finally:
                    # Always ensure NOEXEC is turned off before returning the connection to the pool
                    cursor.execute("SET NOEXEC OFF")
                    close_connection(connection)
                    
            return (True, current_sql)
        except Exception as e:
            if attempt < max_retries:
                error_message = str(e)
                retry_prompt = f"{user_query}\nThe following SQL has an error: {error_message}\nFix it and return only the corrected SQL."
                current_sql = generate_sql(retry_prompt, retrieved_chunks)
                current_sql = sanitize_sql(current_sql)
            else:
                pass
                
    return (False, "UNABLE_TO_GENERATE: Sorry, unable to get the required results.")
