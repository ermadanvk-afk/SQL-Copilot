import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.db_connector import get_connection, close_connection

import re

def execute_query(sql: str, connection_string: str, timeout_seconds: int = 5) -> tuple[list[str], list[dict]]:
    connection = None
    try:
        connection = get_connection(connection_string)
        connection.timeout = timeout_seconds
        cursor = connection.cursor()
        
        cursor.execute(sql)
        column_names = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
        
        rows_as_list_of_dicts = []
        for row in rows:
            rows_as_list_of_dicts.append(dict(zip(column_names, row)))
            
        return (column_names, rows_as_list_of_dicts)
    except Exception as e:
        raise RuntimeError(f"Database execution error: {e}\n\nFAILED SQL:\n{sql}")
    finally:
        if connection is not None:
            close_connection(connection)

if __name__ == "__main__":
    from dotenv import load_dotenv
    
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(env_path)
    
    test_sql = "SELECT DATENAME(MONTH, DocumentDate) AS MonthName, COUNT(DISTINCT PONo) AS POCount FROM [Purchase].[vwaiPurchaseOrderMain] WHERE YEAR(DocumentDate) = YEAR(GETDATE()) GROUP BY MONTH(DocumentDate), DATENAME(MONTH, DocumentDate) ORDER BY MONTH(DocumentDate)"  # Example SQL
    conn_str = os.getenv("DB_CONNECTION_STRING")
    
    if conn_str and conn_str != "your_sql_server_connection_string":
        print(f"Testing execution with SQL:\n{test_sql}\n")
        try:
            cols, rows = execute_query(test_sql, conn_str)
            print(f"Row count: {len(rows)}")
            if rows:
                print("\nData Result:")
                import pandas as pd
                df = pd.DataFrame(rows)
                print(df.to_string(index=False))
            else:
                print("No rows returned.")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("DB_CONNECTION_STRING is not set properly in .env")
