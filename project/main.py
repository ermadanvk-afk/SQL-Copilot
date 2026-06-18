import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

# Now import the pipeline modules
from pipeline.rag_retriever import retrieve_relevant_chunks
from pipeline.sql_generator import generate_sql
from pipeline.validator import validate_and_fix_sql
from pipeline.db_executor import execute_query

import streamlit as st
import pandas as pd

def apply_custom_css():
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #6B7280;
            margin-bottom: 1.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Data Insights Portal", 
        page_icon="📊", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    apply_custom_css()
    
    st.markdown('<div class="main-header">📊 Data Insights Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions in plain English and get instant data insights powered by AI.</div>', unsafe_allow_html=True)
    st.divider()
    
     db_connection_string = os.getenv("DB_CONNECTION_STRING")
    if not db_connection_string or db_connection_string == "your_sql_server_connection_string":
        st.error("⚠️ Database connection is not configured. Please check your settings.")
        
    st.markdown("**What would you like to know?**")
    col1, col2 = st.columns([6, 1])
    with col1:
        user_query = st.text_input("query", placeholder="e.g., Show me the top 5 vendors by revenue this month...", label_visibility="collapsed")
    with col2:
        submit_btn = st.button("Generate Insights", type="primary", use_container_width=True)
    
    if submit_btn:
        if not user_query:
            st.warning("Please enter a question to generate insights.")
            return
            
        with st.status("Processing your request...", expanded=True) as status:
            st.write("🔍 Analyzing request and retrieving context...")
            try:
                chunks = retrieve_relevant_chunks(user_query, top_k_tables=5, top_k_rules=3, top_k_queries=3)
            except Exception as e:
                status.update(label="Failed to retrieve context.", state="error", expanded=True)
                st.error(f"Error: {e}")
                return
                
            st.write("🧠 Translating to SQL using AI...")
            try:
                raw_sql = generate_sql(user_query, chunks)
            except Exception as e:
                status.update(label="Failed to generate SQL.", state="error", expanded=True)
                st.error(f"Error: {e}")
                return
                
            st.write("🛠️ Validating and optimizing query...")
            is_valid, final_sql = validate_and_fix_sql(raw_sql, user_query, chunks, max_retries=2)
            
            if not is_valid:
                status.update(label="Validation failed.", state="error", expanded=True)
                st.error("❌ Failed to generate a valid/safe SQL query.")
                with st.expander("View Invalid SQL"):
                    st.code(final_sql, language="sql")
                return
                
            status.update(label="Query ready!", state="complete", expanded=False)
            
        with st.expander("View Generated SQL Query", expanded=False):
            st.code(final_sql, language="sql")
            
        # Log the interaction for debugging
        import datetime
        log_path = os.path.join(os.path.dirname(__file__), "query_logs.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"--- {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write(f"query asked:\n{user_query}\n\n")
            f.write(f"context used (retrieved from rag):\n{chunks}\n\n")
            f.write(f"final sql generated:\n{final_sql}\n")
            f.write("=" * 80 + "\n\n")
        
        if db_connection_string and db_connection_string != "your_sql_server_connection_string":
            st.markdown("### Results")
            with st.spinner("⚡ Fetching data..."):
                try:
                    columns, rows = execute_query(final_sql, db_connection_string)
                    if rows:
                        df = pd.DataFrame(rows)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.info("The query executed successfully but returned no results.")
                except Exception as e:
                    st.error(f"Error executing query:\n{e}")

if __name__ == "__main__":
    main()
