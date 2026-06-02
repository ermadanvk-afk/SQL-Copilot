import sys
import os
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking_and_vectorization.embedder import create_embeddings
from chunking_and_vectorization.vector_store import initialize_vector_store

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BM25_PATH = os.path.join(PROJECT_ROOT, "bm25_index.pkl")
bm25_data = None
if os.path.exists(BM25_PATH):
    try:
        with open(BM25_PATH, "rb") as f:
            bm25_data = pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load BM25 index: {e}")

# Build in-memory table lookup for fast schema traversal
table_lookup = {}
if bm25_data:
    for idx, meta in enumerate(bm25_data["metadatas"]):
        if meta.get("chunk_type") == "table":
            text = bm25_data["texts"][idx]
            for line in text.split("\n"):
                if line.startswith("Table Name:"):
                    t_name = line.replace("Table Name:", "").strip()
                    table_lookup[t_name] = text
                    break

def make_lite_schema(full_chunk_text: str) -> str:
    lite_lines = []
    in_columns = False
    for line in full_chunk_text.split("\n"):
        if line.startswith("Table Name:"):
            lite_lines.append(line)
            lite_lines.append("Description: [LITE SCHEMA - RELATIONAL COLUMNS ONLY]")
        elif line.startswith("Columns:"):
            in_columns = True
            lite_lines.append(line)
        elif in_columns and line.startswith("  -"):
            lite_lines.append(line)
        elif in_columns and not line.startswith("  -") and line.strip() != "":
            in_columns = False
    return "\n".join(lite_lines)

def get_bm25_results(query: str, chunk_type: str, top_k: int) -> list[str]:
    if not bm25_data:
        return []
    
    query_tokens = query.lower().split(" ")
    bm25 = bm25_data["bm25"]
    
    # Get scores for all documents
    scores = bm25.get_scores(query_tokens)
    
    # Pair scores with indices (only positive scores)
    scored_indices = [(i, score) for i, score in enumerate(scores) if score > 0]
    scored_indices.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for idx, score in scored_indices:
        meta = bm25_data["metadatas"][idx]
        if meta.get("chunk_type") == chunk_type:
            doc = bm25_data["texts"][idx].strip()
            if chunk_type == "sample_query" and "sql" in meta:
                doc += f"\nSQL: {meta['sql']}"
            results.append(doc)
            if len(results) >= top_k:
                break
    return results

def retrieve_relevant_chunks(user_query: str, top_k_tables: int = 5, top_k_rules: int = 8, top_k_queries: int = 3) -> dict:
    # Use the updated batch embedder for a single query
    embedding = create_embeddings([user_query])[0]
    collection = initialize_vector_store()
    
    retrieved_docs = {"tables": [], "rules": [], "queries": []}
    
    # 1. Retrieve Tables
    table_results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k_tables,
        where={"chunk_type": "table"}
    )
    chroma_tables = [doc.strip() for doc in table_results["documents"][0] if doc.strip()] if table_results and "documents" in table_results and table_results["documents"] else []
    bm25_tables = get_bm25_results(user_query, "table", top_k_tables)
    
    # Deduplicate and merge (BM25 matches prioritized)
    core_tables = list(dict.fromkeys(bm25_tables + chroma_tables))[:top_k_tables]
    
    # --- Schema Traversal (Foreign Key Expansion) ---
    core_table_names = set()
    for chunk in core_tables:
        for line in chunk.split("\n"):
            if line.startswith("Table Name:"):
                core_table_names.add(line.replace("Table Name:", "").strip())
                break
                
    relational_tables = set()
    for chunk in core_tables:
        in_fk = False
        for line in chunk.split("\n"):
            line = line.strip()
            if line.startswith("Foreign Keys:"):
                in_fk = True
                continue
            if in_fk:
                if line == "" or not line.startswith("-"):
                    in_fk = False
                    continue
                if "->" in line:
                    # Parse: '- PONo -> Purchase.POMain.PONo'
                    target = line.split("->")[1].strip()
                    target_table = ".".join(target.split(".")[:2])
                    if target_table not in core_table_names:
                        relational_tables.add(target_table)
                        
    extended_tables = list(core_tables)
    for t_name in relational_tables:
        if t_name in table_lookup:
            extended_tables.append(make_lite_schema(table_lookup[t_name]))
            
    retrieved_docs["tables"] = extended_tables
        
    # 2. Retrieve Business Rules
    rule_results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k_rules,
        where={"chunk_type": "business_rule"}
    )
    chroma_rules = [doc.strip() for doc in rule_results["documents"][0] if doc.strip()] if rule_results and "documents" in rule_results and rule_results["documents"] else []
    bm25_rules = get_bm25_results(user_query, "business_rule", top_k_rules)
    
    retrieved_docs["rules"] = list(dict.fromkeys(bm25_rules + chroma_rules))[:top_k_rules]
        
    # 3. Retrieve Sample Queries (and include SQL from metadata)
    query_results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k_queries,
        where={"chunk_type": "sample_query"}
    )
    chroma_queries = []
    if query_results and "documents" in query_results and query_results["documents"]:
        docs = query_results["documents"][0]
        metas = query_results["metadatas"][0] if "metadatas" in query_results and query_results["metadatas"] else [{}] * len(docs)
        for doc, meta in zip(docs, metas):
            formatted_query = f"{doc.strip()}"
            if meta and "sql" in meta:
                formatted_query += f"\nSQL: {meta['sql']}"
            chroma_queries.append(formatted_query)
            
    bm25_queries = get_bm25_results(user_query, "sample_query", top_k_queries)
    retrieved_docs["queries"] = list(dict.fromkeys(bm25_queries + chroma_queries))[:top_k_queries]
            
    return retrieved_docs

if __name__ == "__main__":
    print("Initializing test for RAG Retriever...\n")
    test_query = "which vendors are becoming expensive ?"
    print(f"User Query: '{test_query}'\n")
    
    # Retrieve relevant chunks using partitioned retrieval
    chunks = retrieve_relevant_chunks(test_query)
    
    print(f"Retrieved Chunks:\n" + "="*50)
    for k, v in chunks.items():
        print(f"\n--- {k.upper()} ---")
        for c in v:
            print(c)
            print()

# indent main done
# indent item detail done
# po main done, po itemm detail done, vendor location detail done
# quotation main done, quotation item detail done, rfq main done, rfq item detail done. 
# po tax detail done, po item tax detail done. 
# item master done. 
# po amendment main done
# po amendment item tax detail done
