import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chunking_and_vectorization.embedder import create_embeddings
from chunking_and_vectorization.schema_chunker import (
    build_table_chunks, 
    build_business_rule_chunks, 
    build_sample_query_chunks
)
from chunking_and_vectorization.getting_df import get_df

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

def initialize_vector_store() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="schema_chunks")
    return collection

def store_chunks_to_vector_db() -> None:
    # Completely remove the previous vector db folder to start fresh and avoid orphaned files
    import shutil
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            print("Removed previous vector database folder.")
        except Exception as e:
            print(f"Warning: Could not remove old chroma_db folder: {e}")
    
    # Initialize the client (this will create a fresh chroma_db folder)
    client = chromadb.PersistentClient(path=DB_PATH)

    print("Fetching data from sheets...")
    schema_df = get_df("Sheet2")
    rules_df = get_df("Sheet1")
    queries_df = get_df("Sheet4")
    
    print("Building chunks...")
    schema_chunks = build_table_chunks(schema_df)
    rule_chunks = build_business_rule_chunks(rules_df)
    query_chunks = build_sample_query_chunks(queries_df)
    
    ALL_CHUNKS = schema_chunks + rule_chunks + query_chunks
    
    if not ALL_CHUNKS:
        print("No chunks to ingest.")
        return
        
    collection = initialize_vector_store()
    
    # Prepare batch data
    texts = [chunk["content"] for chunk in ALL_CHUNKS]
    ids = [chunk["chunk_id"] for chunk in ALL_CHUNKS]
    
    metadatas = []
    for chunk in ALL_CHUNKS:
        meta = {
            "chunk_type": chunk["chunk_type"], 
            "chunk_id": chunk["chunk_id"]
        }
        if "sql" in chunk:
            meta["sql"] = chunk["sql"]
        metadatas.append(meta)
    
    print(f"Generating embeddings for {len(ALL_CHUNKS)} chunks (Schema: {len(schema_chunks)}, Rules: {len(rule_chunks)}, Queries: {len(query_chunks)})...")
    
    # Create embeddings in a single efficient batch
    embeddings = create_embeddings(texts)
    
    # Upsert into ChromaDB
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    
    print(f"Done. Successfully stored {len(ALL_CHUNKS)} chunks in the vector database.")
    
    print("Building and saving BM25 index...")
    import pickle
    from rank_bm25 import BM25Okapi
    # simple whitespace tokenizer
    tokenized_corpus = [doc.lower().split(" ") for doc in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_data = {
        "bm25": bm25,
        "texts": texts,
        "metadatas": metadatas,
        "ids": ids
    }
    with open(os.path.join(PROJECT_ROOT, "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25_data, f)
    print("BM25 Index saved successfully to bm25_index.pkl.")

if __name__ == "__main__":
    print("Starting vector store pipeline...")
    store_chunks_to_vector_db()
