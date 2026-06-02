from chromadb.utils import embedding_functions

# Initialize the default local embedding function (uses all-MiniLM-L6-v2)
# This runs entirely locally, requires no API key, and incurs no costs.
local_embedder = embedding_functions.DefaultEmbeddingFunction()

def create_embeddings(texts: list[str]) -> list[list[float]]:
    try:
        # Process multiple texts efficiently in a single batch
        results = local_embedder(texts)
        return [res.tolist() for res in results]
    except Exception as e:
        raise RuntimeError(f"Failed to create batch embeddings: {e}")

if __name__ == "__main__":
    statements = [
        "This is a simple statement to test the embedder.",
        "Here is a second statement for batch testing."
    ]
    print(f"Statements: {statements}")
    
    embeddings = create_embeddings(statements)
    print(f"Number of embeddings generated: {len(embeddings)}")
    print(f"Dimensions of first embedding: {len(embeddings[0])}")
    print(f"First 5 dimensions of first embedding: {embeddings[0][:5]}")