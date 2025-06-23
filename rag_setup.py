import os
from openai import OpenAI
import chromadb
from chromadb.config import Settings

# client = chromadb.PersistentClient(path="./db")
# collections = client.list_collections()
# print("Available collections:", [col.name for col in collections])

# Initialize the OpenAI client (you'll need to set your API key)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_openai_embeddings(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def search_similar_chunks(query, collection, top_k=5):
    query_embedding = generate_openai_embeddings(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Ensure the results have the expected structure
    if not results or 'documents' not in results or 'metadatas' not in results:
        return {'documents': [[]], 'metadatas': [[]]}
    
    return results

def main(user_input):
    """Search for passages similar to ``user_input``.

    Returns a formatted string containing the retrieved passages and
    their references. If nothing is found an empty string is returned.
    """

    # Set up ChromaDB client using the recommended method
    client = chromadb.PersistentClient(path="./db")

    # Access the existing collection
    collection = client.get_collection(name="bible_embeddings")

    # Perform a similarity search
    results = search_similar_chunks(user_input, collection)

    # Validate the results structure
    if (
        not results
        or "documents" not in results
        or "metadatas" not in results
        or not results["documents"]
        or not results["documents"][0]
    ):
        return ""

    passages = []
    for document, metadata in zip(results["documents"][0], results["metadatas"][0]):
        book = metadata.get("book", "N/A")
        chapter = metadata.get("chapter", metadata.get("chapter_start", "N/A"))

        verses = metadata.get("verses")
        if verses is None:
            start = metadata.get("verse_start")
            end = metadata.get("verse_end")
            if start is not None and end is not None:
                verses = f"{start}-{end}" if start != end else str(start)
            else:
                verses = "N/A"

        passages.append(f"{book} {chapter}:{verses} - {document}")

    return "\n".join(passages)

# if __name__ == "__main__":
#     main()