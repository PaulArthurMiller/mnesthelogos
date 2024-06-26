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
    # Set up ChromaDB client using the new recommended method
    client = chromadb.PersistentClient(path="./db")

    # Access the existing collection
    collection = client.get_collection(name="bible_embeddings")

    # Perform a similarity search
    query = user_input
    results = search_similar_chunks(query, collection)

    # Check if results is None or empty
    if not results or 'documents' not in results or 'metadatas' not in results:
        print("No results found or unexpected result format.")
        return

    # Check if the expected lists are present and non-empty
    if not results['documents'] or not results['metadatas']:
        print("No documents or metadata found in the results.")
        return

    # Print the results
    for i, (document, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"Result {i + 1}:")
        print(f"Document: {document}")
        print(f"Book: {metadata.get('book', 'N/A')}, Chapter: {metadata.get('chapter', 'N/A')}, Verses: {metadata.get('verses', 'N/A')}")
        print("\n")        

# if __name__ == "__main__":
#     main()