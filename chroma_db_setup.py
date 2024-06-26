import json
import chromadb
import uuid
import logging

logging.basicConfig(level=logging.INFO)

def load_embeddings(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in file: {file_path}")
        return []

def create_chroma_db():
    return chromadb.PersistentClient(path="./db")

def get_or_create_collection(client, collection_name):
    try:
        return client.get_collection(name=collection_name)
    except ValueError:
        logging.info(f"Collection '{collection_name}' not found. Creating new collection.")
        return client.create_collection(name=collection_name)

def load_embeddings_to_db(collection, embeddings):
    for embedding in embeddings:
        try:
            chunk_text = embedding['chunk_text']
            metadata_list = embedding['metadata']
            vector = embedding['embedding']
            doc_id = str(uuid.uuid4())

            if not metadata_list:
                continue  # Skip if metadata list is empty

            metadata = {
                "book": metadata_list[0]['book'],
                "chapter_start": metadata_list[0]['chapter'],
                "chapter_end": metadata_list[-1]['chapter'],
                "verse_start": metadata_list[0]['verse'],
                "verse_end": metadata_list[-1]['verse'],
            }

            collection.add(
                ids=[doc_id],
                documents=[chunk_text],
                metadatas=[metadata],
                embeddings=[vector]
            )
        except KeyError as e:
            logging.error(f"Missing key in embedding data: {e}")
        except Exception as e:
            logging.error(f"Error adding embedding to collection: {e}")

def main():
    embeddings_file = 'bible_chunks_embeddings_nt.json'
    collection_name = 'bible_embeddings'

    embeddings = load_embeddings(embeddings_file)
    if not embeddings:
        logging.error("No embeddings loaded. Exiting.")
        return

    client = create_chroma_db()
    collection = get_or_create_collection(client, collection_name)

    load_embeddings_to_db(collection, embeddings)

    count = collection.count()
    logging.info(f"Collection '{collection_name}' now contains {count} embeddings")

if __name__ == "__main__":
    main()