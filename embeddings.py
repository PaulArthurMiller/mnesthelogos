import json
import openai
from tqdm import tqdm
import chunk_textiling
from openai import OpenAI
client = OpenAI()

# Ensure you have your OpenAI API key set up
def generate_embeddings(text):
    response = client.embeddings.create(
        input="Your text string goes here",
        model="text-embedding-3-small"
    )    
    return response.data[0].embedding

def save_chunked_data(filename, chunked_data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(chunked_data, f, ensure_ascii=False, indent=4)

def load_chunked_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

# Assuming `chunked_bible_data` is the result from the chunking step
chunked_bible_data = chunk_textiling.map_chunks_to_metadata(chunks, verse_map, concatenated_text)

# Generate embeddings and add to the chunked data
for chunk in tqdm(chunked_bible_data, desc="Generating embeddings"):
    chunk['embedding'] = generate_embeddings(chunk['text'])

# Save the chunked data with embeddings to a file
save_chunked_data('chunked_bible_data_with_embeddings.json', chunked_bible_data)

# Later, you can load the chunked data and embeddings for further processing
loaded_chunked_data = load_chunked_data('chunked_bible_data_with_embeddings.json')

# Print example of loaded data
for chunk in loaded_chunked_data[:3]:  # Show first 3 chunks
    print("Chunk Text:", chunk['text'])
    print("Metadata:", chunk['metadata'])
    print("Embedding:", chunk['embedding'][:10])  # Show first 10 dimensions of the embedding
    print("----")
