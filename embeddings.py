import json
from openai import OpenAI
from typing import List, Dict, Any

# Initialize OpenAI client
client = OpenAI(api_key="OPENAI_API_KEY")

def load_chunks_from_json(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_embeddings(text: str) -> List[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )    
    return response.data[0].embedding

def save_embeddings_to_json(embeddings_data: List[Dict], output_file_path: str):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, ensure_ascii=False, indent=4)

def main(input_file_path: str, output_file_path: str):
    # Load chunks from JSON
    chunks = load_chunks_from_json(input_file_path)
    
    embeddings_data = []
    
    for chunk in chunks:
        chunk_text = chunk['chunk_text']
        metadata = chunk['metadata']
        
        # Generate embedding for the chunk
        embedding = generate_embeddings(chunk_text)
        
        # Append embedding and metadata to the list
        embeddings_data.append({
            'chunk_text': chunk_text,
            'metadata': metadata,
            'embedding': embedding
        })
    
    # Save the embeddings data to JSON
    save_embeddings_to_json(embeddings_data, output_file_path)
    print(f"Embeddings saved to {output_file_path}")

if __name__ == "__main__":
    input_file_path = 'bible_chunks_nt.json'
    output_file_path = 'bible_chunks_embeddings_nt.json'
    main(input_file_path, output_file_path)
