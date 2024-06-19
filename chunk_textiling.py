import re
import nltk
from nltk.tokenize import TextTilingTokenizer
from tqdm import tqdm

nltk.download('punkt')

def parse_bible_text(file_path):
    print("Parsing Bible text...")
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    bible_data = []
    current_book = ""
    current_chapter = 0

    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        
        # Match book names
        book_match = re.match(r'^Book \d+ (.+)', line)
        if book_match:
            current_book = book_match.group(1)
            continue
        
        # Match verses
        verse_match = re.match(r'^(\d{3}):(\d{3}) (.+)', line)
        if verse_match:
            chapter = int(verse_match.group(1))
            verse = int(verse_match.group(2))
            text = verse_match.group(3).strip()
            
            bible_data.append({
                'book': current_book,
                'chapter': chapter,
                'verse': verse,
                'text': text
            })
    
    print(f"Parsed {len(bible_data)} verses.")
    return bible_data

def concatenate_verses(bible_data):
    print("Concatenating verses...")
    concatenated_text = ""
    verse_map = []
    
    for entry in bible_data:
        verse_text = f"{entry['book']} {entry['chapter']}:{entry['verse']} {entry['text']}"
        concatenated_text += verse_text + "\n\n"  # Adding paragraph break
        
        verse_map.append({
            'book': entry['book'],
            'chapter': entry['chapter'],
            'verse': entry['verse'],
            'start_idx': len(concatenated_text) - len(verse_text) - 2,  # Adjust for the paragraph break
            'end_idx': len(concatenated_text)
        })
    
    print("Concatenation complete.")
    return concatenated_text.strip(), verse_map

def map_chunks_to_metadata(chunks, verse_map, concatenated_text):
    print("Mapping chunks to metadata...")
    chunked_data = []

    for chunk in tqdm(chunks, desc="Mapping chunks"):
        chunk_metadata = []
        chunk_start_idx = concatenated_text.find(chunk)
        chunk_end_idx = chunk_start_idx + len(chunk)
        
        for verse in verse_map:
            if verse['start_idx'] >= chunk_start_idx and verse['end_idx'] <= chunk_end_idx:
                chunk_metadata.append(verse)
        
        chunked_data.append({
            'text': chunk,
            'metadata': chunk_metadata
        })
    
    print("Mapping complete.")
    return chunked_data

# Parse the text file
bible_data = parse_bible_text('world_english_bible.txt')

# Concatenate the verses
concatenated_text, verse_map = concatenate_verses(bible_data)

# Initialize TextTilingTokenizer
tt = TextTilingTokenizer()

# Chunk the concatenated text
print("Chunking text...")
chunks = tt.tokenize(concatenated_text)
print(f"Chunking complete. Generated {len(chunks)} chunks.")

# Map chunks to metadata
chunked_bible_data = map_chunks_to_metadata(chunks, verse_map, concatenated_text)

# Print example of chunked data
for chunk in chunked_bible_data[:3]:  # Show first 3 chunks
    print("Chunk Text:", chunk['text'])
    print("Metadata:", chunk['metadata'])
    print("----")

# This is 'experiment-chunking' branch
