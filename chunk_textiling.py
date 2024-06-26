import json
import re
from tqdm import tqdm
from typing import List, Dict, Any

def parse_bible_text(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    bible_verses = []
    current_book = ""
    current_chapter = 0

    book_pattern = re.compile(r"^Book (\d+) (.+)$")
    verse_pattern = re.compile(r"^(\d{3}):(\d{3}) (.+)$")

    for line in lines:
        line = line.strip()
        book_match = book_pattern.match(line)
        verse_match = verse_pattern.match(line)

        if book_match:
            current_book = book_match.group(2)
            current_chapter = 0
        elif verse_match:
            chapter, verse, text = verse_match.groups()
            chapter = int(chapter)
            verse = int(verse)
            if current_chapter != chapter:
                current_chapter = chapter

            bible_verses.append({
                'book': current_book,
                'chapter': chapter,
                'verse': verse,
                'text': text,
                'start_idx': 0,  # Placeholder, will be set later
                'end_idx': 0     # Placeholder, will be set later
            })
        elif current_book and bible_verses:
            # Append to the last verse's text if it's a continuation
            bible_verses[-1]['text'] += ' ' + line

    return bible_verses

def set_indices(verses: List[Dict[str, Any]]) -> None:
    idx = 0
    for verse in verses:
        verse['start_idx'] = idx
        idx += len(verse['text']) + 1  # +1 for the space added during concatenation
        verse['end_idx'] = idx - 1

def concatenate_verses(verses: List[Dict[str, Any]]) -> str:
    return ' '.join([verse['text'] for verse in verses])

def chunk_text(verses: List[Dict[str, Any]], max_chunk_size: int = 500) -> List[str]:
    chunks = []
    current_chunk = []
    current_length = 0

    for verse in verses:
        verse_text = verse['text']
        if current_length + len(verse_text) > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        current_chunk.append(verse_text)
        current_length += len(verse_text)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def map_chunks_to_metadata(chunks: List[str], verses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    index_mapping = []
    verse_index = 0
    chunk_start_idx = 0

    for chunk in chunks:
        chunk_end_idx = chunk_start_idx + len(chunk)
        chunk_metadata = []
        chunk_book = ""

        while verse_index < len(verses) and verses[verse_index]['start_idx'] < chunk_end_idx:
            verse = verses[verse_index]
            if verse['end_idx'] >= chunk_start_idx:
                if not chunk_book:
                    chunk_book = verse['book']
                chunk_metadata.append({
                    'book': verse['book'],
                    'chapter': verse['chapter'],
                    'verse': verse['verse'],
                    'start_idx': max(verse['start_idx'], chunk_start_idx),
                    'end_idx': min(verse['end_idx'], chunk_end_idx)
                })
            verse_index += 1

        index_mapping.append({
            'chunk_text': chunk,
            'book': chunk_book,
            'metadata': chunk_metadata
        })
        chunk_start_idx = chunk_end_idx

    return index_mapping

def save_chunks_to_json(chunks_with_metadata: List[Dict[str, Any]], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=4)

import random

def print_sample_chunks(chunks_with_metadata: List[Dict[str, Any]], num_samples: int = 3):
    total_chunks = len(chunks_with_metadata)
    sample_indices = random.sample(range(total_chunks), min(num_samples, total_chunks))
    
    print(f"\nSample of {num_samples} chunks with metadata:")
    for idx in sample_indices:
        chunk = chunks_with_metadata[idx]
        print(f"\nChunk {idx}:")
        print("Text:", chunk['chunk_text'])
        print("Metadata:")
        if chunk['metadata']:
            book = chunk['metadata'][0]['book']  # Assuming the book is the same for all verses in a chunk
            print(f"  Book: {book}")
            for verse in chunk['metadata']:
                print(f"  Chapter {verse['chapter']}, Verse {verse['verse']}")
        else:
            print("  No metadata available for this chunk")
        print("-" * 50)

def main():
    input_file = 'world_english_bible.txt'
    output_file = 'bible_chunks_nt.json'

    print("Parsing Bible text...")
    verses = parse_bible_text(input_file)
    print(f"Parsed {len(verses)} verses.")

    print("Setting indices...")
    set_indices(verses)

    print("Concatenating verses...")
    concatenated_text = concatenate_verses(verses)
    print(f"Concatenation complete. Total length: {len(concatenated_text)}")

    print("Chunking text...")
    chunks = chunk_text(verses)
    print(f"Chunking complete. Generated {len(chunks)} chunks.")

    print("Mapping chunks to metadata...")
    chunks_with_metadata = map_chunks_to_metadata(chunks, verses)
    print(f"Mapping complete. Total chunks: {len(chunks_with_metadata)}")

    print_sample_chunks(chunks_with_metadata)

    print("Saving chunks to JSON file...")
    save_chunks_to_json(chunks_with_metadata, output_file)
    print(f"Chunks saved to {output_file}")

if __name__ == "__main__":
    main()
