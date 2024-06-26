from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection



# Debug: Print connection attempt
print("Attempting to connect to Milvus...")

# Connect to Milvus
try:
    connections.connect("default", host="172.20.0.2", port="19530")
    print("Connected to Milvus successfully.")
except Exception as e:
    print(f"Failed to connect to Milvus: {e}")

# Define schema fields
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="book", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="chapter", dtype=DataType.INT64),
    FieldSchema(name="verse", dtype=DataType.INT64),
    FieldSchema(name="start_idx", dtype=DataType.INT64),
    FieldSchema(name="end_idx", dtype=DataType.INT64)
]

# Create collection schema
schema = CollectionSchema(fields, "Bible embeddings")

# Create collection
try:
    collection = Collection(name="bible_chunks", schema=schema)
    print("Collection created successfully.")
except Exception as e:
    print(f"Failed to create collection: {e}")
