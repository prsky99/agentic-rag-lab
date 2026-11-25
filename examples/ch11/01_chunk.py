# examples/ch11/01_chunk.py
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    tenant: str
    text: str
    trust: str          # trusted | internal | untrusted
    source_hash: str
    ingested_at: str
