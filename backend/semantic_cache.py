import os
import json
import numpy as np
import faiss
from langchain_openai import OpenAIEmbeddings

CACHE_FILE_PATH = "finsight_metadata/semantic_cache.json"
INDEX_FILE_PATH = "finsight_metadata/semantic_cache.index"
EMBEDDING_DIM = 1536 # For text-embedding-3-small

class SemanticCache:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.cache_data = [] # List of dicts: {"query": ..., "sql": ..., "tenant_id": ...}
        self.index = None
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(CACHE_FILE_PATH) and os.path.exists(INDEX_FILE_PATH):
            try:
                with open(CACHE_FILE_PATH, 'r') as f:
                    self.cache_data = json.load(f)
                self.index = faiss.read_index(INDEX_FILE_PATH)
            except Exception as e:
                print(f"[SemanticCache] Error loading cache: {e}")
                self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        else:
            # Inner product (cosine similarity if vectors are normalized)
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)

    def _save_cache(self):
        try:
            with open(CACHE_FILE_PATH, 'w') as f:
                json.dump(self.cache_data, f)
            faiss.write_index(self.index, INDEX_FILE_PATH)
        except Exception as e:
            print(f"[SemanticCache] Error saving cache: {e}")

    def _has_relative_dates(self, query: str) -> bool:
        """Check if query has relative dates, which shouldn't be cached to avoid staleness."""
        date_terms = ["today", "yesterday", "tomorrow", "this month", "last month", "next month", 
                     "this year", "last year", "next year", "this quarter", "last quarter"]
        query_lower = query.lower()
        return any(term in query_lower for term in date_terms)

    async def search(self, query: str, tenant_id: int, allowed_partner_ids: list[int], threshold: float = 0.95) -> str | None:
        """Search the cache for a highly similar query. Returns the cached SQL if found."""
        if not self.cache_data or self.index.ntotal == 0:
            return None
        
        if self._has_relative_dates(query):
            return None

        # Embed query
        query_vec = await self.embeddings.aembed_query(query)
        # Normalize for cosine similarity
        query_np = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(query_np)

        # Search
        scores, indices = self.index.search(query_np, 1)
        if len(scores) > 0 and len(scores[0]) > 0:
            best_score = scores[0][0]
            best_idx = indices[0][0]

            if best_score >= threshold and best_idx < len(self.cache_data):
                entry = self.cache_data[best_idx]
                # Validate tenant boundaries AND exact allowed_partner_ids scope
                if entry.get("tenant_id") == tenant_id and sorted(entry.get("allowed_partner_ids", [])) == sorted(allowed_partner_ids):
                    return entry.get("sql")
        return None

    async def add(self, query: str, sql: str, tenant_id: int, allowed_partner_ids: list[int]):
        """Add a successful query and its SQL to the cache."""
        if self._has_relative_dates(query):
            return

        query_vec = await self.embeddings.aembed_query(query)
        query_np = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(query_np)

        self.index.add(query_np)
        self.cache_data.append({
            "query": query,
            "sql": sql,
            "tenant_id": tenant_id,
            "allowed_partner_ids": allowed_partner_ids
        })
        self._save_cache()

# Global singleton
semantic_cache = SemanticCache()
