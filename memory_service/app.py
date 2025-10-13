# memory_service/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import uuid
import time
import json
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import base64

DB = "memory.db"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS memory (
        id TEXT PRIMARY KEY,
        text TEXT,
        source TEXT,
        timestamp REAL,
        embedding BLOB
    )
    """)
    conn.commit()
    conn.close()

def insert_memory(text, source="agent"):
    emb = EMBED_MODEL.encode([text])[0].astype(np.float32)
    emb_bytes = emb.tobytes()
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    id_ = str(uuid.uuid4())
    c.execute("INSERT INTO memory (id, text, source, timestamp, embedding) VALUES (?, ?, ?, ?, ?)",
              (id_, text, source, time.time(), emb_bytes))
    conn.commit()
    conn.close()
    return id_

def query_memory(text, topk=5):
    q_emb = EMBED_MODEL.encode([text])[0].astype(np.float32)
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    rows = c.execute("SELECT id, text, embedding FROM memory").fetchall()
    results = []
    for r in rows:
        id_, txt, emb_bytes = r
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        score = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-8))
        results.append((score, id_, txt))
    results.sort(reverse=True, key=lambda x: x[0])
    conn.close()
    return [{"id": r[1], "text": r[2], "score": r[0]} for r in results[:topk]]

app = FastAPI()

class AddReq(BaseModel):
    text: str
    source: str = "agent"

class QueryReq(BaseModel):
    q: str
    k: int = 5

@app.on_event("startup")
def startup():
    init_db()

@app.post("/memory/add")
def add_memory(req: AddReq):
    id_ = insert_memory(req.text, req.source)
    return {"id": id_}

@app.post("/memory/query")
def query(req: QueryReq):
    return {"results": query_memory(req.q, req.k)}
