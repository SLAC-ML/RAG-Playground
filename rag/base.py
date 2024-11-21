import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import threading


# File paths
INDEX_FILE = 'data/knowledge_base.index'
ENTRIES_FILE = 'data/entries.npy'


# Initialize global variables
index = None
entries = []
lock = threading.Lock()

# Initialize the model
model = SentenceTransformer("maidalun1020/bce-embedding-base_v1")
# model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True)


def init_knowledge_base():

    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(ENTRIES_FILE), exist_ok=True)

    global index, entries

    if os.path.exists(INDEX_FILE) and os.path.exists(ENTRIES_FILE):
        print("Loading existing FAISS index and entries...")
        # Load FAISS index
        index = faiss.read_index(INDEX_FILE)
        # Load entries
        entries = np.load(ENTRIES_FILE, allow_pickle=True).tolist()
    else:
        print("Initializing new FAISS index and entries...")
        # Initialize empty entries
        entries = []
        # Initialize FAISS index
        dimension = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dimension)
        # Save the empty index and entries
        faiss.write_index(index, INDEX_FILE)
        np.save(ENTRIES_FILE, np.array(entries, dtype=object))


def save_data():
    # Save the updated index and entries to disk

    global index, entries

    faiss.write_index(index, INDEX_FILE)
    np.save(ENTRIES_FILE, np.array(entries, dtype=object))
    print('Data saved.')


# Function to add a new entry
def add_entries(entry_list: list[str]):

    global entries

    embedding = model.encode(entry_list, show_progress_bar=True)

    with lock:
        index.add(embedding)
        entries += entry_list
        save_data()


# Function to list entries
def list_entries(n=0):
    # Return all entries if n == 0, else return the last n entries

    global entries

    return entries[::-1] if n == 0 else entries[::-1][:n]


# Function to search
def search(query, top_k=5):

    global index, entries, lock

    query_embedding = model.encode([query])
    with lock:
        distances, indices = index.search(query_embedding, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            entry = entries[idx]
            distance = distances[0][i]
            results.append({
                "entry": entry,
                "distance": float(distance),
            })

    return results
