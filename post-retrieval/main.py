import os
import re
import json
import requests
from tqdm import tqdm
from datetime import datetime
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics
import argparse

# === Paramètres ===
parser = argparse.ArgumentParser(description="Post-retrieval difficulty scoring via LLM.")
parser.add_argument("--n", type=int, default=250, help="Nombre de requêtes à traiter (par défaut : 250)")
parser.add_argument("--k", type=int, default=10, help="Top-k documents à récupérer (par défaut : 10)")
args = parser.parse_args()

nombreRequette = args.n
top_k = args.k

corpus = "robust04"

# === Dossier de sauvegarde ===
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = f"result/{now}"
os.makedirs(save_dir, exist_ok=True)

# === Initialisation Pyserini ===
searcher = LuceneSearcher.from_prebuilt_index(corpus)
topics = get_topics(corpus)
queries = {}
for tid in sorted(topics.keys())[:nombreRequette]:
    queries[tid] = topics[tid]['title']

# === Fonction : score LLM difficulté post-retrieval ===
def difficulty_score_via_llm(query, retrieved_titles, model="llama3:8b"):
    doc_list = "\n".join([f"- {title}" for title in retrieved_titles])
    prompt = (
        "You are an expert at evaluating search engine results. "
        "Your goal is to rate how difficult it is to retrieve relevant documents for a given query.\n\n"
        "Consider:\n"
        "- If documents are highly relevant and consistent, difficulty is low.\n"
        "- If they are vague or weakly related, difficulty is high.\n"
        "Return only a score from 0.00 (very easy) to 1.00 (very hard).\n\n"
        f"Query: \"{query}\"\nRetrieved document titles:\n{doc_list}\n\nScore:"
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    raw = response.json()["response"].strip()
    match = re.search(r"\b0(?:\.\d+)?|1(?:\.0+)?\b", raw) # Extraire un score entre 0.00 et 1.00 depuis la réponse texte du LLM
    return float(match.group()) if match else None

# === Traitement principal ===
results = []
for tid, query in tqdm(queries.items(), desc="Évaluation post-retrieval"):
    if tid != 672:
        hits = searcher.search(query, k=top_k)
        titles = [hit.raw.split("\n")[0] for hit in hits if hit.raw]
        score = difficulty_score_via_llm(query, titles)
        results.append({
            "id": tid,
            "query": query,
            "difficulty_score": score
        })

# === Sauvegarde JSON ===
with open(f"{save_dir}/difficulty_scores.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n Scores de difficulté sauvegardés dans : {save_dir}")
