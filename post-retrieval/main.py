# === Imports ===
# Bibliothèques système, HTTP, parsing, Pyserini, visualisation et statistiques
import os
import re
import json
import requests
from tqdm import tqdm
from datetime import datetime
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics
import argparse
import pandas as pd

# === Paramètres CLI ===
# ArgumentParser pour choisir le nombre de requêtes à traiter et la valeur de top-k documents
parser = argparse.ArgumentParser(description="Post-retrieval difficulty scoring via LLM with ranking/BM25 variants.")
parser.add_argument("--n", type=int, default=250, help="Nombre de requêtes à traiter (par défaut : 250)")
parser.add_argument("--k", type=int, default=10, help="Top-k documents à récupérer (par défaut : 10)")
args = parser.parse_args()

nombreRequette = args.n
top_k = args.k
corpus = "robust04"  # Corpus fixé ici pour l’instant

# === Dossier de sauvegarde ===
# Crée un dossier de sortie unique horodaté
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = f"result/{now}"
os.makedirs(save_dir, exist_ok=True)

# === Chargement des requêtes et initialisation du moteur de recherche Pyserini ===
searcher = LuceneSearcher.from_prebuilt_index(corpus)
topics = get_topics(corpus)
queries = {tid: topics[tid]['title'] for tid in sorted(topics.keys())[:nombreRequette]}

# === Fonction pour envoyer un prompt à un modèle LLM local via Ollama ===
def send_prompt(prompt, model="mistral"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    raw = response.json()["response"].strip()
    match = re.search(r"\b0(?:\.\d+)?|1(?:\.0+)?\b", raw)
    return float(match.group()) if match else None

# === Génération des prompts selon 3 variantes (ranking_only, bm25_only, ranking+bm25) ===
def build_prompt(query, hits, variant):
    prompt_intro = (
        "You are an expert at evaluating search engine results.\n"
        "Rate how difficult it is to retrieve relevant documents for a given query.\n"
        "Return ONLY a score from 0.00 (easy) to 1.00 (hard).\n"
        "Do not explain. Do not write anything else.\n\n"
        f"Query: \"{query}\"\n"
        "Retrieved documents:\n"
    )

    # Variante : uniquement les rangs
    if variant == "ranking_only":
        content = "\n".join([f"{i+1}. [DOC]" for i in range(len(hits))])

    # Variante : uniquement les scores BM25
    elif variant == "bm25_only":
        content = "\n".join([f"- Score: {hit.score:.2f}" for hit in hits])

    # Variante : rang + score
    elif variant == "ranking+bm25":
        content = "\n".join([f"{i+1}. Score: {hit.score:.2f}" for i, hit in enumerate(hits)])

    else:
        raise ValueError(f"Unknown prompt variant: {variant}")

    return prompt_intro + content + "\n\nScore:"

# === Boucle principale : envoi des requêtes et évaluation par le LLM ===
results = []

for tid, query in tqdm(queries.items(), desc="Évaluation ranking/BM25"):
    if tid != 672:  # Filtrage d’une requête problématique
        hits = searcher.search(query, k=top_k)

        # Génération de trois prompts selon les variantes
        prompt_rank = build_prompt(query, hits, variant="ranking_only")
        prompt_bm25 = build_prompt(query, hits, variant="bm25_only")
        prompt_both = build_prompt(query, hits, variant="ranking+bm25")

        # Envoi des prompts au LLM et récupération des scores
        score_rank = send_prompt(prompt_rank)
        score_bm25 = send_prompt(prompt_bm25)
        score_both = send_prompt(prompt_both)

        # Stockage des résultats
        results.append({
            "id": tid,
            "query": query,
            "score_ranking_only": score_rank,
            "score_bm25_only": score_bm25,
            "score_ranking+bm25": score_both
        })

# === Sauvegarde des résultats au format JSON ===
output_file = os.path.join(
    save_dir,
    f"ranking_bm25_variants_{corpus}_{nombreRequette}queries_top{top_k}.json"
)
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nScores sauvegardés dans : {output_file}")

# === Analyse corrélation automatique (si scores de performance disponibles) ===
perf_scores_path = "perf_scores.json"  # Fichier à fournir manuellement

if os.path.exists(perf_scores_path):
    with open(perf_scores_path) as f:
        perf_data = json.load(f)
    df_llm = pd.DataFrame(results)
    df_perf = pd.DataFrame(perf_data)
    df = pd.merge(df_llm, df_perf, on="id")

    from scipy.stats import kendalltau, pearsonr
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Calcul des corrélations entre chaque variante LLM et la métrique réelle
    variants = [col for col in df.columns if col.startswith("score_")]
    corr_results = {}
    for variant in variants:
        metric = [col for col in df_perf.columns if col != "id"][0]
        df_filtered = df[[variant, metric]].dropna()
        kendall, _ = kendalltau(df_filtered[variant], df_filtered[metric])
        pearson, _ = pearsonr(df_filtered[variant], df_filtered[metric])
        corr_results[variant] = {"kendall": kendall, "pearson": pearson}

    df_corr = pd.DataFrame(corr_results).T.reset_index().rename(columns={"index": "LLM variant"})

    print("\n=== Corrélations entre scores LLM et performance réelle (nDCG@10) ===")
    print(df_corr)

    # === Sauvegarde dans un fichier texte ===
    corr_txt_path = os.path.join(save_dir, "correlations.txt")
    with open(corr_txt_path, "w") as f:
        f.write("=== Corrélations entre scores LLM et performance réelle ===\n")
        f.write(df_corr.to_string(index=False))
    print(f"\nCorrélations sauvegardées dans : {corr_txt_path}")
