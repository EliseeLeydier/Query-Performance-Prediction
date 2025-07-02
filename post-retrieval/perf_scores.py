import os
import json
import pytrec_eval
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels_file
import argparse

# === Paramètres ===
parser = argparse.ArgumentParser(description="Générer les scores de performance pour les requêtes avec Pyserini et pytrec_eval.")
parser.add_argument("--m", type=str, choices=["n", "m", "p"], default="n", help="Métrique : 'n' pour ndcg_cut_10, 'm' pour map, 'p' pour P_10")
parser.add_argument("--c", type=str, choices=["r04", "DL19"], default="r04", help="Corpus : 'r04' pour Robust04, 'DL19' pour TrecDL19")
args = parser.parse_args()

# === Métrique sélectionnée ===
match args.m:
    case "n":
        metric = "ndcg_cut_10"
    case "m":
        metric = "map"
    case "p":
        metric = "P_10"

# === Corpus ===
match args.c:
    case "r04":
        corpus = "robust04"
    case "DL19":
        corpus = "dl19-passage"

top_k = 1000
output_run = f"my_run_{corpus}.txt"
output_json = f"perf_scores_{corpus}_{metric}.json"

# === Initialisation ===
searcher = LuceneSearcher.from_prebuilt_index(corpus)
topics = get_topics(corpus)

# === Génération du fichier run ===
with open(output_run, "w") as f:
    for tid in sorted(topics):
        query = topics[tid]['title']
        hits = searcher.search(query, k=top_k)
        for rank, hit in enumerate(hits):
            f.write(f"{tid} Q0 {hit.docid} {rank+1} {hit.score} STANDARD\n")

print(f"\nFichier run écrit dans : {output_run}")

# === Récupération automatique des qrels ===
qrels_path = get_qrels_file(corpus)

# === Lecture et évaluation ===
with open(qrels_path) as f:
    qrels = pytrec_eval.parse_qrel(f)
with open(output_run) as f:
    run = pytrec_eval.parse_run(f)

evaluator = pytrec_eval.RelevanceEvaluator(qrels, {metric})
results = evaluator.evaluate(run)

# === Sauvegarde JSON ===
perf_json = [{"id": int(qid), metric: res[metric]} for qid, res in results.items()]
with open(output_json, "w") as f:
    json.dump(perf_json, f, indent=2)

print(f"\nScores {metric} sauvegardés dans : {output_json}")
