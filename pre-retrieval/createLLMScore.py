# === Imports ===
# Librairies système, calculs, traitement texte, requêtes, et manipulation de données
import os
import math
import re
import json
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from scipy.stats import kendalltau, pearsonr

# Librairies Pyserini et évaluation TREC
from pyserini.index.lucene import LuceneIndexReader as IndexReader
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels_file
import pytrec_eval
import argparse

# === Fonctions de calcul ===

# Calcule des métriques QPP (pre-retrieval) : idf, scq, ictf
def pre_retrieval_qpp(query, index_reader):
    analyzed_terms = index_reader.analyze(query)
    N = index_reader.stats()['documents']
    idf_list, scq_list, ictf_list = [], [], []

    for term in analyzed_terms:
        try:
            df, cf = index_reader.get_term_counts(term)
            idf = math.log(N / df) if df > 0 else 0
            scq = (1 + math.log(1 + cf)) * math.log(1 + N / df)
            ictf = math.log(N / cf) if cf > 0 else 0
            idf_list.append(idf)
            scq_list.append(scq)
            ictf_list.append(ictf)
        except:
            continue

    return {
        "idf": sum(idf_list) / len(idf_list) if idf_list else 0,
        "scq": sum(scq_list),
        "ictf": sum(ictf_list) / len(ictf_list) if ictf_list else 0
    }

# Envoie une requête à un LLM local pour obtenir un score qualitatif de la requête (0 à 1)
def query_quality_score_via_llm(query, model="llama3:8b"):
    prompt = (
        "You are a critical evaluator of search queries. Your goal is to identify weaknesses and ambiguities, not to praise.'"
        "Only give high scores to truly well-formed, precise queries. "
        "Evaluate how effectively a search query will retrieve relevant documents from a search engine.\n\n"
        "Consider these criteria:\n"
        "1. Clarity - measure divergence KL between query model and collection (Clarity Score)\n"
        "2. Term informativeness - high IDF/ICTF terms\n"
        "3. Specificity - precise phrasing, not too general\n"
        "4. Ambiguity - avoids vague or polysemous terms\n\n"
        "Use this scale strictly:"
        "- 0.00 to 0.30: Poor or vague queries"
        "- 0.31 to 0.60: Average queries"
        "- 0.61 to 0.85: Good queries"
        "- 0.86 to 1.00: Excellent queries with high clarity and specificity"
        "Return only a number between 0.00 and 1.00, rounded to two decimals."
        "### Examples:\n"
        "Query: \"anorexia nervosa bulimia\" → High clarity → Score: 0.85\n"
        "Query: \"illegal technology transfer\" → Very low clarity → Score: 0.10\n"
        "Query: \"supercritical fluids\" → Medium clarity → Score: 0.60\n\n"
    )
    full_prompt = f"{prompt}### Now evaluate:\nQuery: \"{query}\" → Score:"

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": full_prompt, "stream": False}
    )
    raw = response.json()["response"].strip()
    match = re.search(r"\b0(?:\.\d+)?|1(?:\.0+)?\b", raw)
    return float(match.group()) if match else None


# === Gestion des arguments en ligne de commande ===
parser = argparse.ArgumentParser(description="Évaluer la qualité de requêtes avec QPP et LLM.")
parser.add_argument("--n", type=int, help="Nombre de requêtes à traiter (automatique si non précisé)")
parser.add_argument("--m", type=str, choices=["n", "m", "p"], default="n", help="Métrique : 'n', 'm' ou 'p'")
parser.add_argument("--c", type=str, choices=["r04", "DL19"], default="r04", help="Corpus à utiliser")
args = parser.parse_args()

# === Configuration selon le corpus choisi ===
match args.c:
    case "r04":
        corpus = "robust04"
        corpusRun = "run.robust04.txt"
        indexName = "robust04"
        default_n = 300
    case "DL19":
        corpus = "dl19-passage"
        corpusRun = "run.dl19-passage.txt"
        indexName = "msmarco-v1-passage"
        default_n = 43

# Détermination du nombre de requêtes à évaluer
nombreRequette = args.n if args.n is not None else default_n

# Choix de la métrique d’évaluation
match args.m:
    case "n":
        metrique = "ndcg_cut_10"
    case "m":
        metrique = "map"
    case "p":
        metrique = "P_10"

# === Affichage de confirmation des paramètres ===
print(f"Métrique sélectionnée : {metrique}")
print(f"Corpus sélectionné : {corpus}")
print(f"Nombre de requêtes : {nombreRequette}")

# === Création du dossier de sortie pour enregistrer les résultats ===
save_dir = f"resultat/{corpus}/{metrique}"
os.makedirs(save_dir, exist_ok=True)

# === Étape 1 : Récupération des requêtes manuelles ===
index_reader = IndexReader.from_prebuilt_index(indexName)
topics = get_topics(corpus)
manual_queries = {tid: topics[tid]['title'] for tid in sorted(topics.keys())[:nombreRequette]}

# === Étape 2 : Calcul des scores QPP et LLM pour chaque requête ===
results = []
for tid, query in tqdm(manual_queries.items(), desc="Évaluation des requêtes"):
    if (corpus == "robust04" and tid != 672) or corpus == "dl19-passage":
        qpp_scores = pre_retrieval_qpp(query, index_reader)
        llm_score = query_quality_score_via_llm(query)
        results.append({
            "id": tid,
            "query": query,
            "idf": qpp_scores["idf"],
            "scq": qpp_scores["scq"],
            "ictf": qpp_scores["ictf"],
            "llm_score": llm_score
        })

# === Étape 3 : Sauvegarde des résultats dans un fichier JSON ===
with open(f"{save_dir}/query_scores.json", "w") as f:
    json.dump(results, f, indent=2)

# === Analyse de corrélation entre les scores ===
df = pd.DataFrame(results)
df[["idf", "scq", "ictf", "llm_score"]] = df[["idf", "scq", "ictf", "llm_score"]].astype(float)
correlation_matrix = df[["idf", "scq", "ictf", "llm_score"]].corr()

# === Étape 4 : Évaluation des performances via pytrec_eval ===
qrels_path = get_qrels_file(corpus)
with open(qrels_path) as f:
    qrels = pytrec_eval.parse_qrel(f)
with open(corpusRun) as f:
    run = pytrec_eval.parse_run(f)

evaluator = pytrec_eval.RelevanceEvaluator(qrels, {metrique})
ndcg_scores = {qid: scores[metrique] for qid, scores in evaluator.evaluate(run).items()}
df[metrique] = df["id"].astype(str).map(ndcg_scores).astype(float)

# === Étape 5 : Statistiques finales et sauvegarde des résultats ===
notes = []
notes.append("Matrice de corrélation :\n")
notes.append(correlation_matrix.round(3).to_string())
notes.append("\n\nExtrait des scores par requête :")
notes.append(df[["id", metrique, "idf", "scq", "ictf", "llm_score"]].to_string(index=False))
notes.append(f"\n\nMoyenne {metrique} : {df[metrique].mean():.4f}\n")

print("\nValeurs uniques par colonne :")
for col in ["idf", "scq", "ictf", "llm_score"]:
    print(f"{col} → uniques: {df[col].nunique()}, NaN: {df[col].isna().sum()}")

    if df[col].nunique() > 1:
        k, _ = kendalltau(df[col], df[metrique])
        p, _ = pearsonr(df[col], df[metrique])
        notes.append(f"\n{col} — Kendall: {k:.3f}, Pearson: {p:.3f}")
    else:
        notes.append(f"\n{col} — Pas de variance, impossible de calculer la corrélation.")

# Sauvegarde des notes de l’analyse dans un fichier texte
with open(f"{save_dir}/notes.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(notes))

print(f"\n Résultats sauvegardés dans : {save_dir}")