import os
import math
import re
import json
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from scipy.stats import kendalltau, pearsonr
from pyserini.index.lucene import LuceneIndexReader as IndexReader
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels_file
import pytrec_eval
import argparse


# === Fonctions de calcul ===
def pre_retrieval_qpp(query, index_reader):
    print("index : ", index_reader)
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

def query_quality_score_via_llm(query, model="mistral"):
    prompt = (
        "You are a query performance predictor. "
        "Evaluate how effectively a search query will retrieve relevant documents from a Lucene index.\n\n"
        "Criteria:\n"
        "1. Clarity (& ambiguity via KL divergence – clarity score) [see Cronen‑Townsend et al. 2002]\n"
        "2. Term informativeness – presence of high‑IDF/ICTF terms [e.g. avgIDF, avgICTF]  \n"
        "3. Specificity – low ambiguity, coherent with collection language model\n\n"
        "Return only a single numeric score between 0.00 (poor) and 1.00 (excellent), rounded to two decimals.\n\n"
        "### Examples from published QPP research:\n"
        # Example 1: Cronen‑Townsend clarity-based
        "Query: \"anorexia nervosa bulimia\" → high clarity (~0.28 in experiments) → Score: 0.85\n"
        # Example 2: low clarity
        "Query: \"illegal technology transfer\" → clarity score ~0.02 (very ambiguous) → Score: 0.10\n"
        "Query: \"supercritical fluids\" → clarity score ~0.16 → Score: 0.60\n\n"
        f"### Now evaluate:\nQuery: \"{query}\" → Score:"
    )
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    raw = response.json()["response"].strip()
    match = re.search(r"\b0(?:\.\d+)?|1(?:\.0+)?\b", raw)
    return float(match.group()) if match else None

# === Création dossier de sauvegarde ===
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = f"resultPreRetrieval/{now}"
os.makedirs(save_dir, exist_ok=True)
corpus =  "dl19-passage" 
corpusRun = "run.dl19-passage.txt"

# === récupérer le nombre de requêtes ===
parser = argparse.ArgumentParser(description="Évaluer la qualité de requêtes avec QPP et LLM.")
parser.add_argument("--n", type=int, default=43, help="Nombre de requêtes à traiter (par défaut : 250)")
args = parser.parse_args()

nombreRequette = args.n

# === Étape 1 : récupérer les requêtes ===
index_reader = IndexReader.from_prebuilt_index("msmarco-v1-passage")
topics = get_topics(corpus)
manual_queries = {tid: topics[tid]['title'] for tid in sorted(topics.keys())[:nombreRequette]}

# === Étape 2 : calculer les scores ===
results = []
for tid, query in tqdm(manual_queries.items(), desc="Évaluation des requêtes"):
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

# === Étape 3 : sauvegarde JSON ===
with open(f"{save_dir}/query_scores.json", "w") as f:
    json.dump(results, f, indent=2)


df = pd.DataFrame(results)
df[["idf", "scq", "ictf", "llm_score"]] = df[["idf", "scq", "ictf", "llm_score"]].astype(float)
correlation_matrix = df[["idf", "scq", "ictf", "llm_score"]].corr()


# === Étape 4 : récupérer qrels et run ===
qrels_path = get_qrels_file(corpus)
with open(qrels_path) as f:
    qrels = pytrec_eval.parse_qrel(f)
with open(corpusRun) as f:
    run = pytrec_eval.parse_run(f)

evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
ndcg_scores = {qid: scores["ndcg_cut_10"] for qid, scores in evaluator.evaluate(run).items()}
df["ndcg@10"] = df["id"].astype(str).map(ndcg_scores).astype(float)


# === Étape 5 : stats et sauvegarde ===
notes = []
notes.append("Matrice de corrélation :\n")
notes.append(correlation_matrix.round(3).to_string())
notes.append("\n\nExtrait des scores par requête :")
notes.append(df[["id", "ndcg@10", "idf", "scq", "ictf", "llm_score"]].to_string(index=False))
notes.append(f"\n\nMoyenne nDCG@10 : {df['ndcg@10'].mean():.4f}\n")

print("\nValeurs uniques par colonne :")
for col in ["idf", "scq", "ictf", "llm_score"]:
    print(f"{col} → uniques: {df[col].nunique()}, NaN: {df[col].isna().sum()}")

    if df[col].nunique() > 1:
        k, _ = kendalltau(df[col], df["ndcg@10"])
        p, _ = pearsonr(df[col], df["ndcg@10"])
        notes.append(f"\n{col} — Kendall: {k:.3f}, Pearson: {p:.3f}")
    else:
        notes.append(f"\n{col} — Pas de variance, impossible de calculer la corrélation.")

with open(f"{save_dir}/notes.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(notes))

print(f"\n✅ Résultats sauvegardés dans : {save_dir}")
