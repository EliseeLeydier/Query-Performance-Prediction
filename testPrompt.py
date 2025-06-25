import os
import re
import json
import requests
from datetime import datetime
from pyserini.search import get_topics
from tqdm import tqdm


prompts = {
        "prompt1": (
        "Evaluate the quality of the following search query for retrieving relevant documents from a search engine. "
        "Give a score between 0 (very poor) and 1 (excellent) based on the following criteria: "
        "1. Clarity, 2. Specificity, 3. Technicality, 4. Avoidance of ambiguity. "
        "Respond with a single score between 0 and 1, with no explanation."
    ),
       "prompt2": (
        "You are a query performance prediction assistant. "
        "Evaluate how effectively a search query will retrieve relevant documents from a search engine.\n\n"
        "Consider these criteria:\n"
        "1. Clarity - measure divergence KL between query model and collection (Clarity Score)\n"
        "2. Term informativeness - high IDF/ICTF terms\n"
        "3. Specificity - precise phrasing, not too general\n"
        "4. Ambiguity - avoids vague or polysemous terms\n\n"
        "Return only a single numeric score between 0.00 (poor) and 1.00 (excellent), rounded to two decimals.\n\n"
    ),
    "prompt3": (
        "You are a query performance prediction assistant. "
        "Evaluate how effectively a search query will retrieve relevant documents from a search engine.\n\n"
        "Consider these criteria:\n"
        "1. Clarity - measure divergence KL between query model and collection (Clarity Score)\n"
        "2. Term informativeness - high IDF/ICTF terms\n"
        "3. Specificity - precise phrasing, not too general\n"
        "4. Ambiguity - avoids vague or polysemous terms\n\n"
        "Return only a single numeric score between 0.00 (poor) and 1.00 (excellent), rounded to two decimals.\n\n"
        "### Examples:\n"
        "Query: \"anorexia nervosa bulimia\" → High clarity (~0.28 KL) → Score: 0.85\n"
        "Query: \"illegal technology transfer\" → Very low clarity (~0.02 KL) → Score: 0.10\n"
        "Query: \"supercritical fluids\" → Medium clarity (~0.16 KL) → Score: 0.60\n\n"
    ),
    "prompt4": (
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
 
}
# mistral
# ollama run gemma2:2b
# ollama run llama3:8b
def score_with_prompt(prompt, query, model="llama3:8b"):
    full_prompt = f"{prompt}### Now evaluate:\nQuery: \"{query}\" → Score:"
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": full_prompt, "stream": False}
    )
    raw = response.json().get("response", "").strip()
    print("réponse : ", raw)
    match = re.search(r"\d\.\d{2}", raw)
    
    print("========> MATCH : ", match)
    return float(match.group()) if match else None

# === Setup output & topics ===
now = datetime.now().strftime("%Y%m%d_%H%M")
out_dir = f"scoresPrompt"
os.makedirs(out_dir, exist_ok=True)

topics = get_topics("robust04")
queries = [topics[qid]['title'] for qid in sorted(topics.keys())[:30]]  # 30 requêtes

# === Boucle sur les prompts ===
for name, prompt in prompts.items():
    print(f"\n Traitement de {name}...")
    results = []

    for q in tqdm(queries, desc=f"{name} - Scoring"):
        score = score_with_prompt(prompt, q)
        results.append({"query": q, "score": score})

    # === Sauvegarde TXT ===
    with open(os.path.join(out_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
        f.write("============ Prompt ============ \n")
        f.write(prompt)
        f.write("\n ================================ \n")
        f.write("=== Scores ===\n")
        for r in results:
            f.write(f"{r['query']} || {r['score']}\n")


print(f"\n Tous les résultats sont disponibles dans : {out_dir}")
