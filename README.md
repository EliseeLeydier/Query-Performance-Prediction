# Évaluation de la qualité des requêtes avec QPP et LLM

Ce script évalue la qualité de requêtes de recherche en combinant des métriques pré-retrieval (QPP) — `idf`, `scq`, `ictf` — et un score qualitatif généré par un LLM local (exécuté via l'API Ollama).  
Il compare ensuite ces scores à la performance réelle (ex. `nDCG@10`, `MAP`, `P@10`) calculée à l’aide de `pytrec_eval`.

---

## Prérequis

- Python 3.8+
- Java (requis par Pyserini)
- Serveur LLM local (ex: [Ollama](https://ollama.com/) avec un modèle type `llama3`)
- Index Lucene préconstruit (ex: `robust04`, `msmarco-v1-passage`)

### Installation des dépendances

```bash
python3.10 -m venv env
source env/bin/activate

pip install pandas requests tqdm scipy pyserini pytrec_eval
```

---

## Usage

```bash
python evaluate_queries.py --n 100 --m n --c r04
```

### Paramètres

- `--n` : nombre de requêtes à traiter (défaut : automatique selon le corpus)
- `--m` : métrique d’évaluation (`n` pour `ndcg_cut_10`, `m` pour `map`, `p` pour `P_10`)
- `--c` : corpus (`r04` pour Robust04, `DL19` pour TREC Deep Learning 2019)

---

## Fonctionnalités principales

### Étape 1 : Récupération des requêtes  
Extraction des requêtes manuelles à partir des topics du corpus sélectionné.

### Étape 2 : Calcul des scores QPP  
- **idf** : Inverse Document Frequency  
- **scq** : Simplified Clarity Score  
- **ictf** : Inverse Collection Term Frequency  
Calculés en amont de la recherche (pré-retrieval).

### Étape 3 : Score qualitatif par LLM  
Chaque requête est envoyée à un modèle local via l’API Ollama.  
Le modèle retourne un score entre `0.00` et `1.00` selon ce prompt strict :

> You are a critical evaluator of search queries. Your goal is to identify weaknesses and ambiguities, not to praise.  
> Only give high scores to truly well-formed, precise queries.  
> Evaluate how effectively a search query will retrieve relevant documents from a search engine.  
> Consider these criteria:  
> 1. Clarity - KL divergence from collection  
> 2. Term informativeness - high IDF/ICTF terms  
> 3. Specificity - precise phrasing  
> 4. Ambiguity - avoid vagueness  
> Use this scale strictly:  
> - 0.00 to 0.30: Poor  
> - 0.31 to 0.60: Average  
> - 0.61 to 0.85: Good  
> - 0.86 to 1.00: Excellent  
> Return only a number between 0.00 and 1.00.

### Étape 4 : Corrélation avec la performance réelle  
Les scores sont comparés à la performance réelle de chaque requête (basée sur les fichiers `qrels` et `run`) via `pytrec_eval`.

---

## Résultats

Les résultats sont enregistrés automatiquement dans :

```
resultat/<corpus>/<métrique>/
├── query_scores.json     # Scores QPP + score LLM pour chaque requête
├── notes.txt             # Matrice de corrélation, stats par requête et corrélations Kendall/Pearson
```

---

## Sorties principales

- Matrice de **corrélation** entre `idf`, `scq`, `ictf`, `llm_score` et la métrique sélectionnée (`nDCG@10`, `MAP`, etc.)
- **Statistiques descriptives** : moyenne, variabilité, valeurs NaN, etc.
- **Corrélations Kendall & Pearson** entre chaque score QPP/LLM et la métrique d’évaluation
- Fichier `.json` regroupant les scores de toutes les requêtes traitées
