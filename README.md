# Évaluation de la qualité des requêtes avec QPP et LLM

# pré-retrieval

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

# post-retrieval
Ce script évalue la difficulté à retrouver des documents pertinents pour une requête, en se basant sur les résultats de recherche. Il utilise un modèle LLM local pour noter chaque requête selon différentes variantes de présentation des résultats (classement seul, scores BM25 seuls, ou les deux combinés).

---

## Prérequis

- Python 3.8+
- Java (requis pour Pyserini)
- Serveur LLM local (ex. : Ollama avec un modèle compatible type `mistral`)
- Index Pyserini préconstruit (ex. : `robust04`)

### Installation des dépendances

```bash
python3.10 -m venv env
source env/bin/activate

pip install pandas requests tqdm pyserini matplotlib seaborn scipy
```

---

## Usage

```bash
python evaluate_postretrieval.py --n 100 --k 10
```

### Paramètres

- `--n` : nombre de requêtes à traiter (défaut : 250)
- `--k` : nombre de documents à récupérer par requête (top-k, défaut : 10)

---

## Fonctionnement

### Étape 1 : Extraction des requêtes  
Les requêtes sont extraites depuis les topics du corpus (ici `robust04`).

### Étape 2 : Récupération des résultats de recherche  
Pour chaque requête, les `k` premiers documents sont récupérés à l’aide de Pyserini.

### Étape 3 : Génération des prompts  
Trois variantes sont évaluées :
- `ranking_only` : positions des documents sans score
- `bm25_only` : scores BM25 uniquement
- `ranking+bm25` : combinaison des deux

### Étape 4 : Envoi au LLM local  
Chaque prompt est évalué par le modèle via l’API Ollama. Le modèle retourne une **note de difficulté entre 0.00 (facile) et 1.00 (difficile)**.

---

## Résultats

Les scores sont enregistrés automatiquement dans :

```
result/YYYY-MM-DD_HH-MM-SS/
├── ranking_bm25_variants_robust04_<n>queries_top<k>.json
├── correlations.txt   # si un fichier de performance est fourni
```

---

## Analyse facultative

Si un fichier de scores de performance réels (ex. : `perf_scores.json`) est disponible, le script :
- fusionne les scores LLM et les performances (ex. `nDCG@10`)
- calcule les corrélations **Kendall** et **Pearson** pour chaque variante
- génère un fichier texte `correlations.txt` avec les résultats

Le fichier `perf_scores.json` doit contenir :
```json
[
  { "id": 301, "ndcg_cut_10": 0.45 },
  { "id": 302, "ndcg_cut_10": 0.61 },
  ...
]
```

---

## Sorties principales

- Fichier `.json` contenant les scores par variante (`ranking_only`, `bm25_only`, `ranking+bm25`)
- Corrélations (si fichier de performance présent)

---

## Exemple de prompt envoyé

```
You are an expert at evaluating search engine results.
Rate how difficult it is to retrieve relevant documents for a given query.
Return ONLY a score from 0.00 (easy) to 1.00 (hard).
Do not explain. Do not write anything else.

Query: "impact of internet censorship"
Retrieved documents:
1. [DOC]
2. [DOC]
3. [DOC]
...
Score:
```
