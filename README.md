
# Évaluation de la qualité des requêtes avec QPP et LLM

Ce script évalue la qualité de requêtes de recherche en utilisant à la fois des mesures QPP (pré-retrieval) telles que `idf`, `scq`, `ictf`, et un score qualitatif généré par un LLM local(Ollama). Il compare ensuite ces scores à la performance réelle (via `nDCG@10`) à l’aide de `pytrec_eval`.

---

## Prérequis

- Python 3.8+
- Java (pour Pyserini)
- Serveur LLM local compatible avec l’API Ollama
- Index Lucene préconstruit pour le corpus (par ex. `robust04`)

### Installation des dépendances

```bash
python3.10 -m venv env
source env/bin/activate

pip3 install pandas requests tqdm scipy pyserini pytrec_eval
```

---

## Usage

### Lancer le script :

```bash
python evaluate_queries.py --n 100
```

> `--n` : nombre de requêtes à évaluer (défaut : 250)

---

## Fonctionnalités principales

### Étape 1 : Extraction des requêtes  
À partir du corpus (`robust04` `TREC DL 19`), on récupère les requêtes manuelles.

### Étape 2 : Calcul des scores QPP  
- **idf** : Inverse Document Frequency  
- **scq** : Simplified Clarity Score  
- **ictf** : Inverse Collection Term Frequency

### Étape 3 : Score qualitatif par LLM  
Envoi de chaque requête au modèle local pour retour d’un score entre `0.00` et `1.00`.
Prompt : 
> You are a critical evaluator of search queries. Your goal is to identify weaknesses and ambiguities, not to praise.
> Only give high scores to truly well-formed, precise queries.
> Evaluate how effectively a search query will retrieve relevant documents from a search engine.
> Consider these criteria:
> 1. Clarity - measure divergence KL between query model and collection (Clarity Score)
> 2. Term informativeness - high IDF/ICTF terms
> 3. Specificity - precise phrasing, not too general
> 4. Ambiguity - avoids vague or polysemous terms
> Use this scale strictly:
> - 0.00 to 0.30: Poor or vague queries
> - 0.31 to 0.60: Average queries
> - 0.61 to 0.85: Good queries
> - 0.86 to 1.00: Excellent queries with high clarity and specificity
> Return only a number between 0.00 and 1.00, rounded to two decimals.
> Examples:
> Query: "anorexia nervosa bulimia" → High clarity → Score: 0.85
> Query: "illegal technology transfer" → Very low clarity → Score: 0.10
> Query: "supercritical fluids" → Medium clarity → Score: 0.60

### Étape 4 : Corrélation avec performance réelle  
On compare les scores avec les `nDCG@10` obtenus via `pytrec_eval`.

---

## Résultats

Les résultats sont automatiquement sauvegardés dans un dossier :

```
resultPreRetrieval/YYYY-MM-DD_HH-MM-SS/
├── query_scores.json     # Scores QPP et LLM par requête
├── notes.txt             # Corrélations et statistiques descriptives
```

---

## Sorties principales

- **Matrice de corrélation** entre les scores (idf, scq, ictf, llm_score) et nDCG@10
- **Statistiques par requête** : ID, scores QPP, score LLM, ndcg@10
- **Corrélations Kendall & Pearson** pour chaque score

---

DRAFT
-------------

# Évaluation de la Qualité des Requêtes avec Pyserini, QPP et un LLM

- de **scores QPP pré-retrieval** (`idf`, `scq`, `ictf`),
- d’un **score qualitatif produit par un LLM local** (via Ollama),
- et de la **performance réelle** (via `nDCG@10` à partir des `qrels` et d’un `run` BM25).

---
## Résumé des formules

# Résumé des Formules QPP et nDCG

## 1. **IDF (Inverse Document Frequency)**

Mesure la rareté d’un terme dans la collection :
$\[
\text{idf}(t) = \log \left( \frac{N}{df_t} \right)
\]$$

- *N* : nombre total de documents dans l’index  
- *dfₜ* : nombre de documents contenant le terme *t*

## 2. **SCQ (Similarity Collection Query)**

Mesure combinée de la fréquence du terme et de sa rareté :

$\[
\text{scq}(t) = \left(1 + \log(1 + cf_t)\right) \cdot \log\left(1 + \frac{N}{df_t}\right)
\]$$

- *cfₜ* : fréquence totale du terme *t* dans la collection  
- *dfₜ* : nombre de documents contenant le terme *t*

## 3. **ICTF (Inverse Collection Term Frequency)**

Une autre mesure de rareté du terme dans toute la collection :

$\[
\text{ictf}(t) = \log\left( \frac{N}{cf_t} \right)
\]$$

## 4. **BM25 (Best Matching 25)**


```math
\text{BM25}(t, d) = \text{idf}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}
```


- \( t \) : terme de la requête  
- \( d \) : document  
- \( f(t, d) \) : fréquence du terme \( t \) dans le document \( d \)  
- \( |d| \) : longueur du document (nombre total de mots)  
- \( \text{avgdl} \) : longueur moyenne des documents dans la collection  
- \( k_1 \) : paramètre d'ajustement (souvent entre 1.2 et 2.0)  
- \( b \) : paramètre de normalisation de la longueur (souvent 0.75)  
- \( \text{idf}(t) \) : inverse document frequency du terme \( t \)  

Formule de l’IDF dans BM25 :

```math
\text{idf}(t) = \log \left( \frac{N - df_t + 0.5}{df_t + 0.5} + 1 \right)
```

- \( N \) : nombre total de documents  
- \( df_t \) : nombre de documents contenant le terme \( t \)


## 5. **nDCG@10 (Normalized Discounted Cumulative Gain)**

Évalue la pertinence des résultats en tenant compte de leur rang :

$\[
\text{nDCG@10} = \frac{DCG@10}{IDCG@10}
\]$$

où :

$\[
DCG@10 = \sum_{i=1}^{10} \frac{rel_i}{\log_2(i + 1)}
\]$$

$\[
IDCG@10 = \text{DCG@10 pour un classement idéal}
\]$$

## 6. **Corrélations**

Pour mesurer la corrélation entre un score QPP et la qualité réelle (nDCG), on utilise :

- **Kendall tau** : mesure de concordance d’ordre
- **Pearson** : mesure de corrélation linéaire

---

## Génération du run BM25 (à faire dans le terminal)

```bash
python -m pyserini.search.lucene --topics robust04 --index robust04 --output run.robust04.txt --bm25
```
