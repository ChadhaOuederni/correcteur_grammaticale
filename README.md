#  Correction automatique de texte avec T5 et LoRA

## 📘 Description du projet

Ce projet vise à **entraîner et évaluer un modèle T5** (Text-To-Text Transfer Transformer) pour effectuer une **correction de texte automatique**.  
L’objectif est d’apprendre au modèle à transformer une phrase "brute" ou "erronée" en une version "propre" ou "corrigée".  

Le modèle est affiné grâce à **LoRA (Low-Rank Adaptation)**, une technique légère d’adaptation des poids qui permet de réduire les coûts mémoire et d’entraînement, tout en préservant les performances.

Le script a été exécuté sur **Kaggle** avec GPU activé.

---

## ⚙️ Modèle et outils utilisés

| Élément | Description |
|----------|--------------|
| **Modèle de base** | `t5-small` |
| **Méthode d’adaptation** | LoRA (r=16, alpha=64, dropout=0.05) |
| **Tokenisation** | `T5Tokenizer` (max_length = 128) |
| **Frameworks** | Hugging Face Transformers, Datasets, PEFT |
| **Langage** | Python 3.10+ |
| **Environnement** | GPU (CUDA activé sur Kaggle) |

---

##  Étapes principales du pipeline

### 1️⃣ Chargement et prétraitement des données
- Lecture du fichier `.tsv` par **chunks de 10 000 lignes** pour éviter la surcharge mémoire.  
- Colonnes : `text` (texte original) et `clean` (texte corrigé).  
- Nettoyage des textes (`clean_text`) : suppression des espaces multiples et des caractères inutiles.  
- Filtrage des lignes selon un **ratio de longueur ≤ 2.0** pour éliminer les paires déséquilibrées.

### 2️⃣ Conversion en Dataset Hugging Face
Chaque chunk est transformé en un objet `Dataset`, puis tous sont concaténés.  
Une limite de **100 000 lignes** est fixée pour la démonstration.

Répartition :
- **80 %** pour l’entraînement (`train_dataset`)
- **20 %** pour le test (`test_dataset`)

---

### 3️⃣ Tokenisation
Les textes sont tokenisés avec `T5Tokenizer` :
- Longueur maximale : 128 tokens  
- Padding : `max_length`  
- Masquage des tokens de remplissage (`-100`) dans les labels pour le calcul de la perte.

---

### 4️⃣ Configuration LoRA
Une configuration **LoRA** est appliquée sur les couches d’attention `q` et `v` du modèle :

```python
config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
