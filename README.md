# Lab_NLP
# Détection de Sarcasme - Analyse NLP

Ce projet a pour objectif de développer et d'évaluer des modèles de classification binaire pour distinguer les titres d'articles sarcastiques des titres non-sarcastiques.

## 1. Données

Le jeu de données utilisé est `Sarcasm.json`, téléchargé via KaggleHub. Il contient des titres d'articles de presse et une étiquette (`is_sarcastic`) indiquant si le titre est sarcastique (1) ou non (0).

*   **Source:** [shariphthapa/sarcasm-json-datasets](https://www.kaggle.com/datasets/shariphthapa/sarcasm-json-datasets)
*   **Shape initiale:** (26709, 3) après chargement.
*   **Colonnes:** `article_link`, `headline`, `is_sarcastic`.
*   **Distribution de la cible:** 
    *   Non-sarcastique (0): 14985 (56.1%)
    *   Sarcastique (1): 11724 (43.9%)

La colonne `article_link` a été supprimée car elle n'était pas pertinente pour la classification.

## 2. Prétraitement Textuel

Les étapes de prétraitement suivantes ont été appliquées aux titres :

*   **Nettoyage:** Conversion en minuscules, suppression de la ponctuation.
*   **Tokenisation:** Utilisation de `nltk.word_tokenize` pour diviser les phrases en mots.
*   **Suppression des mots vides (Stop Words):** Suppression des mots courants (comme 'the', 'a', 'is') à l'aide de la liste `nltk.corpus.stopwords`.
*   **Tokenization Keras:** Les tokens nettoyés sont convertis en séquences numériques à l'aide de `tensorflow.keras.preprocessing.text.Tokenizer`.
    *   `num_words=5000`: Le vocabulaire est limité aux 5000 mots les plus fréquents.
    *   `oov_token='<OOV>'`: Un token spécial est utilisé pour les mots hors vocabulaire.
*   **Padding:** Les séquences sont complétées ou tronquées à une longueur maximale (`max_len = 27`) pour uniformiser la taille des entrées du modèle.

## 3. Modèles de Classification

Deux approches de régression logistique ont été comparées :

### 3.1. Baseline: Régression Logistique sur Séquences Directes

Ce modèle utilise les séquences de tokens `padded_sarcasm` directement comme entrées. Chaque mot est représenté par son index numérique unique.

*   **Modèle:** `LogisticRegression` de scikit-learn.
*   **Paramètres:** `max_iter=5000`, `random_state=42`, `class_weight='balanced'`, `solver='lbfgs'`.

### 3.2. Régression Logistique sur Embeddings

Cette approche introduit une couche d'embedding pour transformer les séquences de tokens en représentations vectorielles denses et continues. Un pooling par moyenne est appliqué aux embeddings pour obtenir une représentation vectorielle unique par titre, qui est ensuite utilisée comme entrée pour la régression logistique.

*   **Couche d'Embedding:** `tensorflow.keras.layers.Embedding`
    *   `input_dim=vocab_size` (5001)
    *   `output_dim=64` (dimension de l'embedding)
    *   `input_length=max_len` (27)
*   **Pooling:** Les embeddings des mots d'un titre sont moyennés pour former un seul vecteur de 64 dimensions.
*   **Modèle:** `LogisticRegression` de scikit-learn, avec les mêmes paramètres que le modèle baseline.

## 4. Évaluation et Résultats

Les modèles ont été évalués sur un jeu de test de 30% des données, en utilisant l'Accuracy, l'AUC ROC, la matrice de confusion et le rapport de classification.

### 4.1. Résultats du Modèle Baseline (Séquences Directes)

*   **Accuracy:** 0.5657
*   **ROC AUC:** 0.5623
*   **Rappel Sarcastic:** 0.41
*   **Commentaires:** Performance médiocre, à peine meilleure que le hasard, avec une difficulté significative à détecter le sarcasme (faible rappel).

### 4.2. Résultats du Modèle avec Embeddings

*   **Accuracy:** 0.5547
*   **ROC AUC:** 0.5909
*   **Rappel Sarcastic:** 0.54
*   **Commentaires:** 
    *   Légère baisse d'Accuracy (-1.10%) par rapport au baseline.
    *   **Amélioration notable de l'AUC ROC** (+0.0285), indiquant une meilleure capacité à distinguer les classes.
    *   **Amélioration du rappel pour la classe 'Sarcastic'** (de 0.41 à 0.54), signifiant une meilleure détection du sarcasme.
    *   Cette amélioration s'est faite au prix d'une augmentation des faux positifs pour la classe 'Non-Sarcastic'.

## 5. Conclusion

L'intégration d'une couche d'embedding a permis au modèle de mieux capturer les nuances sémantiques des titres, conduisant à une amélioration de la capacité de discrimination entre les classes (ROC AUC) et un meilleur rappel pour la détection du sarcasme. Bien que la précision globale n'ait pas augmenté, l'approche par embeddings démontre un potentiel prometteur pour cette tâche, suggérant que des architectures de réseaux de neurones plus complexes basées sur ces embeddings pourraient offrir de meilleures performances.

## 6. Credits

* Author: Azami Hassani Adnane.
* Supervisor: Prof. Masrour Tawfik.
