

Code memento · MD
# Code-mémento — projet `dslr` (Data Science × Logistic Regression)
 
Ce document rassemble, fichier par fichier et dans l'ordre logique du projet, une explication détaillée de chaque fonction du code — dans l'ordre où elle s'exécute réellement, pas dans l'ordre où elle est écrite dans le fichier. Objectif : que n'importe qui puisse comprendre et défendre ce code, même sans l'avoir écrit soi-même.
 
## Sommaire
 
1. [`describe.py`](#describepy) — analyse statistique manuelle du dataset
2. [`histogram.py`](#histogrampy) — quelle matière a une distribution homogène entre les 4 maisons ?
3. [`scatter_plot.py`](#scatter_plotpy) — quelles sont les deux features similaires ?
4. [`pair_plot.py`](#pair_plotpy) — quelles features choisir pour la régression logistique ?
5. [`logreg_train.py`](#logreg_trainpy) — entraînement des 4 modèles (one-vs-all)
6. [`logreg_predict.py`](#logreg_predictpy) — prédiction sur le jeu de test
7. [`my_logistic_regression.py`](#my_logistic_regressionpy) — la classe utilisée par les deux scripts précédents
---
 
# `describe.py` — memento dans l'ordre d'exécution
 
**Rôle du fichier** : recalculer, sans aucune fonction statistique toute faite, l'équivalent de `pandas.describe()` sur les colonnes numériques d'un dataset, **plus 5 statistiques bonus** (Missing, Variance, Range, IQR, Skewness). Usage : `python3 describe.py dataset_train.csv`.
 
Ce document suit le programme **dans l'ordre où il s'exécute réellement**, pas dans l'ordre où les fonctions sont écrites dans le fichier.
 
---
 
## Étape 1 — Point d'entrée : `if __name__ == "__main__":`
 
C'est ici que tout commence quand on lance `python3 describe.py dataset_train.csv`.
 
1. Vérifie qu'un argument (le nom du fichier) a bien été donné en ligne de commande, sinon affiche une erreur et s'arrête (`sys.exit(1)`).
2. Vérifie que le fichier existe bel et bien sur le disque (`Path.exists()` et `is_file()`).
3. Lit la première ligne du CSV à la main (`f.readline()`) pour récupérer les noms de colonnes (`features`) — une alternative simple à `pandas`, qui montre qu'on n'a pas besoin de `df.columns` pour ça.
4. Charge tout le CSV avec `pd.read_csv` (utiliser pandas pour la **lecture** du fichier est autorisé par le sujet ; ce qui est interdit, ce sont les fonctions de **calcul statistique** comme `.describe()`, `.mean()`, etc.), converti en tableau numpy brut.
5. Appelle `numerical_data(data, features)` — étape 2 ci-dessous.
6. Puis appelle `data_display(data_model, features_model)` avec le résultat — étape 4 ci-dessous.
---
 
## Étape 2 — `numerical_data(data: np.ndarray, features)`
 
Filtre le dataset brut pour ne garder **que les colonnes numériques** (élimine "First Name", "Hogwarts House", "Best Hand", etc., qui sont des colonnes textuelles).
 
**Logique** :
1. Regarde la **première ligne** du dataset (`first_line_data`) pour déterminer, colonne par colonne, si la valeur est convertible en `float`.
2. Si oui, garde le nom de la colonne (`numeric_data_features`) et toute la colonne de données (`new_data`).
3. Si la conversion échoue (`ValueError`), la colonne est ignorée entièrement.
4. Convertit le résultat en tableau numpy de type `float`, puis le transpose (`.T`) pour repasser d'une liste de colonnes à un tableau lignes × colonnes classique.
5. Appelle enfin `supp_index(new_data, numeric_data_features)` — étape 3 ci-dessous — pour retirer la colonne "Index" (qui est numérique donc aurait été gardée à l'étape précédente).
**Limite à connaître** : cette détection "numérique ou pas" ne regarde que la **première ligne** du fichier. Si par malheur la toute première ligne avait une valeur manquante sur une colonne numérique, cette colonne serait ignorée à tort. Dans la pratique, avec `dataset_train.csv`, ça ne pose pas de problème (vérifié).
 
---
 
## Étape 3 — `supp_index(new_data, numeric_data_features)`
 
Appelée depuis l'intérieur de `numerical_data`. Retire la colonne "Index" du tableau de données et de la liste des noms de colonnes, si elle est présente. Le sujet ne demande pas d'afficher les statistiques de la colonne technique "Index" (elle contient juste le numéro de ligne, sans intérêt statistique).
 
**Logique** : cherche "Index" ou "index" dans la liste des noms, et si trouvé, supprime la colonne correspondante du tableau numpy (`np.delete`, `axis=1` = suppression d'une colonne) et le nom associé, puis s'arrête (`break`, une seule colonne "Index" est attendue).
 
Une fois cette étape terminée, `numerical_data` renvoie son résultat au `main`, qui appelle ensuite `data_display`.
 
---
 
## Étape 4 — `data_display(data, features)`
 
La fonction "chef d'orchestre" : calcule et affiche, ligne par ligne, les 8 statistiques du tableau final (Count, Mean, Std, Min, Max, 25%, 50%, 75%). Elle appelle, **dans cet ordre précis**, toutes les fonctions détaillées dans les étapes suivantes.
 
### 4.1 — L'en-tête des colonnes (premier affichage)
 
```python
for ft in features:
    displayed = trunc(str(ft))
    print(f"{displayed}{add_space(displayed)}", end="")
```
 
C'est la toute première chose affichée à l'écran, avant le moindre calcul statistique. Utilise `trunc` et `add_space` (détaillées en étape 4.2) pour formater proprement les noms de colonnes.
 
### 4.2 — `trunc(text: str)` et `add_space(text: str)`
 
Ces deux fonctions utilitaires sont appelées ici pour la première fois, et seront réutilisées à chaque ligne du tableau.
 
**`trunc(text)`** : coupe une chaîne de caractères à ses 10 premiers caractères (`text[:10]`). Sert uniquement pour les **noms de colonnes** (en-tête) et la ligne **Count**, afin qu'un nom de matière trop long (ex : "Defense Against the Dark Arts") ne casse pas l'alignement du tableau (il devient "Defense Ag").
 
**`add_space(text) -> str`** : calcule le nombre d'espaces à ajouter après un texte pour que chaque colonne fasse une largeur totale fixe de 15 caractères. Logique : `15 - longueur du texte`, avec un minimum forcé à 1 espace.
 
*Point corrigé* : une première version mettait un forfait fixe de 5 espaces dès que le texte dépassait 10 caractères, sans recalculer selon la longueur exacte. Problème : deux valeurs différentes de 10 et 11 caractères (ex : `49634.5702` vs `-24370.0000`, à cause du signe négatif) recevaient toutes les deux 5 espaces, donnant des largeurs de colonnes différentes (15 vs 16) et décalant l'alignement entre certaines lignes du tableau. La version actuelle recalcule toujours l'espacement exact à partir de la longueur réelle du texte affiché, ce qui garantit un alignement constant partout.
 
### 4.3 — `display_count(data, features)`
 
Appelée juste après l'en-tête, pour produire la ligne "Count". Compte le nombre de valeurs valides (non vides, non `NaN`, convertibles en `float`) par colonne.
 
**Point important** : cette fonction a été corrigée — une première version ne vérifiait pas les valeurs et renvoyait `len(data)` (le nombre total de lignes) pour **toutes** les colonnes, peu importe le nombre réel de valeurs manquantes. La version actuelle vérifie chaque valeur une par une, exactement comme `display_min`/`display_max` (étape 4.6), et ne compte que les valeurs réellement exploitables.
 
Son résultat (`count`) est **conservé** pour être réutilisé plus tard par `display_std` (étape 4.5) et `display_quartiles` (étape 4.7).
 
### 4.4 — `display_mean(data, features)`
 
Appelée juste après, pour la ligne "Mean". Calcule la moyenne de chaque colonne : pour chaque colonne, fait la somme des valeurs valides (`count_sum`) et compte combien il y en a (`count_len`), en ignorant les vides/`NaN`/non-convertibles — puis divise la somme par le compte. Retourne `None` si une colonne n'a aucune valeur valide (évite une division par zéro).
 
Son résultat (`mean`) est **conservé** pour être réutilisé juste après par `display_std`.
 
### 4.5 — `display_std(data, features, mean, count)`
 
Appelée ensuite, pour la ligne "Std". Reçoit `mean` (calculé à l'étape 4.4) et `count` (calculé à l'étape 4.3), qu'elle réutilise directement au lieu de tout recalculer.
 
**Logique** :
1. Pour chaque colonne, fait la somme des carrés des écarts à la moyenne : `(valeur - moyenne)²`, cumulés dans `res_sum`.
2. Divise cette somme par `count[idx_col] - 1` (et non `count[idx_col]`) — c'est la **correction de Bessel**, qui donne l'écart-type "échantillon" utilisé par défaut par `pandas.describe()` (diviser par `n` donnerait l'écart-type "population", légèrement plus petit). Une sécurité (`ddof = ... if count > 1 else 1`) évite une division par zéro si une colonne n'a qu'une seule valeur valide.
3. Prend la racine carrée du résultat (`** 0.5`).
**Formule** : `std = √( Σ(x - moyenne)² / (n - 1) )`
 
### 4.6 — `display_min(data, features)` puis `display_max(data, features)`
 
Appelées ensuite, l'une après l'autre, pour les lignes "Min" et "Max". Cherchent respectivement le minimum et le maximum de chaque colonne, à la main.
 
**Logique** (identique pour les deux, seule la comparaison change) :
- Pour chaque colonne, parcourt toutes les lignes une par une.
- Ignore une valeur si elle est vide (`""`) ou `NaN` (`pd.isna`).
- Convertit en `float`, et met à jour `mini`/`maxi` si la valeur bat le record actuel.
- Si la conversion en `float` échoue (`ValueError`), la ligne est ignorée (sécurité contre des valeurs non numériques résiduelles).
### 4.7 — `display_quartiles(data, features, count)`
 
Appelée en dernier, pour les 3 lignes "25%", "50%", "75%". Reçoit `count` (calculé à l'étape 4.3) mais ne l'utilise **jamais** dans son corps — paramètre mort, sans impact sur le résultat, juste un peu de code à nettoyer si on veut être strict.
 
**Logique** :
1. Convertit chaque colonne en tableau numpy, en remplaçant les valeurs vides par `NaN`.
2. Retire les `NaN` de la colonne (`valid_data`), puis trie les valeurs restantes (`np.sort`).
3. Calcule une **position théorique** dans le tableau trié pour chaque quartile : par exemple pour le 25%, la position est `(n-1) * 0.25`.
4. Si cette position tombe pile sur un indice entier, on prend directement la valeur à cet indice.
5. Sinon, on **interpole linéairement** entre les deux valeurs encadrantes (`lower` et `upper`), pondérées par la distance à chacune (`weight`). C'est la même méthode que `numpy.percentile` (méthode linéaire par défaut) et que `pandas.describe()`.
**Retourne** : trois listes (`q1`, `med`, `q3`), une valeur par colonne — c'est la dernière chose calculée pour les 8 statistiques du mandatory.
 
### 4.8 — Bonus : 5 statistiques supplémentaires
 
Après les 8 lignes du mandatory (identiques à `pandas.describe()`), `data_display` enchaîne sur 5 lignes bonus, sur le même principe : tout calculé à la main, rien de tout fait.
 
**`display_missing(data, features, count)`** — ligne "Missing". Calcule simplement `len(data) - count[idx_col]` pour chaque colonne : le nombre total de lignes moins le nombre de valeurs valides déjà connu (calculé à l'étape 4.3) donne directement le nombre de valeurs manquantes. Aucun nouveau parcours des données n'est nécessaire, juste une soustraction.
 
**`display_variance(std)`** — ligne "Variance". Encore plus direct : la variance est par définition le carré de l'écart-type (`s ** 2` pour chaque valeur de `std`, déjà calculé à l'étape 4.5). Une simple liste en compréhension, aucune boucle sur les données.
 
**`display_range(mini, maxi)`** — ligne "Range". `maxi - mini` pour chaque colonne, réutilisant les résultats déjà calculés aux étapes 4.6.
 
**`display_iqr(q1, q3)`** — ligne "IQR" (écart interquartile). `q3 - q1` pour chaque colonne, réutilisant les quartiles de l'étape 4.7.
 
**`display_skewness(data, features, mean, count)`** — ligne "Skewness", la seule des 5 qui recalcule vraiment quelque chose de nouveau à partir des données brutes (les 4 précédentes ne faisaient que combiner des résultats déjà obtenus). Mesure l'asymétrie de la distribution :
 
```python
diff = value - mean[idx_col]
m2_sum += diff ** 2
m3_sum += diff ** 3
...
m2 = m2_sum / n
m3 = m3_sum / n
skew = m3 / (m2 ** 1.5)
```
 
C'est le **coefficient de Fisher-Pearson** : `m2` est le moment d'ordre 2 (la variance en version "population", divisée par `n` et non `n-1`), `m3` le moment d'ordre 3. Le rapport `m3 / m2^1.5` normalise le résultat pour qu'il ne dépende pas de l'échelle de la variable — validé numériquement contre `scipy.stats.skew()` (écart négligeable, ~5e-5, de l'ordre de l'arrondi flottant).
 
**Lecture du résultat** : une skewness proche de 0 indique une distribution symétrique (ex : Arithmancy, Astronomy). Une skewness négative indique une distribution étalée vers la gauche, avec une "queue" de valeurs basses (ex : Divination, History of Magic). Une skewness positive indique l'inverse, une queue vers les valeurs hautes (ex : Muggle Studies, Flying).
 
---
 
## Résumé du déroulé complet
 
```
main
 └─ numerical_data(data, features)
     └─ supp_index(new_data, numeric_data_features)
 └─ data_display(data_model, features_model)
     ├─ [en-tête] trunc + add_space
     ├─ display_count       → ligne "Count"   (résultat retenu pour la suite)
     ├─ display_mean        → ligne "Mean"    (résultat retenu pour la suite)
     ├─ display_std         → ligne "Std"     (utilise mean + count)
     ├─ display_min         → ligne "Min"
     ├─ display_max         → ligne "Max"
     ├─ display_quartiles   → lignes "25%", "50%", "75%"
     │
     ├─ [BONUS] display_missing    → ligne "Missing"   (len(data) - count)
     ├─ [BONUS] display_variance   → ligne "Variance"  (std au carre)
     ├─ [BONUS] display_range      → ligne "Range"     (max - min)
     ├─ [BONUS] display_iqr        → ligne "IQR"       (q3 - q1)
     └─ [BONUS] display_skewness   → ligne "Skewness"  (nouveau calcul : moments d'ordre 2 et 3)
```
-e 
 
---
 
 
# `histogram.py` — memento dans l'ordre d'exécution
 
**Rôle du fichier** : afficher un histogramme par matière, avec les 4 maisons superposées en couleur, pour répondre à la question du sujet : *quelle matière a une distribution homogène entre les 4 maisons ?* Usage : `python3 histogram.py dataset_train.csv`.
 
**Verdict de la relecture** : ce fichier ne contient aucun bug, rien n'a été modifié. Structure la plus simple des 4 fichiers de visualisation (seulement 2 fonctions).
 
---
 
## Étape 1 — Point d'entrée : `if __name__ == "__main__":`
 
1. Vérifie qu'il y a exactement 2 arguments (le nom du script + le fichier CSV) — plus strict que `describe.py` qui acceptait "2 ou plus" (`< 2`), ici c'est `!= 2` : ni trop peu, ni trop d'arguments.
2. Appelle `load_data(filename)` — étape 2 ci-dessous — pour charger le CSV.
3. Appelle `plot_histogram(df)` avec le résultat — étape 3 ci-dessous — pour tracer et afficher les graphes.
---
 
## Étape 2 — `load_data(filename)`
 
Charge le CSV en toute sécurité.
 
**Logique** : essaie `pd.read_csv(filename)` dans un `try`. Si ça échoue pour n'importe quelle raison (fichier introuvable, CSV mal formé...), le `except Exception as e` attrape l'erreur, affiche un message clair et arrête le programme (`sys.exit(1)`) plutôt que de laisser Python afficher une trace d'erreur brute et illisible.
 
*Différence avec `describe.py`* : ce fichier ne vérifie pas explicitement l'existence du fichier avant de l'ouvrir (`Path.exists()`) — il laisse `pandas` échouer et rattrape l'erreur après coup. Les deux approches sont valables, juste un style différent.
 
---
 
## Étape 3 — `plot_histogram(df)`
 
La fonction qui fait tout le travail de tracé. Appelée une seule fois depuis le `main`, avec le DataFrame déjà chargé.
 
### 3.1 — Sélection des colonnes à tracer
 
```python
house_col = "Hogwarts House"
course_cols = [col for col in df.columns if col not in [house_col, "Index", "First Name", "Last Name", "Birthday", "Best Hand"]]
```
 
Garde uniquement les colonnes de **matières** (13 au total), en excluant explicitement toutes les colonnes qui ne sont pas des notes (identité de l'élève, maison). Contrairement à `describe.py` qui détecte automatiquement les colonnes numériques, ici c'est une liste d'exclusion écrite en dur — fonctionne très bien tant que la structure du CSV ne change pas.
 
### 3.2 — Préparation des maisons et des couleurs
 
```python
houses = df[house_col].unique()
colors = ['red', 'green', 'blue', 'yellow']
```
 
Récupère les 4 maisons présentes dans le fichier (dans leur ordre de première apparition dans le CSV — c'est pour ça que la légende est toujours dans le même ordre à chaque exécution, tant qu'on utilise le même dataset). Une couleur est associée à chaque maison via son index dans la boucle (voir étape 3.4).
 
### 3.3 — Création de la grille de sous-graphes
 
```python
fig, axes = plt.subplots(4, 4, figsize=(15, 10))
axes = axes.flatten()
```
 
Crée une grille de 4×4 = 16 emplacements (alors qu'il n'y a que 13 matières à tracer — les 3 cases en trop sont masquées à l'étape 3.5). `flatten()` transforme la grille 2D (4 lignes × 4 colonnes) en une simple liste de 16 axes, pour pouvoir les indexer un par un avec `axes[i]` dans la boucle suivante.
 
### 3.4 — La double boucle : une matière, puis une maison à la fois
 
```python
for i, course in enumerate(course_cols):
    ax = axes[i]
    for j, house in enumerate(houses):
        if pd.isna(house):
            continue
        house_data = df[df[house_col] == house][course].dropna()
        ax.hist(house_data, bins=20, alpha=0.6, label=house, color=colors[j % len(colors)])
    ax.set_title(f'{course}')
    ...
```
 
Pour chaque matière (`course_cols`), on choisit le bon sous-graphe (`axes[i]`), puis on trace **4 histogrammes superposés**, un par maison :
1. Si la maison elle-même est `NaN` (un élève sans maison assignée dans le fichier), on l'ignore.
2. On filtre le DataFrame pour ne garder que les lignes de cette maison, on isole la colonne de la matière en cours, et on retire les notes manquantes (`.dropna()`).
3. `ax.hist(...)` trace l'histogramme, avec `alpha=0.6` (semi-transparence) pour que les 4 histogrammes superposés restent tous visibles même là où ils se chevauchent — c'est ce qui permet de voir d'un coup d'œil si les maisons se distinguent ou se confondent sur une matière donnée.
4. `colors[j % len(colors)]` assigne une couleur fixe par maison, cohérente sur tous les sous-graphes.
### 3.5 — Masquer les cases inutilisées
 
```python
for idx in range(len(course_cols), len(axes)):
    axes[idx].set_visible(False)
```
 
Comme il y a 13 matières pour 16 emplacements, les 3 derniers axes (indices 13, 14, 15) sont rendus invisibles plutôt que laissés vides et visibles — c'est ce qu'on voit dans ta capture d'écran : la grille s'arrête proprement après "Flying", sans cases blanches parasites.
 
### 3.6 — Affichage final
 
```python
plt.tight_layout()
plt.suptitle('Distribution des notes par cours et par maison', y=1.02)
plt.show()
```
 
`tight_layout()` ajuste automatiquement l'espacement pour éviter que les titres/légendes se chevauchent entre sous-graphes. `suptitle` ajoute un titre général au-dessus de toute la grille. `plt.show()` ouvre la fenêtre (c'est une opération bloquante : le script attend que tu fermes la fenêtre avant de rendre la main au terminal).
 
---
 
## Comment lire le résultat pour répondre à la question du sujet
 
La question posée est : *quelle matière a une distribution homogène entre les 4 maisons ?* Une matière "homogène" = les 4 histogrammes colorés se superposent presque parfaitement (les 4 maisons ont la même distribution de notes, impossible de les distinguer). Une matière "discriminante" = les 4 histogrammes sont nettement séparés (facile de deviner la maison rien qu'avec cette note).
 
En regardant ta capture : **Arithmancy** et **Care of Magical Creatures** sont les deux matières où les 4 couleurs se superposent le plus largement sur toute la plage de notes — ce sont tes meilleures réponses à la question du sujet. À l'inverse, des matières comme **Astronomy**, **Herbology**, **Defense Against the Dark Arts** ou **Charms** montrent des groupes de couleurs nettement séparés : elles sont très discriminantes, donc **pas** homogènes.
 
---
 
## Résumé du déroulé complet
 
```
main
 ├─ load_data(filename)          → charge le CSV, ou arrete proprement si erreur
 └─ plot_histogram(df)
     ├─ selection des 13 colonnes de matieres
     ├─ recuperation des 4 maisons + couleurs associees
     ├─ creation de la grille 4x4 (16 emplacements)
     ├─ boucle : pour chaque matiere -> pour chaque maison -> trace un histogramme
     ├─ masque les 3 emplacements inutilises (16 - 13)
     └─ affiche la figure (plt.show)
```
-e 
 
---
 
 
# `scatter_plot.py` — memento dans l'ordre d'exécution
 
**Rôle du fichier** : afficher les paires de matières les plus corrélées entre elles, pour répondre à la question du sujet : *quelles sont les deux features qui sont similaires ?* Usage : `python3 scatter_plot.py dataset_train.csv`.
 
**Verdict de la relecture** : aucun bug, déjà corrigé lors d'une passe précédente (le tri par corrélation, qui garantit que la bonne réponse apparaît toujours en premier peu importe l'ordre des colonnes du CSV). Capture d'écran vérifiée : Astronomy vs Defense Against the Dark Arts apparaît bien en première position avec Corr: -1.000.
 
---
 
## Étape 1 — Point d'entrée : `if __name__ == "__main__":`
 
1. Vérifie qu'il y a exactement 2 arguments (script + fichier CSV).
2. Charge le CSV avec `pd.read_csv(filename)`.
3. Appelle `plot_scatter(df)` — étape 2 ci-dessous.
---
 
## Étape 2 — `plot_scatter(df)`
 
### 2.1 — Sélection des colonnes numériques
 
```python
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Index' in numeric_cols:
    numeric_cols.remove('Index')
n_features = len(numeric_cols)
```
 
`select_dtypes(include=[np.number])` demande à pandas de ne garder que les colonnes dont le type de données est numérique (int ou float) — méthode différente de `histogram.py` qui excluait les colonnes par leur nom. Retire "Index" (numérique mais sans intérêt statistique), puis `n_features` = 13.
 
### 2.2 — Calcul de TOUTES les corrélations possibles (la boucle clé)
 
```python
pairs = []
for i in range(n_features):
    for j in range(i + 1, n_features):
        data_clean = df[[numeric_cols[i], numeric_cols[j]]].dropna()
        corr = data_clean[numeric_cols[i]].corr(data_clean[numeric_cols[j]])
        pairs.append((abs(corr), corr, numeric_cols[i], numeric_cols[j]))
```
 
C'est la boucle centrale du fichier, à savoir expliquer précisément :
 
- `for i in range(n_features):` → `i` parcourt les indices 0 à 12 (chaque matière, une par une).
- `for j in range(i + 1, n_features):` → **point technique important** : `j` ne repart pas de 0, il repart de `i + 1`. Ça évite deux problèmes à la fois :
  1. **Les paires en double** : sans ce `+1`, on calculerait à la fois "Astronomy vs Herbology" (i=1, j=2) ET "Herbology vs Astronomy" (i=2, j=1) — la même information, deux fois. En forçant `j > i`, chaque paire n'est calculée qu'une seule fois.
  2. **Les paires d'une matière avec elle-même** : sans le `+1` (donc avec `range(i, n_features)`), on calculerait aussi "Astronomy vs Astronomy" (i=1, j=1), une corrélation toujours égale à 1.0, sans intérêt.
  
  Avec 13 matières, ce calcul produit exactement **C(13,2) = 78 paires uniques**.
- `data_clean = df[[numeric_cols[i], numeric_cols[j]]].dropna()` : isole les deux colonnes concernées, et retire les lignes où l'une des deux notes est manquante — nécessaire, sinon `pandas` ne pourrait pas calculer une corrélation avec des `NaN` au milieu.
- `data_clean[numeric_cols[i]].corr(data_clean[numeric_cols[j]])` : `.corr()` est une méthode pandas qui calcule le **coefficient de corrélation de Pearson** entre deux séries — une fonction "toute faite", mais autorisée ici (voir la discussion sur seaborn : ce n'est pas une fonction de la famille `describe()`/`mean()`/`std()` visée par l'interdiction du sujet, section V.1 uniquement).
- `pairs.append((abs(corr), corr, numeric_cols[i], numeric_cols[j]))` : on stocke un **tuple à 4 éléments** pour chaque paire : la valeur absolue de la corrélation (pour le tri, étape 2.3), la vraie valeur signée (pour l'affichage, qui doit montrer -1.000 et pas 1.000), et les deux noms de colonnes.
### 2.3 — Le tri par force de corrélation
 
```python
pairs.sort(key=lambda p: p[0], reverse=True)
```
 
Trie la liste `pairs` en utilisant comme critère `p[0]` — c'est-à-dire le **premier élément** de chaque tuple, la valeur absolue de la corrélation qu'on avait calculée juste avant. `reverse=True` trie du plus grand au plus petit. C'est ce tri qui garantit que les paires les plus fortement corrélées (positivement ou négativement) se retrouvent en tête de liste — peu importe l'ordre dans lequel les colonnes apparaissaient dans le CSV d'origine.
 
*Point corrigé dans une version précédente* : sans ce tri, le code affichait simplement les 20 premières paires rencontrées dans l'ordre naturel de la double boucle (`i`, `j` croissants), ce qui pouvait, sur un autre dataset ou un CSV aux colonnes réordonnées, cacher la vraie paire la plus corrélée hors des 20 cases disponibles.
 
### 2.4 — Création de la grille et affichage des 20 meilleures paires
 
```python
fig, axes = plt.subplots(4, 5, figsize=(18, 12))
axes = axes.flatten()
 
for plot_idx, (_, corr, col_i, col_j) in enumerate(pairs[:len(axes)]):
```
 
Grille de 4×5 = 20 emplacements. `pairs[:len(axes)]` prend les 20 premières paires de la liste **déjà triée** (donc les 20 plus fortement corrélées sur les 78 possibles). Le `enumerate(...)` fournit à la fois `plot_idx` (0 à 19, la position dans la grille) et le contenu du tuple déballé directement : `_` (la valeur absolue, ignorée ici — on ne s'en servait que pour trier), `corr`, `col_i`, `col_j`.
 
```python
    ax = axes[plot_idx]
    data_clean = df[[col_i, col_j]].dropna()
    ax.scatter(data_clean[col_i], data_clean[col_j], alpha=0.6)
    ax.set_xlabel(col_i)
    ax.set_ylabel(col_j)
    ax.set_title(f'{col_i} vs {col_j}')
    ax.grid(True, alpha=0.3)
```
 
Pour chaque paire retenue : nouveau nettoyage des `NaN` (recalculé ici, séparément de celui de l'étape 2.2 — un peu redondant en performance, mais sans impact fonctionnel), puis un scatter plot classique avec `col_i` en abscisse et `col_j` en ordonnée — cohérent avec le titre et les labels des axes, qui utilisent le même ordre.
 
```python
    ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
```
 
Affiche la valeur de corrélation **directement sur le graphe**, dans un petit encadré. `transform=ax.transAxes` est le détail technique important : ça dit à matplotlib que les coordonnées `(0.05, 0.95)` ne sont **pas** en unités de données (notes, etc.) mais en **coordonnées relatives au sous-graphe**, où `(0,0)` est le coin bas-gauche et `(1,1)` le coin haut-droit, peu importe l'échelle des données affichées. `(0.05, 0.95)` place donc systématiquement le texte près du coin supérieur gauche de chaque case, quelle que soit la plage de valeurs de la matière concernée.
 
### 2.5 — Masquer les cases inutilisées
 
```python
for idx in range(len(pairs), len(axes)):
    axes[idx].set_visible(False)
```
 
Ici, `len(pairs)` = 78 (toutes les paires), et `len(axes)` = 20 (la grille). Comme 78 > 20, cette boucle `range(78, 20)` est en réalité **vide** (un `range` où le début dépasse la fin ne produit aucune itération) — cette ligne ne fait donc jamais rien avec ce dataset précis. Elle reste utile comme sécurité si jamais le dataset avait moins de matières (donc moins de 20 paires possibles au total), auquel cas certaines cases resteraient à masquer.
 
---
 
## Résumé du déroulé complet
 
```
main
 └─ plot_scatter(df)
     ├─ selection des 13 colonnes numeriques
     ├─ calcul des 78 correlations possibles (boucle i, j=i+1..12)
     ├─ tri des 78 paires par |correlation| decroissante
     ├─ grille 4x5 (20 emplacements)
     ├─ affichage des 20 paires les plus correlees, avec Corr: x.xxx sur chaque case
     └─ masque les cases inutilisees (sans effet ici, car 78 > 20)
```
-e 
 
---
 
 
# `pair_plot.py` — memento dans l'ordre d'exécution
 
**Rôle du fichier** : afficher une matrice de nuages de points croisant toutes les paires de matières, colorée par maison, pour répondre à la question du sujet : *quelles features vas-tu utiliser pour ta régression logistique ?* Usage : `python3 pair_plot.py dataset_train.csv`.
 
**Verdict de la relecture** : le code est correct et déjà corrigé (détection seaborn sécurisée, fallback lisible). Un seul point cosmétique : une ligne commentée `#HAS_SEABORN = False` à retirer avant le rendu, c'est un reste de test.
 
---
 
## Étape 1 — Le chargement de seaborn (avant même le `main`)
 
```python
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
```
 
C'est la toute première chose exécutée quand Python lit le fichier, avant même d'atteindre le `if __name__ == "__main__":`. `seaborn` est une librairie externe, pas installée par défaut sur toutes les machines (notamment la machine de correction). Le `try/except` vérifie sa disponibilité une seule fois, et stocke le résultat dans `HAS_SEABORN` (`True` ou `False`), consulté plus tard à l'étape 3.
 
*Point corrigé* : dans une version précédente, l'import n'était pas protégé — si seaborn manquait, tout le script plantait immédiatement. Désormais, l'absence de seaborn ne fait jamais planter le script : il bascule automatiquement sur un mode de secours (voir étape 3.2).
 
---
 
## Étape 2 — Point d'entrée : `if __name__ == "__main__":`
 
1. Vérifie qu'il y a exactement 2 arguments (script + fichier CSV).
2. Charge le CSV avec `pd.read_csv(filename)`.
3. Appelle `plot_pairplot(df)` — étape 3 ci-dessous — avec le DataFrame chargé.
---
 
## Étape 3 — `plot_pairplot(df)`
 
### 3.1 — Préparation des données (commune aux deux modes)
 
```python
house_col = "Hogwarts House"
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Index' in numeric_cols:
    numeric_cols.remove('Index')
plot_data = df[numeric_cols + [house_col]].copy()
plot_data = plot_data.dropna()
```
 
Sélectionne automatiquement toutes les colonnes numériques (donc les 13 matières), retire "Index" si présente, garde en plus la colonne "Hogwarts House" pour pouvoir colorer par maison. `dropna()` retire toute ligne avec au moins une valeur manquante — contrairement à `describe.py` ou `histogram.py`, qui géraient les `NaN` colonne par colonne, ici on retire **toute la ligne** de l'élève dès qu'une seule matière lui manque. C'est plus strict, mais nécessaire : un pair plot doit comparer les mêmes élèves sur toutes les paires de matières à la fois.
 
### 3.2 — Aiguillage : seaborn ou fallback fait main ?
 
```python
if HAS_SEABORN:
    sns.pairplot(plot_data, hue=house_col, diag_kind='hist', plot_kws={'alpha':0.6})
    ...
else:
    ... (grille manuelle, voir 3.3)
```
 
C'est ici que la variable `HAS_SEABORN` (calculée à l'étape 1) est consultée. **C'est exactement le chemin que ta capture d'écran a emprunté** : `sns.pairplot(...)` fait tout le travail en une seule ligne — trace automatiquement une grille complète (13×13 pour 13 matières), avec un histogramme sur la diagonale et un nuage de points partout ailleurs, colorés par maison via `hue=house_col`.
 
Juste avant `plt.show()`, le script sauvegarde aussi la figure dans un fichier (`plt.savefig('pair_plot.png', dpi=150)`), avec un message qui l'indique. C'est utile car la fenêtre affichée à l'écran est souvent trop petite pour lire les noms des matières sur une grille 13×13 — le fichier PNG, en haute résolution, permet de zoomer dedans confortablement après coup.
 
### 3.3 — Le mode de secours (si seaborn manque) : décryptage ligne par ligne
 
Non emprunté dans ta capture (tu as seaborn), mais essentiel à comprendre si on te demande d'expliquer comment on reconstruit un pair plot "à la main". Voici chaque ligne, avec la signification précise de chaque élément.
 
```python
n_features = len(numeric_cols)
```
`n_features` = 13 (le nombre de matières). C'est à la fois le nombre de lignes **et** de colonnes de la grille qu'on va construire : un pair plot croise toujours une liste de variables avec elle-même.
 
```python
houses = plot_data[house_col].unique()
colors = ['red', 'green', 'blue', 'orange']
```
`houses` = les 4 noms de maison, dans leur ordre de première apparition dans le CSV. `colors` = une couleur fixe assignée à chaque maison (voir plus bas comment l'association se fait).
 
```python
fig, axes = plt.subplots(n_features, n_features, figsize=(28, 28))
```
Crée une grille de sous-graphes **13 lignes × 13 colonnes**. Contrairement à `histogram.py` qui faisait `.flatten()` pour obtenir une liste simple, ici `axes` reste un tableau 2D : `axes[i, j]` donne directement le sous-graphe à la ligne `i`, colonne `j`.
 
```python
for i in range(n_features):
    for j in range(n_features):
        ax = axes[i, j]
```
La boucle imbriquée : `i` parcourt les lignes (0 à 12), `j` parcourt les colonnes (0 à 12), donc `i` et `j` prennent ensemble les 169 combinaisons possibles (13×13). **Convention essentielle à retenir** : dans tout le reste du bloc, `i` désigne toujours la matière affichée **en ordonnée (axe Y)**, et `j` la matière affichée **en abscisse (axe X)** — c'est la convention standard d'un pair plot. `ax` est le sous-graphe précis correspondant à cette case (`i`, `j`) de la grille.
 
```python
        if i == j:
```
Teste si on est sur la **diagonale** de la grille : la ligne `i` (matière en Y) et la colonne `j` (matière en X) désignent la **même matière**. C'est le cas où un scatter plot n'aurait aucun sens (voir la FAQ plus haut : tous les points tomberaient sur une droite y=x sans information).
 
```python
            for k, house in enumerate(houses):
                house_data = plot_data[plot_data[house_col] == house][numeric_cols[i]]
                ax.hist(house_data, bins=15, alpha=0.5, color=colors[k % len(colors)])
```
Sur la diagonale, on trace un **histogramme empilé par maison** au lieu d'un scatter :
- `enumerate(houses)` donne à la fois l'indice `k` (0, 1, 2, 3) et le nom `house` de chaque maison, un par un.
- `plot_data[house_col] == house` est un filtre booléen : garde uniquement les lignes de cette maison précise.
- `[numeric_cols[i]]` sélectionne ensuite, sur ces lignes filtrées, la seule colonne qui nous intéresse ici : la matière numéro `i` (celle de la diagonale actuelle).
- `ax.hist(...)` dessine l'histogramme de cette sous-population sur le sous-graphe courant. `bins=15` = 15 barres. `alpha=0.5` = semi-transparence, pour voir les 4 histogrammes superposés sans que l'un cache totalement les autres.
- `colors[k % len(colors)]` : prend la couleur d'indice `k` dans la liste `colors`. Le `% len(colors)` (modulo) est une sécurité — si un jour il y avait plus de maisons que de couleurs définies, ça reboucle au début de la liste au lieu de planter avec une erreur d'index hors limites. Ici, avec exactement 4 maisons et 4 couleurs, le modulo ne change jamais rien en pratique (`k` reste toujours < 4).
```python
        else:
            for k, house in enumerate(houses):
                house_data = plot_data[plot_data[house_col] == house]
                ax.scatter(house_data[numeric_cols[j]], house_data[numeric_cols[i]],
                           alpha=0.4, s=3, color=colors[k % len(colors)])
```
Hors diagonale, un vrai nuage de points croisant deux matières différentes :
- Même filtre par maison qu'au-dessus, mais cette fois `house_data` garde **toutes les colonnes** (pas de `[numeric_cols[i]]` immédiat), parce qu'on a besoin à la fois de la matière X et de la matière Y pour chaque élève.
- `ax.scatter(x, y, ...)` : le **premier argument est l'axe X**, le **second l'axe Y**. On lui donne `house_data[numeric_cols[j]]` en X (la matière de la **colonne** `j`) et `house_data[numeric_cols[i]]` en Y (la matière de la **ligne** `i`) — exactement la convention posée plus haut.
- `s=3` : taille des marqueurs en points², volontairement minuscule (la valeur par défaut de matplotlib est 36) pour que ~1600 points superposés restent lisibles au lieu de former une tache pleine — c'est le fix qu'on avait fait ensemble.
- `alpha=0.4` : transparence, pour voir la densité des points là où plusieurs maisons se chevauchent.
```python
        ax.set_xticks([])
        ax.set_yticks([])
```
Supprime les graduations numériques sur les deux axes de **chaque** sous-graphe (exécuté pour les 169 cases, diagonale comprise). Sans ça, avec des sous-graphes aussi petits, les chiffres des graduations se chevaucheraient et deviendraient illisibles.
 
```python
        if i == n_features - 1:
            ax.set_xlabel(numeric_cols[j], rotation=45, ha='right', fontsize=8)
```
N'affiche le nom de la matière en abscisse **que sur la dernière ligne** de la grille (`i == 12`, la ligne du bas) — pas besoin de répéter le même nom de colonne 13 fois de haut en bas, une seule fois suffit, tout en bas. `rotation=45` incline le texte en diagonale (pour que les noms longs ne se chevauchent pas horizontalement), `ha='right'` ("horizontal alignment") ancre l'extrémité droite du texte contre la graduation, ce qui donne un rendu propre avec du texte incliné.
 
```python
        if j == 0:
            ax.set_ylabel(numeric_cols[i], rotation=0, ha='right', fontsize=8)
```
Même logique, mais pour l'axe Y : n'affiche le nom de la matière **que sur la première colonne** (`j == 0`, la colonne de gauche). `rotation=0` garde le texte à l'horizontale (plus lisible pour un label vertical de ce type), `ha='right'` colle le texte contre l'axe.
 
```python
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[k % len(colors)],
           markersize=8, label=house) for k, house in enumerate(houses)]
fig.legend(handles=handles, loc='upper right', fontsize=12)
```
Une fois les 169 sous-graphes tracés, on construit une **légende globale unique** pour toute la figure (plutôt que 169 légendes répétées, une par sous-graphe, ce qui serait illisible) :
- `plt.Line2D([0],[0], ...)` crée un **faux petit segment de ligne invisible**, uniquement utilisé comme "modèle" pour générer une pastille de légende — `color='w'` (blanc, donc la ligne elle-même est invisible), `marker='o'` (un rond), `markerfacecolor=colors[k % len(colors)]` (rempli de la couleur de cette maison), `label=house` (le texte affiché à côté). C'est une astuce classique en matplotlib pour fabriquer une légende "à la main" quand les éléments réels du graphe (ici, des centaines de petits points epars) ne s'y prêtent pas directement.
- La liste en compréhension (`for k, house in enumerate(houses)`) répète cette construction une fois par maison → 4 pastilles de légende au total.
- `fig.legend(...)` (et non `ax.legend()`) place cette légende au niveau de la **figure entière**, pas d'un sous-graphe précis — `loc='upper right'` la positionne en haut à droite de toute la grille.
### 3.4 — Message d'aide, affiché dans tous les cas
 
```python
print("Analyse pour la régression logistique:")
print("Choisis les features avec:")
print("- Faible corrélation entre elles (évite la multicolinéarité)")
print("- Bonne séparation des classes dans le pair plot")
print("- Distributions différentes selon les maisons")
```
 
Affiché après la fermeture de la fenêtre graphique (`plt.show()` est bloquant, donc ces lignes s'exécutent seulement une fois que tu as fermé le graphe). Ce n'est pas une analyse automatique — juste un rappel textuel des critères à appliquer toi-même en regardant le graphe.
 
---
 
## Comment lire le pair plot (questions fréquentes)
 
**Pourquoi une ligne droite parfaite sur certaines cases ?**
 
Une case du pair plot met en scatter plot deux matières l'une contre l'autre : la position horizontale d'un point est la note de la matière en colonne, la position verticale la note de la matière en ligne. Si les deux notes n'ont aucun lien, les points se dispersent en un nuage. Mais si les deux notes sont mathématiquement liées (par exemple si l'une est une transformation linéaire de l'autre), chaque élève se retrouve exactement sur une droite, sans exception, puisque connaître l'une des deux notes suffit à calculer l'autre. C'est le cas d'Astronomy et Defense Against the Dark Arts : une corrélation de -1 ou +1 (mesurée dans `scatter_plot.py`) se traduit toujours visuellement par une droite parfaite dans le pair plot.
 
**Pourquoi des histogrammes sur la diagonale, et pas des scatter plots ?**
 
Sur la diagonale, une case mettrait une matière contre elle-même (ex: Astronomy vs Astronomy) — tous les points tomberaient forcément sur une droite y=x parfaite, sans aucune information utile. Seaborn remplace donc automatiquement ces cases par un histogramme de la matière seule (coloré par maison), qui montre sa distribution — comme `histogram.py`, mais condensé dans la grille. C'est systématique : la diagonale d'un pair plot est toujours faite d'histogrammes, quel que soit le dataset.
 
**Pourquoi les noms des matières ne sont pas toujours visibles ?**
 
C'est un problème d'échelle d'affichage, pas un bug du code. Avec 13 matières, `sns.pairplot()` crée une grille de 13×13 = 169 sous-graphes dans une seule fenêtre. Les noms des matières ne sont affichés qu'une seule fois chacun, tout en bas et tout à gauche de la grille entière — pas sur chaque case. Si la fenêtre affichée ou la capture d'écran ne montre qu'une partie de la grille (par exemple seulement les premières lignes), les noms n'apparaissent simplement pas encore, plus bas ou plus à droite dans la figure complète.
 
Pour voir les noms sans ambiguïté, deux options : maximiser la fenêtre et naviguer jusqu'aux bords de la grille, ou sauvegarder la figure dans un fichier avant `plt.show()` (ex: `plt.savefig("pairplot.png", dpi=150)`) pour l'ouvrir ensuite et zoomer tranquillement dedans.
 
---
 
## Résumé du déroulé complet
 
```
[chargement du fichier] try/except import seaborn -> HAS_SEABORN
main
 └─ plot_pairplot(df)
     ├─ selection des colonnes numeriques + maison, dropna()
     ├─ si HAS_SEABORN : sns.pairplot() en une ligne  <- ta capture
     ├─ sinon : grille manuelle matplotlib (double boucle)
     └─ affiche les 3 conseils de choix de features
```
-e 
 
---
 
 
# `logreg_train.py` — memento dans l'ordre d'exécution
 
**Rôle du fichier** : entraîner 4 modèles de régression logistique (un par maison, stratégie one-vs-all) par descente de gradient, et sauvegarder leurs poids dans `thetas.csv`. Usage : `python3 logreg_train.py dataset_train.csv`.
 
**Verdict de la relecture** : le fichier utilise bien les **13 features** (toutes les matières numériques), version corrigée après qu'on ait mesuré que 2 features seulement (Astronomy, Herbology) plafonnaient l'accuracy à ~94.69%, sous le seuil des 98% exigé par le sujet. Vérifie que c'est bien cette version-là qui est dans ton repo avant la correction.
 
---
 
## Étape 1 — Point d'entrée : `if __name__ == "__main__":` puis `main()`
 
```python
if __name__ == "__main__":
    main()
```
 
Ne fait qu'appeler `main()`, qui contient tout le programme (contrairement à `describe.py` où le `main` faisait le travail directement sans passer par une fonction séparée).
 
### 1.1 — Vérification des arguments
 
```python
if len(sys.argv) < 2:
    print("Erreur: mauvais nombre d'argument")
    sys.exit(1)
```
 
Vérifie qu'un fichier CSV a bien été donné en argument (`python3 logreg_train.py dataset_train.csv`).
 
### 1.2 — Chargement des données et sélection des features
 
```python
data = pd.read_csv(sys.argv[1])
 
features = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts",
            "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
            "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]
```
 
`features` est la liste des 13 colonnes utilisées pour entraîner le modèle — c'est ta réponse concrète à la question posée par `pair_plot.py` ("quelles features vas-tu utiliser ?"). Toutes les matières sont gardées ici, un choix validé empiriquement (moyenne de 98.41% d'accuracy sur 10 découpages aléatoires différents du dataset).
 
```python
X = data[features].copy().fillna(data[features].mean())
X = np.array(X)
```
 
Isole ces 13 colonnes dans `X`. `.fillna(data[features].mean())` remplace chaque valeur manquante par la **moyenne de sa propre colonne** (et non une valeur arbitraire comme 0, qui fausserait le modèle) — c'est une stratégie d'imputation simple mais raisonnable. Conversion finale en tableau numpy pour les calculs matriciels qui suivent.
 
---
 
## Étape 2 — Standardisation : `scaler.standardize(X)`
 
```python
scaler = MyLR(thetas=np.zeros((X.shape[1] + 1, 1)))
X_norm = scaler.standardize(X)
mean_train = scaler.mean_
std_train = scaler.std_
```
 
Crée une instance de `MyLogisticRegression` (voir le fichier `my_logistic_regression.py`, détaillé séparément) uniquement pour utiliser sa méthode `standardize` — les `thetas` initialisés ici à zéro ne servent à rien pour cette étape précise, c'est juste un prérequis technique pour instancier la classe.
 
`scaler.standardize(X)` calcule la moyenne et l'écart-type de chaque colonne de `X`, les **stocke** dans `scaler.mean_` et `scaler.std_` (puisqu'appelée sans arguments `mean`/`std`, voir le memento de `my_logistic_regression.py`), et renvoie `X_norm` : les données centrées-réduites, prêtes pour la descente de gradient.
 
`mean_train` et `std_train` sont récupérés ici pour être **sauvegardés plus tard** (étape 4) — c'est ce qui permettra à `logreg_predict.py` de standardiser le jeu de test exactement de la même façon, sans recalculer ses propres statistiques (le bug qu'on avait corrigé ensemble).
 
---
 
## Étape 3 — La boucle one-vs-all : un modèle par maison
 
```python
houses = data["Hogwarts House"].unique()
all_thetas = {}
 
for house in houses:
    Y_binary = (data["Hogwarts House"] == house).astype(int).values.reshape(-1, 1)
    thetas_init = np.zeros((X_norm.shape[1]+1, 1))
    model = MyLR(thetas=thetas_init, alpha=0.01, max_iter=50000)
    trained_thetas = model.fit_(X_norm, Y_binary)
    all_thetas[house] = trained_thetas.flatten()
```
 
Le cœur de la stratégie one-vs-all, ligne par ligne :
 
- `houses = data["Hogwarts House"].unique()` : les 4 noms de maison, dans leur ordre de première apparition dans le CSV.
- `for house in houses:` : boucle 4 fois, une fois par maison — chaque tour entraîne un modèle **complètement indépendant**.
- `Y_binary = (data["Hogwarts House"] == house).astype(int)...` : transforme la colonne "Hogwarts House" (4 catégories textuelles) en une colonne de **0 et 1** : `1` si l'élève appartient à cette maison précise, `0` sinon. C'est la transformation qui fait passer d'un problème "4 catégories" à 4 problèmes "oui/non" séparés. `.reshape(-1, 1)` force le tableau en colonne verticale (forme `(m, 1)`), le format attendu par `fit_`.
- `thetas_init = np.zeros((X_norm.shape[1]+1, 1))` : initialise les thetas à zéro. `X_norm.shape[1]` = 13 (le nombre de features), `+1` pour le biais (θ0) — donc un vecteur de 14 zéros, un par paramètre à apprendre.
- `model = MyLR(thetas=thetas_init, alpha=0.01, max_iter=50000)` : crée **un nouveau modèle à chaque tour de boucle** (pas de réutilisation entre maisons — chacune a ses propres thetas, indépendants des 3 autres).
- `trained_thetas = model.fit_(X_norm, Y_binary)` : lance la descente de gradient (détaillée dans le memento de `my_logistic_regression.py`) pour cette maison précise.
- `all_thetas[house] = trained_thetas.flatten()` : stocke le résultat dans un dictionnaire, indexé par nom de maison. `.flatten()` transforme la colonne verticale `(14, 1)` en un simple vecteur `(14,)`, plus pratique à manipuler ensuite.
À la fin de cette boucle, `all_thetas` contient 4 entrées, chacune avec 14 valeurs (1 biais + 13 coefficients de features).
 
---
 
## Étape 4 — Construction et sauvegarde de `thetas.csv`
 
```python
dict_thetas = {}
for house, thetas in all_thetas.items():
    for i, theta in enumerate(thetas):
        dict_thetas[f"{house}_theta_{i}"] = [float(theta)]
```
 
Double boucle : pour chaque maison (`all_thetas.items()`), puis pour chaque valeur de theta de cette maison (`enumerate(thetas)`, donnant l'indice `i` et la valeur), crée une entrée dans `dict_thetas` avec une clé du type `"Ravenclaw_theta_0"`, `"Ravenclaw_theta_1"`, etc. — une colonne par coefficient, par maison. Avec 4 maisons × 14 coefficients (θ0 à θ13), ça fait 56 colonnes de thetas au total.
 
```python
for i, feature in enumerate(features):
    dict_thetas[f"mean_{feature}"] = [float(mean_train[i])]
    dict_thetas[f"std_{feature}"] = [float(std_train[i])]
```
 
Ajoute ensuite, dans ce même dictionnaire, la moyenne et l'écart-type **de chaque feature individuelle** (13 × 2 = 26 colonnes supplémentaires : `mean_Arithmancy`, `std_Arithmancy`, `mean_Astronomy`, `std_Astronomy`, etc.) — les statistiques calculées à l'étape 2, essentielles pour que `logreg_predict.py` standardise le jeu de test de façon cohérente.
 
```python
df_thetas = pd.DataFrame(dict_thetas)
df_thetas.to_csv("thetas.csv", index=False)
print("Training finished. Thetas saved to 'thetas.csv'.")
```
 
Transforme le dictionnaire en DataFrame (une seule ligne, puisque chaque valeur est entourée de crochets `[...]` dans le dictionnaire — un DataFrame pandas attend des listes de valeurs, même pour une seule ligne), puis l'écrit dans `thetas.csv` (`index=False` évite d'ajouter une colonne d'index inutile en plus). Au total, le fichier contient 56 + 26 = **82 colonnes**, sur une seule ligne.
 
---
 
## Résumé du déroulé complet
 
```
main
 ├─ lecture du CSV + selection des 13 features
 ├─ standardisation (scaler.standardize) -> X_norm, mean_train, std_train retenus
 ├─ pour chaque maison (x4) :
 │   ├─ construction de Y_binary (0/1 pour cette maison)
 │   └─ model.fit_(X_norm, Y_binary) -> descente de gradient -> thetas de cette maison
 └─ construction de thetas.csv :
     ├─ 56 colonnes de thetas (4 maisons x 14 coefficients)
     └─ 26 colonnes de mean_/std_ (13 features x 2)
```
-e 
 
---
 
 
# `logreg_predict.py` — memento dans l'ordre d'exécution
 
**Rôle du fichier** : appliquer les 4 modèles déjà entraînés (sauvegardés dans `thetas.csv`) à de nouveaux élèves, et prédire leur maison. Usage : `python3 logreg_predict.py dataset_test.csv thetas.csv`.
 
**Verdict de la relecture** : code correct (13 features, standardisation avec les stats du train, format de sortie conforme au sujet). Point de vigilance : vérifier que le premier argument est bien `dataset_test.csv` et non `dataset_train.csv` (voir la note en tête de ce document).
 
---
 
## Étape 1 — Point d'entrée : `if __name__ == "__main__":` puis `main()`
 
### 1.1 — Vérification des arguments
 
```python
if len(sys.argv) < 3:
    print("Erreur: mauvais nombre d'argument")
    print("Usage: python3 logreg_predict.py <dataset_test.csv> <thetas.csv>")
    sys.exit(1)
```
 
Contrairement à `logreg_train.py` qui n'attend qu'un seul argument, celui-ci en attend **deux** : le dataset à prédire, et le fichier de poids. C'est la correction qu'on avait faite ensemble pour respecter le sujet à la lettre (*"it takes dataset_test.csv as a parameter and a file containing the weights"*).
 
### 1.2 — Chargement du fichier de poids
 
```python
file_theta = Path(sys.argv[2])
if not file_theta.exists():
    print("Erreur: pas de fichier thetas trouve")
    sys.exit(1)
 
thetas_df = pd.read_csv(sys.argv[2])
```
 
Vérifie que le fichier de poids (2e argument) existe avant de tenter de le lire, puis le charge dans `thetas_df` — le DataFrame à une seule ligne et 82 colonnes qu'on a examiné ensemble juste avant.
 
### 1.3 — Chargement et préparation du dataset à prédire
 
```python
features = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts",
            "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
            "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]
 
X_test = pd.read_csv(sys.argv[1])
X_features = X_test[features].copy().fillna(X_test[features].mean())
X = np.array(X_features)
```
 
**Point critique** : `features` doit être **rigoureusement identique**, dans le même ordre, à la liste utilisée dans `logreg_train.py` — sinon la colonne 3 du modèle (qui a appris "Herbology") se retrouverait appliquée à une colonne différente lors de la prédiction, faussant tout. `X_test = pd.read_csv(sys.argv[1])` charge le **premier** argument (le dataset à prédire) — c'est cette ligne qui doit recevoir `dataset_test.csv`, pas `dataset_train.csv`.
 
`.fillna(X_test[features].mean())` : remplace les valeurs manquantes par la moyenne **du jeu de test lui-même** (pas celle du train). Ce n'est pas incohérent avec la correction qu'on a faite pour la standardisation : combler un trou avec une estimation raisonnable est différent de la standardisation, qui elle doit absolument utiliser l'échelle apprise sur le train.
 
---
 
## Étape 2 — Standardisation avec les statistiques DU TRAIN
 
```python
mean_train = np.array([thetas_df[f"mean_{f}"].iloc[0] for f in features])
std_train = np.array([thetas_df[f"std_{f}"].iloc[0] for f in features])
 
scaler = MyLR(thetas=np.zeros((X.shape[1] + 1, 1)))
X_norm = scaler.standardize(X, mean=mean_train, std=std_train)
```
 
Ligne par ligne :
- La liste en compréhension parcourt `features` (dans le même ordre que `logreg_train.py`) et va chercher, pour chaque matière, les colonnes `mean_<matière>` et `std_<matière>` **déjà calculées et sauvegardées par `logreg_train.py`** — pas recalculées ici. `.iloc[0]` prend la valeur de l'unique ligne du DataFrame `thetas_df`.
- `scaler = MyLR(...)` : encore une instance "jetable", uniquement pour accéder à la méthode `standardize`.
- `scaler.standardize(X, mean=mean_train, std=std_train)` : **c'est ici que le bug de recalcul est évité**. En passant explicitement `mean` et `std`, la méthode (voir `my_logistic_regression.py`) n'en recalcule pas de nouveaux à partir de `X` — elle applique directement ceux fournis. C'est le point technique le plus important de tout le pipeline train/predict : la garantie que le jeu de test est standardisé exactement sur la même échelle que celle apprise pendant l'entraînement.
---
 
## Étape 3 — Reconstruction des 4 modèles à partir de `thetas.csv`
 
```python
houses = [col.split("_theta_0")[0] for col in thetas_df.columns if "_theta_0" in col]
probs = np.zeros((X_norm.shape[0], len(houses)))
```
 
`houses` : retrouve les 4 noms de maison **à partir des noms de colonnes** de `thetas_df` (et non depuis le dataset, contrairement à `logreg_train.py`). Pour chaque colonne qui contient `"_theta_0"` (donc `"Ravenclaw_theta_0"`, `"Slytherin_theta_0"`, etc. — une seule par maison, puisque chaque maison n'a qu'un seul θ0), on coupe la chaîne juste avant `"_theta_0"` pour ne garder que le nom de la maison. `probs` : un tableau vide, une ligne par élève, une colonne par maison — sera rempli à l'étape suivante avec les probabilités prédites.
 
```python
for j, house in enumerate(houses):
    theta_cols = [col for col in thetas_df.columns if col.startswith(f"{house}_theta")]
    theta_values = thetas_df[theta_cols].values.flatten().reshape(-1, 1)
 
    X_ = np.c_[np.ones((X_norm.shape[0], 1)), X_norm]
    y_hat = sigmoid(X_.dot(theta_values))
    probs[:, j] = y_hat.flatten()
```
 
Boucle sur les 4 maisons :
- `theta_cols` : sélectionne, parmi les 82 colonnes de `thetas_df`, seulement les 14 qui appartiennent à cette maison précise (`"Ravenclaw_theta_0"` à `"Ravenclaw_theta_13"`) — grâce au préfixe `f"{house}_theta"` (le `startswith` évite toute confusion avec les colonnes `mean_`/`std_`, qui ne commencent jamais par un nom de maison).
- `theta_values` : récupère ces 14 valeurs, les aplatit (`.flatten()`), puis les remet en colonne verticale (`.reshape(-1, 1)`) — le format `(14, 1)` attendu pour la multiplication matricielle qui suit.
- `X_ = np.c_[np.ones(...), X_norm]` : ajoute la colonne de biais à `X_norm`, exactement comme dans `fit_` — nécessaire pour que la multiplication matricielle avec les 14 thetas (1 biais + 13 features) soit dimensionnellement cohérente.
- `sigmoid(X_.dot(theta_values))` : calcule la probabilité prédite par **ce modèle spécifique** (cette maison) pour **tous les élèves en même temps** — la fonction `sigmoid` est redéfinie localement en haut du fichier (recopiée depuis la classe, plutôt que d'instancier `MyLR` juste pour ça).
- `probs[:, j] = ...` : stocke ce résultat dans la colonne `j` du tableau `probs` (une colonne par maison).
À la fin de cette boucle, `probs` est un tableau `(nombre d'élèves, 4)` : la probabilité que chaque élève appartienne à chacune des 4 maisons, selon chacun des 4 modèles interrogés indépendamment.
 
---
 
## Étape 4 — Le choix final : `argmax`
 
```python
predicted_indices = np.argmax(probs, axis=1)
predicted_houses = [houses[i] for i in predicted_indices]
```
 
`np.argmax(probs, axis=1)` : pour chaque élève (chaque **ligne** de `probs`, d'où `axis=1` pour chercher le maximum **le long des colonnes**, donc entre les 4 maisons), renvoie l'**indice** de la colonne où la probabilité est la plus élevée — pas la probabilité elle-même, sa position. `predicted_houses` traduit ensuite chaque indice en nom de maison via la liste `houses` construite à l'étape 3. C'est la mise en pratique concrète du "one-vs-all" : on demande aux 4 modèles indépendants leur avis, et on garde celui qui est le plus confiant.
 
---
 
## Étape 5 — Sauvegarde du résultat
 
```python
df_pred = pd.DataFrame({
    "Index": np.arange(len(predicted_houses)),
    "Hogwarts House": predicted_houses
})
df_pred.to_csv("houses.csv", index=False)
print("Predictions saved to 'houses.csv'.")
```
 
Construit le fichier final avec exactement les deux colonnes attendues par le sujet (`Index`, `Hogwarts House`). `np.arange(len(predicted_houses))` génère `0, 1, 2, ...` jusqu'au nombre d'élèves prédits — **cette ligne suppose que les élèves de `dataset_test.csv` sont numérotés consécutivement à partir de 0**, sans vérifier la vraie colonne "Index" du fichier d'entrée. Tant que le fichier de test respecte cette convention (ce qui est le cas ici), pas de souci — mais c'est une hypothèse implicite à connaître.
 
---
 
## Résumé du déroulé complet
 
```
main
 ├─ verification des 2 arguments (dataset + thetas)
 ├─ chargement de thetas.csv
 ├─ chargement du dataset a predire + selection des 13 features
 ├─ standardisation avec mean_train/std_train LUS DANS thetas.csv (pas recalcules)
 ├─ reconstruction des 4 modeles a partir des colonnes de thetas.csv
 ├─ pour chaque maison : calcul de la probabilite predite pour tous les eleves
 ├─ argmax : garde la maison la plus probable, eleve par eleve
 └─ sauvegarde de houses.csv (Index, Hogwarts House)
```
-e 
 
---
 
 
# `my_logistic_regression.py` — memento dans l'ordre d'utilisation
 
**Rôle du fichier** : la classe `MyLogisticRegression`, le "moteur mathématique" partagé par `logreg_train.py` et `logreg_predict.py`. Contrairement aux fichiers précédents, ce n'est pas un script avec un `main` — c'est une classe qu'on instancie et sollicite depuis l'extérieur. Ce document suit donc l'ordre dans lequel ses méthodes sont **réellement appelées** au fil du pipeline complet (entraînement puis prédiction), pas l'ordre d'écriture dans le fichier.
 
**Verdict de la relecture** : correct, déjà validé (formules de sigmoïde et de gradient conformes au sujet, testées).
 
---
 
## Étape 1 — `__init__` : à chaque création d'un modèle
 
```python
def __init__(self, thetas, alpha=0.001, max_iter=1000):
    self.alpha = alpha
    self.max_iter = max_iter
    if not isinstance(thetas, np.ndarray):
        self.thetas = np.array(thetas, dtype=float)
    else:
        self.thetas = thetas.astype(float)
```
 
Appelée à chaque fois qu'on écrit `MyLR(...)` — et ça arrive **6 fois** au total dans tout le pipeline : une fois comme "scaler" dans `logreg_train.py`, 4 fois (une par maison) toujours dans `logreg_train.py`, et une fois comme "scaler" dans `logreg_predict.py`.
 
- `self.alpha` et `self.max_iter` : stockés tels quels, avec des valeurs par défaut (`0.001` et `1000`) si non précisées à l'appel — dans la pratique, `logreg_train.py` les précise toujours explicitement (`alpha=0.01, max_iter=50000`).
- La vérification `isinstance(thetas, np.ndarray)` : accepte aussi bien une simple liste Python (`[0, 0, 0]`) qu'un tableau numpy déjà construit, et convertit systématiquement en tableau numpy de type `float` (`astype(float)`) — utile car `np.zeros(...)` renvoie déjà un tableau numpy, donc dans ce fichier c'est toujours la branche `else` qui s'exécute en pratique, mais la classe reste utilisable avec une simple liste si besoin.
---
 
## Étape 2 — `standardize` : premier appel, à l'entraînement (dans `logreg_train.py`)
 
```python
def standardize(self, X, mean=None, std=None):
    if not isinstance(X, np.ndarray):
        return None
    if X.size == 0:
        return None
 
    if mean is None or std is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
 
    self.mean_ = mean
    self.std_ = std
 
    new_X = (X - mean) / std
    return new_X
```
 
**Premier appel du pipeline complet**, fait par `logreg_train.py` juste après le chargement des données, sans préciser `mean`/`std` :
- Les deux `if` de garde renvoient `None` si `X` n'est pas un tableau numpy ou s'il est vide — sécurité basique.
- `if mean is None or std is None:` : comme aucun des deux n'est fourni à ce premier appel, cette condition est vraie, donc `mean` et `std` sont **calculés** à partir de `X` (`X.mean(axis=0)` et `X.std(axis=0)` calculent une moyenne/écart-type **par colonne**, grâce à `axis=0`).
- `self.mean_ = mean` / `self.std_ = std` : ces deux lignes **stockent** le résultat comme attributs de l'objet, même si l'appelant ne les redemande pas explicitement en retour — c'est ce qui permet à `logreg_train.py` de les récupérer juste après via `scaler.mean_` et `scaler.std_`, pour les sauvegarder dans `thetas.csv`.
- `new_X = (X - mean) / std` : la formule de standardisation elle-même, appliquée à toutes les colonnes en une seule opération grâce au broadcasting numpy.
---
 
## Étape 3 — `fit_` : appelée 4 fois, une par maison (dans `logreg_train.py`)
 
```python
def fit_(self, X, Y):
    m = len(Y)
    X_ = np.c_[np.ones((m, 1)), X]
 
    for _ in range(self.max_iter):
        y_hat = self.sigmoid(X_.dot(self.thetas))
        gradient = (1 / m) * X_.T.dot(y_hat - Y)
        self.thetas -= self.alpha * gradient
    return self.thetas
```
 
Appelée juste après la standardisation, une fois par maison, sur les données déjà standardisées (`X_norm`) :
- `X_ = np.c_[np.ones((m, 1)), X]` : ajoute la colonne de biais, comme dans `ft_linear_regression`.
- La boucle `for _ in range(self.max_iter)` : 50 000 itérations de descente de gradient (valeur passée par `logreg_train.py`), à chaque tour :
  - `y_hat = self.sigmoid(X_.dot(self.thetas))` : calcule la probabilité actuellement prédite pour tous les élèves — **c'est ici que `sigmoid` est appelée**, à chaque itération, 50 000 fois par maison (voir étape 4).
  - `gradient = (1/m) * X_.T.dot(y_hat - Y)` : le gradient de la log loss, dans sa forme vectorisée (voir le cours général pour la démonstration mathématique complète).
  - `self.thetas -= self.alpha * gradient` : mise à jour simultanée de tous les thetas.
- `return self.thetas` : renvoie le résultat final, récupéré par `logreg_train.py` dans `trained_thetas`.
---
 
## Étape 4 — `sigmoid` : la fonction la plus sollicitée de tout le fichier
 
```python
def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
```
 
Ne fait qu'une seule ligne, mais c'est la fonction **la plus appelée** de tout le pipeline : une fois par itération de `fit_`, donc 50 000 fois × 4 maisons = **200 000 appels** rien que pendant l'entraînement. `np.exp(-z)` calcule l'exponentielle de `-z` pour chaque élément du tableau `z` en une seule opération vectorisée (pas de boucle Python explicite), ce qui la rend rapide malgré le nombre d'appels.
 
*Remarque* : `logreg_predict.py` a sa **propre** fonction `sigmoid`, redéfinie localement en dehors de la classe (même formule, code dupliqué), plutôt que d'appeler `model.sigmoid(...)` sur une instance — un choix de style qui fonctionne, mais qui duplique légèrement le code.
 
---
 
## Étape 5 — `standardize` : deuxième appel, à la prédiction (dans `logreg_predict.py`)
 
Même méthode qu'à l'étape 2, mais cette fois **appelée avec `mean` et `std` fournis explicitement** :
 
```python
X_norm = scaler.standardize(X, mean=mean_train, std=std_train)
```
 
Cette fois, `mean is None or std is None` est **faux** (les deux sont fournis), donc le bloc de calcul est sauté entièrement — `mean` et `std` gardent les valeurs passées en argument (celles lues depuis `thetas.csv`, calculées par le train). C'est la même méthode, mais son comportement change complètement selon qu'on lui donne ou non ces deux arguments — exactement la mécanique qui a permis de corriger le bug de standardisation qu'on avait identifié ensemble.
 
---
 
## Résumé de l'ordre d'utilisation réel dans le pipeline complet
 
```
logreg_train.py
 ├─ MyLR(...)  [__init__]           -> instance "scaler"
 ├─ scaler.standardize(X)           -> calcule ET retient mean_/std_
 └─ pour chaque maison (x4) :
     ├─ MyLR(...)  [__init__]       -> nouvelle instance, un modele par maison
     └─ model.fit_(X_norm, Y_binary)
         └─ (x50000, appelle self.sigmoid(...) a chaque tour)
 
logreg_predict.py
 ├─ MyLR(...)  [__init__]           -> instance "scaler"
 ├─ scaler.standardize(X, mean=mean_train, std=std_train)  -> REUTILISE, ne recalcule pas
 └─ sigmoid(...) [fonction locale, PAS la methode de la classe]
     └─ argmax entre les 4 maisons
```
 
Le menu 1Password est disponible. Appuyez sur la flèche vers le bas pour sélectionner.