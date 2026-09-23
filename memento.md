# Memento notions — projet `dslr`

Ce document explique le projet comme une histoire : on part du sujet, et on suit le déroulé réel, fichier par fichier, en expliquant à chaque étape *pourquoi* on fait ce qu'on fait et *quelles notions* ça mobilise. Pour le détail ligne par ligne du code, va voir `code-memento.md` — ce document-ci y renvoie chaque fois que le code devient trop technique pour être résumé en quelques phrases.

---

## Introduction : le sujet, et ce qu'on doit faire

Le prétexte du sujet : le Choixpeau magique de Poudlard, qui répartit chaque élève dans l'une des 4 maisons (Gryffondor, Serpentard, Serdaigle, Poufsouffle) selon sa personnalité, est cassé. On nous demande de le recréer avec du Machine Learning : à partir des notes de chaque élève dans 13 matières (Astronomie, Potions, Botanique...), deviner dans quelle maison il aurait dû être réparti.

Concrètement, ça se découpe en deux grandes phases, dans cet ordre :

1. **Comprendre les données** avant de faire quoi que ce soit (analyse statistique + visualisation) — c'est la partie la moins spectaculaire, mais celle qui conditionne la réussite de la suite.
2. **Construire un modèle de classification** (la régression logistique) qui apprend, à partir des notes d'élèves déjà répartis (`dataset_train.csv`), à deviner la maison de nouveaux élèves (`dataset_test.csv`).

Le sujet impose une règle simple mais stricte pour la première partie : pas de fonction toute faite qui calcule les statistiques à ta place (`.mean()`, `.std()`, `.describe()`...) — il faut recoder ces calculs à la main, pour prouver qu'on comprend ce qu'il y a dedans, pas juste qu'on sait appeler une librairie.

---

## Partie 1 : le déroulé du projet, fichier par fichier

### Étape 1 — Se faire une idée des données brutes : `describe.py`

Avant même de penser à un modèle, la première chose à faire est de regarder ses données en face : combien de valeurs par colonne, quelles échelles, y a-t-il des trous ? C'est le rôle de `describe.py` (détail complet du code dans `code-memento.md`).

Il affiche, pour chacune des 13 matières, huit statistiques de base : **Count** (combien de notes valides), **Mean** (la moyenne, le centre de gravité de la distribution), **Std** (l'écart-type, à quel point les notes s'écartent en moyenne de la moyenne), **Min/Max**, et les **quartiles** (25%, 50%, 75% — les seuils qui découpent les élèves triés en 4 groupes égaux).

Un détail technique à connaître sur le Std : la formule divise par `n-1` et non `n` (la **correction de Bessel**). Intuition : la moyenne qu'on calcule est elle-même approximative, collée aux données qu'on a sous la main — diviser par `n` tendrait à légèrement sous-estimer la vraie dispersion. C'est la convention que `pandas.describe()` utilise par défaut, donc celle qu'on a adoptée pour que nos résultats matchent exactement les siens (vérifié, écart de 0.00005 seulement, de l'arrondi flottant).

On a ensuite ajouté 5 statistiques bonus, dans le même esprit : **Missing** (le nombre de notes manquantes), **Variance** (le Std au carré), **Range** (Max - Min), **IQR** (Q3 - Q1, une version du Range moins sensible aux valeurs extrêmes), et la plus intéressante, la **Skewness**.

**Comprendre la Skewness avec une image** : imagine l'histogramme des notes d'une matière. Si la forme ressemble à une cloche bien symétrique, la skewness vaut à peu près 0. Mais si la distribution a une "queue" qui s'étire beaucoup plus d'un côté que de l'autre (par exemple, la plupart des élèves ont de bonnes notes regroupées, mais quelques-uns décrochent avec des notes très basses qui traînent loin derrière), la skewness s'éloigne de 0. Le signe est piégeux : une skewness **négative** veut dire que la queue part vers la **gauche** (valeurs basses), même si la majorité des élèves est plutôt regroupée à droite. Techniquement, la formule élève les écarts à la moyenne **au cube** (pas au carré comme pour le Std) : élever un nombre négatif au carré le rend toujours positif, donc le Std ne dit jamais dans quel sens penche une distribution — mais élever au cube garde le signe, donc la skewness, elle, le peut.

### Étape 2 — Répondre à "quelle matière est homogène ?" : `histogram.py`

Une fois qu'on connaît la forme générale des données, la première vraie question du sujet arrive : *quelle matière a une distribution de notes homogène entre les 4 maisons ?*

Un histogramme découpe la plage de notes d'une matière en tranches, et compte combien d'élèves tombent dans chaque tranche — ça donne une photo de la forme de la distribution. En superposant les 4 histogrammes (un par maison, avec de la transparence pour voir les chevauchements), deux cas de figure apparaissent : soit les 4 couleurs se **confondent** presque parfaitement sur toute la plage de notes — la matière est "homogène", elle ne dit rien sur la maison de l'élève — soit les 4 couleurs forment des **groupes bien séparés** — la matière est "discriminante", très utile pour deviner la maison. Sur ce dataset, Arithmancy et Care of Magical Creatures sont les meilleures réponses à la question (couleurs très mélangées), alors qu'Astronomy ou Charms montrent des groupes nettement séparés.

### Étape 3 — Répondre à "quelles features sont similaires ?" : `scatter_plot.py`

Deuxième question du sujet : *quelles sont les deux matières similaires ?* "Similaire" ici veut dire : les notes des deux matières sont **corrélées**, elles bougent ensemble.

Un scatter plot place chaque élève comme un point, une matière en abscisse et une autre en ordonnée. Le nombre qui résume cette relation s'appelle la **corrélation de Pearson**, entre -1 et +1 : proche de +1, quand une note monte l'autre monte aussi (systématiquement) ; proche de -1, quand une note monte l'autre descend (systématiquement) ; proche de 0, aucun lien. Visuellement, une corrélation proche de ±1 se traduit par un nuage de points aligné en une droite quasi parfaite — c'est exactement ce qu'on trouve entre Astronomy et Defense Against the Dark Arts (corrélation de -1.000, une droite parfaite) : ce sont en réalité la même information, juste présentée sur une échelle différente. `scatter_plot.py` calcule la corrélation de toutes les paires possibles de matières et affiche les 20 plus fortes, triées — le calcul précis de cette boucle (pourquoi `j` repart de `i+1` et pas de 0) est détaillé dans `code-memento.md`.

### Étape 4 — Choisir ses features : `pair_plot.py`

Troisième question : *à partir de cette visualisation, quelles features vas-tu utiliser pour ta régression logistique ?*

Le pair plot est une grille qui croise **toutes les paires de matières entre elles** en une seule vue : chaque case hors diagonale est un scatter plot (comme à l'étape 3, mais pour toutes les paires à la fois), chaque case sur la diagonale est un histogramme (comparer une matière à elle-même en scatter donnerait toujours une droite parfaite sans intérêt, donc seaborn la remplace automatiquement par un histogramme). Colorer par maison permet de repérer d'un coup d'œil quelles paires de matières séparent bien les 4 groupes — les meilleures candidates pour le modèle. Construire ce genre de grille à la main (si seaborn n'est pas disponible) demande une double boucle un peu technique, entièrement décortiquée dans `code-memento.md`.

À l'issue de cette étape, on a choisi de garder **les 13 matières** plutôt qu'un sous-ensemble : empiriquement, ça donne une bien meilleure précision (98.41% de moyenne contre 94.69% avec seulement 2 features), ce qui compte plus ici que la parcimonie du modèle.

### Étape 5 — Entraîner le modèle : `logreg_train.py` (et `my_logistic_regression.py`)

On arrive au cœur du projet : reconstruire le Choixpeau. Ici, tout change de nature par rapport à un projet de régression classique (comme `ft_linear_regression`, qui prédisait un prix, un nombre continu) : on doit prédire une **catégorie** parmi 4 possibles. Une droite classique ne convient pas telle quelle, il faut un nouvel outil : la **régression logistique**.

**La sigmoïde**, le cœur de l'outil : `g(z) = 1 / (1 + e^(-z))`. Imagine un videur de boîte de nuit qui ne laisse jamais entrer plus de 100% ni moins de 0% des gens, peu importe à quel point la foule pousse — la sigmoïde fait pareil avec les nombres : elle écrase n'importe quelle valeur, même énorme ou très négative, dans l'intervalle [0, 1]. Ça permet de lire le résultat comme une **probabilité**.

L'**hypothèse** du modèle (`hθ(x) = g(θᵀx)`) fait ça en deux temps : d'abord une combinaison linéaire classique des 13 notes (`θ0 + θ1×Arithmancy + θ2×Astronomy + ...`, exactement comme dans `ft_linear_regression`), puis le résultat passe par la sigmoïde pour devenir une probabilité. Une régression logistique, ce n'est jamais qu'**une régression linéaire, plus une sigmoïde par-dessus**.

Pour ajuster les thetas (les coefficients), il faut une fonction de coût qui mesure les erreurs, puis la minimiser. Ici, c'est la **log loss**, pas le MSE utilisé en régression linéaire — parce qu'avec une sigmoïde dans le modèle, le MSE donnerait une fonction de coût avec plusieurs creux (la descente de gradient risquerait de rester coincée dans un mauvais minimum). La log loss punit sévèrement une prédiction confiante et fausse (si la vraie réponse est "oui" et que le modèle prédit 0.01, le coût explose), et garde une seule vallée bien lisse, garantissant que la descente de gradient trouve le bon minimum.

Fait amusant et très utile en soutenance : le **gradient** de cette log loss a exactement la même forme que celui du MSE en régression linéaire — `(prédiction - réalité) × feature`, moyenné sur tous les élèves. La **descente de gradient** fonctionne donc de façon identique aux deux projets : à chaque itération (50 000 fois ici), on calcule ce gradient sur tout le dataset, puis on avance tous les thetas d'un petit pas dans la direction opposée — toujours la même image du randonneur qui descend une montagne dans le brouillard, en sentant la pente sous ses pieds à chaque pas.

Reste un problème : la sigmoïde ne répond qu'à une question binaire ("oui/non"), mais on a **4 maisons**. La solution s'appelle le **one-vs-all** : on entraîne **4 modèles complètement indépendants**, chacun spécialisé sur une seule question ("est-ce Gryffondor, ou pas ?", "est-ce Serpentard, ou pas ?"...). C'est exactement la boucle centrale de `logreg_train.py` (détail dans `code-memento.md`) : transformer la colonne "Hogwarts House" en une colonne de 0 et 1 pour chaque maison, et entraîner un modèle séparé sur chacune.

Dernier ingrédient technique : la **standardisation**. Comme dans `ft_linear_regression`, mélanger des matières notées sur des échelles très différentes (des milliers de points pour Arithmancy, entre -5 et 10 pour Care of Magical Creatures) ferait diverger la descente de gradient. Ici on utilise le **z-score** (`(x - moyenne) / écart-type`) plutôt que le min-max utilisé dans l'autre projet — plus robuste face aux valeurs extrêmes, puisqu'il s'appuie sur la moyenne et l'écart-type plutôt que sur le min/max bruts. La règle d'or à ne jamais oublier : calculer cette moyenne/écart-type sur le jeu d'**entraînement**, et réutiliser ces mêmes valeurs telles quelles à la prédiction — jamais les recalculer sur le jeu de test (c'est le bug qu'on a corrigé ensemble dans `logreg_predict.py`).

### Étape 6 — Deviner la maison de nouveaux élèves : `logreg_predict.py`

Une fois les 4 modèles entraînés et leurs thetas sauvegardés dans `thetas.csv`, `logreg_predict.py` les applique à de nouveaux élèves (`dataset_test.csv`, dont on ne connaît pas la maison). Pour chaque élève, on interroge les 4 modèles indépendamment, et on garde la maison dont le modèle est **le plus confiant** — techniquement, ça s'appelle l'`argmax` : on prend l'indice de la plus grande probabilité parmi les 4. C'est la mise en pratique concrète du "demander à 4 experts spécialisés, et écouter celui qui est le plus sûr de lui".

Le résultat final atteint en moyenne **98.41%** de bonnes réponses (mesuré nous-mêmes par validation croisée, en cachant volontairement 20% du jeu d'entraînement pour vérifier les prédictions du modèle contre les vraies réponses) — au-dessus du seuil de 98% exigé par le sujet pour que "l'algorithme soit comparable au Choixpeau".

---

## Glossaire express

| Terme | Définition en une phrase |
|---|---|
| **Correction de Bessel** | Diviser par `n-1` plutôt que `n` pour un écart-type non biaisé sur un échantillon |
| **IQR** | Écart interquartile (Q3 - Q1), une mesure de dispersion robuste aux valeurs extrêmes |
| **Skewness** | Mesure de l'asymétrie d'une distribution (0 = symétrique) |
| **Corrélation de Pearson** | Mesure (entre -1 et +1) de la relation linéaire entre deux variables |
| **Sigmoïde** | Fonction qui écrase n'importe quel nombre réel dans [0, 1] |
| **Log loss / cross-entropy** | Fonction de coût de la régression logistique, punit les erreurs confiantes |
| **One-vs-all** | Un modèle binaire par catégorie, pour gérer une classification à plusieurs classes |
| **argmax** | Choisir la catégorie dont le modèle est le plus confiant |
| **Standardisation (z-score)** | (x - moyenne) / écart-type, pour mettre toutes les features à la même échelle |