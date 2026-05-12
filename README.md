# Prédiction de la potabilité de l’eau par IA

Ce projet fournit une application Web utilisant un réseau de neurones artificiels pour prédire si une eau est **potable** ou **non potable** à partir de trois paramètres physico‑chimiques : la teneur en fer (mg/L), en nitrates (mg/L) et en chlorures (mg/L).

## Fonctionnement

1. **Génération des données d'entraînement** : le script génère 10 000 analyses d'eau simulées. Une eau est considérée comme potable si les concentrations en fer, nitrates et chlorures sont toutes inférieures aux seuils de l'OMS (respectivement 0,3 mg/L, 50 mg/L et 250 mg/L).
2. **Entraînement du modèle Deep Learning** : un réseau de neurones avec TensorFlow/Keras est entraîné sur ces données pour apprendre la relation entre les concentrations et la potabilité.
3. **Interface utilisateur** : grâce à Streamlit, l'utilisateur peut saisir de nouvelles valeurs et obtenir instantanément un verdict, accompagné d'une probabilité et des raisons d'une éventuelle non‑conformité.

## Utilisation locale

1. Cloner ce dépôt :

```bash
git clone https://github.com/<votre-nom-utilisateur>/water-potability-ai.git
cd water-potability-ai
```

2. Installer les dépendances (idéalement dans un environnement virtuel ):

```bash
pip install -r requirements.txt
```

3. Lancer l’application Streamlit :

```bash
streamlit run app.py
```

L’application se lance alors dans votre navigateur par défaut. Entrez les valeurs de fer, nitrates et chlorures pour connaître la potabilité estimée par l’IA.

## Déploiement

Pour déployer cette application sur [Streamlit Community Cloud](https://streamlit.io/cloud), créez un dépôt GitHub public contenant ce code et les fichiers listés ci‑dessus. Ensuite :

1. Connectez‑vous sur Streamlit Community Cloud avec votre compte GitHub.
2. Choisissez « **New app** », sélectionnez ce dépôt et indiquez `app.py` comme fichier principal.
3. Lancez le déploiement : votre application sera accessible via une URL publique.

Vous pouvez également déployer cette application sur n’importe quelle plateforme prenant en charge Streamlit (Heroku, Google Cloud Run, etc.).

## Licence

Ce projet est fourni à titre pédagogique sans garantie. Vous pouvez l'utiliser et le modifier librement dans le cadre de vos études.