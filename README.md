# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
import traceback
import sys
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Définir la date de référence
start_date = datetime(2000, 1, 1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Charger et prétraiter les données
def charger_et_pretraiter_donnees(csv_path):
    delimiter = ";"
    usecols = ['annee_numero_de_tirage', 'date_de_tirage', 'boule1', 'boule2', 'boule3', 'boule4', 'boule5',
               'boule6', 'boule7', 'boule8', 'boule9', 'boule10', 'boule11', 'boule12', 'boule13', 'boule14',
               'boule15', 'boule16', 'boule17', 'boule18', 'boule19', 'boule20']
    date_column = 'date_de_tirage'
    date_format = '%d/%m/%Y'

    data = pd.read_csv(csv_path, delimiter=delimiter, usecols=usecols)

    data[date_column] = pd.to_datetime(data[date_column], format=date_format)

    return data

# Diviser les données en ensembles d'entraînement et de test
def diviser_donnees(data):
    jours_depuis_debut = (data['date_de_tirage'] - start_date).dt.days
    numeros_jeu = data[['boule1', 'boule2', 'boule3', 'boule4', 'boule5', 'boule6', 'boule7', 'boule8', 'boule9', 'boule10', 'boule11', 'boule12', 'boule13', 'boule14', 'boule15', 'boule16', 'boule17', 'boule18', 'boule19', 'boule20']].values

    return jours_depuis_debut, numeros_jeu

# Ajuster les hyperparamètres du modèle SVM
def ajuster_hyperparametres_svm(X_train, y_train, X_test, y_test, time_limit_minutes=60):
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=time_limit_minutes)

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    param_grid = {
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['auto', 'scale'] + list(np.logspace(-3, 3, 7)),
        'epsilon': np.linspace(0.1, 1.0, 10)
    }

    modele_svm = SVR()

    random_search = RandomizedSearchCV(modele_svm, param_distributions=param_grid, n_iter=10, cv=kf, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')

    random_search.fit(X_train, y_train)

    elapsed_time = datetime.now() - start_time

    if datetime.now() < end_time:
        print(f"Ajustement SVM terminé en {elapsed_time}")
        best_params = random_search.best_params_
        print("Meilleurs hyperparamètres pour SVM : ", best_params)

        modele_final = SVR(**best_params)
        modele_final.fit(X_train, y_train)
        mse = evaluer_modele(modele_final, X_test, y_test)
        print(f"Score MSE sur les données de test : {mse}")
        return modele_final, True, mse
    else:
        print(f"La recherche d'hyperparamètres pour SVM a été interrompue en raison du temps imparti.")
        return None, False, None

# Évaluer les performances du modèle
def evaluer_modele(modele, X_test, y_test):
    predictions = faire_predictions(modele, X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

# Faire des prédictions avec le modèle
def faire_predictions(modele, X):
    return modele.predict(X)

# Sauvegarder le modèle dans un fichier
def sauvegarder_modele(modele, nom_fichier):
    try:
        with open(nom_fichier, 'wb') as modele_fichier:
            pickle.dump(modele, modele_fichier)
        logging.info(f"Modèle sauvegardé avec succès dans {nom_fichier}")
    except Exception as e:
        error_message = f"Erreur lors de la sauvegarde du modèle : {e}"
        traceback_message = traceback.format_exc()
        full_error_message = error_message + "\n" + traceback_message
        log_error(error_message, traceback_message)
        print(full_error_message)

# Charger un modèle à partir d'un fichier
def charger_modele(nom_fichier):
    try:
        with open(nom_fichier, 'rb') as modele_fichier:
            modele = pickle.load(modele_fichier)
        logging.info(f"Modèle chargé avec succès depuis {nom_fichier}")
        return modele
    except Exception as e:
        error_message = f"Erreur lors du chargement du modèle : {e}"
        traceback_message = traceback.format_exc()
        full_error_message = error_message + "\n" + traceback_message
        log_error(error_message, traceback_message)
        print(full_error_message)
        return None

# Fonction principale
def main():
    global data_test, data_train  # Rendre data_train global

    # Charger les données d'entraînement à partir d'un fichier CSV
    data_train = charger_et_pretraiter_donnees("/home/fred/entrainement.csv")

    # Diviser les données en ensembles d'entraînement et de test
    jours_depuis_debut_train, numeros_jeu_train = diviser_donnees(data_train)
    X_train, X_test, y_train, y_test = train_test_split(numeros_jeu_train, jours_depuis_debut_train, test_size=0.2, random_state=42)

    # Ajuster les hyperparamètres du modèle SVM
    meilleur_modele_svm, _, _ = ajuster_hyperparametres_svm(X_train, y_train, X_test, y_test)

    # Sauvegarder le meilleur modèle SVM trouvé
    meilleur_modele_svm_nom_fichier = "meilleur_modele_svm.pkl"
    sauvegarder_modele(meilleur_modele_svm, meilleur_modele_svm_nom_fichier)

    # Charger les données de test à partir d'un fichier CSV distinct
    data_test = charger_et_pretraiter_donnees("/home/fred/test.csv")

    # Collecter les caractéristiques des numéros les plus récents que vous souhaitez prédire
    derniers_50_tirages = data_test[:50]
    X_new = derniers_50_tirages.iloc[:, 2:].values  # Sélectionnez toutes les colonnes à partir de la troisième (indice 2)

    # Faire des prédictions sur X_new avec le meilleur modèle SVM chargé
    predictions = faire_predictions(meilleur_modele_svm, X_new)

    # Afficher les prédictions pour les 20 prochains numéros
    print("Prédictions SVM pour les 20 prochains numéros :")
    print(predictions[:20])

if __name__ == '__main__':
    main()
import re
import seaborn as sns
def charger_et_pretraiter_donnees(csv_path):
    # ...
    data['date_de_tirage'] = pd.to_datetime(data['date_de_tirage'], format='%d/%m/%Y')
    # ...
colonnes_numeriques = ['boule1', 'boule2', 'boule3', 'boule4', 'boule5',
                       'boule6', 'boule7', 'boule8', 'boule9', 'boule10',
                       'boule11', 'boule12', 'boule13', 'boule14', 'boule15',
                       'boule16', 'boule17', 'boule18', 'boule19', 'boule20']

plt.figure(figsize=(15, 10))
for i, colonne in enumerate(colonnes_numeriques, 1):
    plt.subplot(5, 4, i)
    sns.histplot(data_train[colonne], kde=True)
    plt.title(f'Distribution de {colonne}')
    plt.xlabel(colonne)
    plt.ylabel('Fréquence')

plt.tight_layout()
plt.show()

def ajuster_hyperparametres_svm_manuel(X_train, y_train, X_test, y_test, time_limit_minutes=60):
    # ...
    random_search = RandomizedSearchCV(modele_svm, param_distributions=param_grid, n_iter=100, cv=kf, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
    # ...

for iteration in tqdm(range(1, nombre_total_iterations_svm + 1), desc="SVM Progress"):
    # ...

    with open("output.txt", "w") as output_file:
    # ...
       output_file.write(f"Iteration {iteration}/{nombre_total_iterations_svm} - MSE: {mse}\n")

if len(data_test) < 50:
    print("Il n'y a pas suffisamment de données de test pour faire des prédictions.")
    exit(1)

# Collectez les caractéristiques des numéros les plus récents que vous souhaitez prédire
derniers_50_tirages = data_test[:50]
# ...
predictions = meilleur_modele_svm.predict(X_new)
print("Prédictions SVM pour les 20 prochains numéros :")
print(predictions[:20])

if __name__ == '__main__':
    main()
