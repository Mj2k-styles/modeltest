# Prédictions Machine Learning : 4 modèles

Projet de Machine Learning (L3) : 2 modèles de classification et 2 modèles de régression, réunis dans une application Streamlit.

| Modèle | Type | Données | Variable prédite | Notebook |
|---|---|---|---|---|
| Téléphones | Classification | `Telephone_data.csv` | `etat` (D'occasion, Neuf, Réconditionné, Venant) | `Copie de Clas_Log_Reg_Dec_Tree_Random_Forest.ipynb` |
| Voitures (venant / occasion) | Classification | `Car_Data (1).csv` | `Venant_Occasion` | `Car_Data_Clas_Log_Reg_Dec_Tree_Random_Forest.ipynb` |
| Voitures (prix) | Régression | `Car_dataset.csv` | `Selling_Price` | `Car_dataset_Reg_Mul_Reg_Ridge_Dec_Tree_Random_Forest.ipynb` |
| Taxi | Régression | `taxi_trip_pricing.csv` | `Trip_Price` | `Copie de Reg_Mul_Reg_Ridge_Dec_Tree_Random_Forest.ipynb` |

Chaque notebook suit la même démarche : exploration, préparation (encodage, valeurs aberrantes, normalisation), comparaison de plusieurs modèles (régression logistique / linéaire, Ridge, Decision Tree, Random Forest), puis **déploiement** : le modèle le plus performant est enregistré dans `models/<nom>/`.

## Résultats sur les données de validation

| Modèle | Meilleur algorithme | Score |
|---|---|---|
| Téléphones | Random Forest | Accuracy ≈ 0.86 |
| Voitures (venant / occasion) | Random Forest | Accuracy ≈ 0.82 |
| Voitures (prix) | Random Forest | R² ≈ 0.88 |
| Taxi | Random Forest | R² ≈ 0.90 |

## Application

Chaque modèle a deux interfaces :
- **Prédiction simple** : un formulaire, une prédiction.
- **Prédiction multiple** : on envoie un fichier csv, on récupère le fichier avec les prédictions. Un fichier d'exemple est téléchargeable dans l'application (`models/<nom>/exemple.csv`).

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pour réentraîner un modèle, exécuter son notebook : la dernière cellule met à jour `models/<nom>/`.
