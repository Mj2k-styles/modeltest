# Application Streamlit : les 4 modèles (2 classifications, 2 régressions)
# Lancer avec :  streamlit run app.py
import os
import joblib as jb
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# dossier dans models/ : nom de l'onglet
MODELS = {
  'telephone': '📱 Téléphones',
  'car_data': '🚙 Voitures (venant / occasion)',
  'car_dataset': '🚗 Voitures (prix)',
  'taxi': '🚕 Taxi',
}

# Charger les objets d'un modèle une seule fois (mis en cache)
@st.cache_resource
def load_objects(name):
  path = os.path.join(BASE_DIR, 'models', name)
  model = jb.load(os.path.join(path, 'model.joblib'))
  scaler = jb.load(os.path.join(path, 'scaler.joblib'))
  encoders = jb.load(os.path.join(path, 'encoders.joblib'))
  infos = jb.load(os.path.join(path, 'infos.joblib'))
  return model, scaler, encoders, infos


def Pred_func(df, model, scaler, encoders, infos):
  # garder les variables du modèle, dans le bon ordre
  x_new = df[infos['features']].copy()
  # Encoder les variables catégorielles
  for col in infos['categories']:
    x_new[col] = encoders[col].transform(x_new[col].astype(str))
  #normaliser
  x_new = scaler.transform(x_new.values.astype(float))
  # Prédire
  if infos['task'] == 'classification':
    classes = np.array(infos['classes'])
    y_pred = model.predict(x_new)
    proba = model.predict_proba(x_new).max(axis = 1) if hasattr(model, 'predict_proba') else None
    return classes[y_pred], proba
  return np.round(model.predict(x_new), 2), None


def unknown_values(df, infos):
  # valeurs catégorielles que le modèle n'a jamais vues
  errors = []
  for col, values in infos['categories'].items():
    bad = sorted(set(df[col].astype(str)) - set(values))
    if bad:
      errors.append(f"{col} : {', '.join(bad[:5])}")
  return errors


def simple_interface(name, model, scaler, encoders, infos):
  with st.form(f'simple_{name}'):
    cols = st.columns(2)
    values = {}
    for i, col in enumerate(infos['features']):
      with cols[i % 2]:
        if col in infos['categories']:
          values[col] = st.selectbox(col, infos['categories'][col], key = f'{name}_{col}')
        elif col in infos['integers']:
          values[col] = st.number_input(col, value = int(round(infos['defaults'][col])), step = 1, key = f'{name}_{col}')
        else:
          values[col] = st.number_input(col, value = float(round(infos['defaults'][col], 2)), key = f'{name}_{col}')
    submitted = st.form_submit_button('Prédire', type = 'primary')
  if submitted:
    y_pred, proba = Pred_func(pd.DataFrame([values]), model, scaler, encoders, infos)
    if infos['task'] == 'classification':
      st.metric(infos['target'], y_pred[0])
      if proba is not None:
        st.caption(f'Confiance du modèle : {proba[0]:.0%}')
    else:
      st.metric(infos['target'], f"{y_pred[0]:,.2f} {infos.get('unit', '')}".replace(',', ' '))


def multiple_interface(name, model, scaler, encoders, infos):
  st.caption('Colonnes attendues : ' + ', '.join(infos['features']))
  exemple = os.path.join(BASE_DIR, 'models', name, 'exemple.csv')
  if os.path.exists(exemple):
    with open(exemple, 'rb') as f:
      st.download_button("Télécharger un fichier d'exemple", f.read(), file_name = f'exemple_{name}.csv', mime = 'text/csv', key = f'ex_{name}')
  file = st.file_uploader('Fichier csv', type = 'csv', key = f'file_{name}')
  if file is None:
    return
  df = pd.read_csv(file)
  missing = [col for col in infos['features'] if col not in df.columns]
  if missing:
    st.error('Colonnes manquantes : ' + ', '.join(missing))
    return
  # les lignes avec des valeurs manquantes ne sont pas prédites
  complete = df[infos['features']].notna().all(axis = 1)
  if not complete.all():
    st.warning(f"{(~complete).sum()} ligne(s) avec des valeurs manquantes : pas de prédiction pour elles.")
  errors = unknown_values(df[complete], infos)
  if errors:
    st.error('Valeurs inconnues du modèle → ' + ' | '.join(errors))
    return
  df[infos['target']] = None
  if complete.any():
    y_pred, proba = Pred_func(df[complete], model, scaler, encoders, infos)
    df.loc[complete, infos['target']] = y_pred
    if proba is not None:
      df.loc[complete, 'Confiance'] = np.round(proba, 3)
  st.dataframe(df, width = 'stretch')
  st.download_button('Télécharger les prédictions', df.to_csv(index = False), file_name = f'predictions_{name}.csv', mime = 'text/csv', key = f'dl_{name}')


st.set_page_config(page_title = 'Prédictions ML', page_icon = '🤖', layout = 'centered')
st.title('🤖 Prédictions Machine Learning')
st.write('4 modèles : 2 classifications (téléphones, voitures venant / occasion) et 2 régressions (prix des voitures, prix des courses de taxi).')

for tab, (name, label) in zip(st.tabs(list(MODELS.values())), MODELS.items()):
  with tab:
    if not os.path.exists(os.path.join(BASE_DIR, 'models', name, 'infos.joblib')):
      st.warning(f"Modèle introuvable : exécute d'abord le notebook qui crée models/{name}.")
      continue
    model, scaler, encoders, infos = load_objects(name)
    st.subheader(infos['title'])
    simple, multiple = st.tabs(['Prédiction simple', 'Prédiction multiple'])
    with simple:
      simple_interface(name, model, scaler, encoders, infos)
    with multiple:
      multiple_interface(name, model, scaler, encoders, infos)
