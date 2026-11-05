# Application Streamlit : les 4 modèles (2 classifications, 2 régressions)
# Lancer avec :  streamlit run app.py
import os
import joblib as jb
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Présentation de chaque modèle (dossier dans models/)
MODELS = {
  'telephone': {
    'icon': '📱', 'name': 'Téléphones', 'type': 'Classification', 'color': '#4F46E5',
    'question': "Dans quel état est ce téléphone ?",
    'description': "À partir du prix, de la marque, de l'écran, de la RAM et du stockage, le modèle prédit si le téléphone est neuf, reconditionné, d'occasion ou venant.",
    'score': 'Accuracy 0.86', 'unit': 'FCFA',
  },
  'car_data': {
    'icon': '🚙', 'name': 'Voitures : venant ou occasion', 'type': 'Classification', 'color': '#0EA5E9',
    'question': "Cette voiture est-elle venant ou d'occasion ?",
    'description': "À partir de la marque, de l'année, de la boîte de vitesse, de l'adresse et du prix, le modèle prédit si la voiture vient de l'étranger (venant) ou est d'occasion locale.",
    'score': 'Accuracy 0.82', 'unit': 'FCFA',
  },
  'car_dataset': {
    'icon': '🚗', 'name': 'Voitures : prix de revente', 'type': 'Régression', 'color': '#10B981',
    'question': "À combien cette voiture peut-elle se revendre ?",
    'description': "À partir du kilométrage, du prix neuf, du carburant, du vendeur, de la boîte de vitesse et de l'âge, le modèle estime le prix de revente.",
    'score': 'R² 0.88', 'unit': '',
  },
  'taxi': {
    'icon': '🚕', 'name': 'Taxi : prix de la course', 'type': 'Régression', 'color': '#F59E0B',
    'question': "Combien va coûter cette course ?",
    'description': "À partir de la distance, de la durée, du moment de la journée, du trafic, de la météo et des tarifs, le modèle estime le prix de la course.",
    'score': 'R² 0.90', 'unit': '$',
  },
}

# Libellés en français des variables
LABELS = {
  'prix': 'Prix (FCFA)', 'adresse': 'Adresse', 'marque': 'Marque', 'dim_ecr': "Taille de l'écran (pouces)",
  'ram': 'RAM (Go)', 'stockage': 'Stockage (Go)', 'etat': 'État',
  'Marque': 'Marque', 'Année': 'Année', 'Boite_Vitesse': 'Boîte de vitesse', 'Adresse': 'Adresse',
  'Prix': 'Prix (FCFA)', 'Venant_Occasion': 'Provenance',
  'Kms_Driven': 'Kilométrage (km)', 'Present_Price': 'Prix neuf actuel', 'Fuel_Type': 'Carburant',
  'Seller_Type': 'Type de vendeur', 'Transmission': 'Boîte de vitesse', 'Age': 'Âge (années)',
  'Selling_Price': 'Prix de revente',
  'Trip_Distance_km': 'Distance (km)', 'Time_of_Day': 'Moment de la journée', 'Day_of_Week': 'Jour',
  'Passenger_Count': 'Nombre de passagers', 'Traffic_Conditions': 'Trafic', 'Weather': 'Météo',
  'Base_Fare': 'Prise en charge', 'Per_Km_Rate': 'Tarif par km', 'Per_Minute_Rate': 'Tarif par minute',
  'Trip_Duration_Minutes': 'Durée (minutes)', 'Trip_Price': 'Prix de la course',
}

# Traduction des valeurs catégorielles (affichage seulement)
VALUES = {
  'Petrol': 'Essence', 'Diesel': 'Diesel', 'CNG': 'GNV', 'Dealer': 'Concessionnaire', 'Individual': 'Particulier',
  'Automatic': 'Automatique', 'Manual': 'Manuelle', 'Morning': 'Matin', 'Afternoon': 'Après-midi',
  'Evening': 'Soir', 'Night': 'Nuit', 'Weekday': 'Semaine', 'Weekend': 'Week-end', 'Low': 'Faible',
  'Medium': 'Moyen', 'High': 'Dense', 'Clear': 'Dégagé', 'Rain': 'Pluie', 'Snow': 'Neige',
}

# Couleur de chaque classe prédite
CLASS_COLORS = {
  'Neuf': '#10B981', 'Réconditionné': '#0EA5E9', "D'occasion": '#F59E0B', 'Venant': '#8B5CF6',
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button, input, select, textarea { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1100px; }
#MainMenu, footer { visibility: hidden; }

/* bandeau */
.hero { background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #DB2777 100%); border-radius: 22px;
  padding: 2.2rem 2.4rem; color: white; margin-bottom: 1.6rem; box-shadow: 0 18px 40px -18px rgba(79,70,229,.6); }
.hero h1 { color: white; font-size: 2.3rem; font-weight: 800; margin: 0 0 .4rem 0; letter-spacing: -0.02em; }
.hero p { color: rgba(255,255,255,.88); font-size: 1.05rem; margin: 0; max-width: 680px; }
.hero .pills { margin-top: 1.1rem; display: flex; gap: .5rem; flex-wrap: wrap; }
.hero .pill { background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.3); padding: .3rem .8rem;
  border-radius: 999px; font-size: .85rem; font-weight: 500; }

/* cartes */
.card { background: white; border-radius: 18px; padding: 1.4rem 1.5rem; border: 1px solid #E8EAF3;
  box-shadow: 0 4px 18px -8px rgba(30,27,75,.12); height: 100%; transition: transform .15s, box-shadow .15s; }
.card:hover { transform: translateY(-2px); box-shadow: 0 12px 28px -10px rgba(30,27,75,.2); }
.card .icon { font-size: 2rem; width: 3.2rem; height: 3.2rem; border-radius: 14px; display: flex;
  align-items: center; justify-content: center; margin-bottom: .8rem; }
.card h3 { font-size: 1.12rem; font-weight: 700; margin: 0 0 .35rem 0; color: #1E1B4B; }
.card p { color: #5B5F7A; font-size: .92rem; margin: 0 0 .9rem 0; line-height: 1.45; }
.badge { display: inline-block; padding: .22rem .65rem; border-radius: 999px; font-size: .78rem; font-weight: 600; margin-right: .35rem; }

/* en-tête d'un modèle */
.model-head { display: flex; align-items: center; gap: 1rem; background: white; border: 1px solid #E8EAF3;
  border-radius: 18px; padding: 1.2rem 1.4rem; margin-bottom: 1.2rem; box-shadow: 0 4px 18px -8px rgba(30,27,75,.12); }
.model-head .icon { font-size: 2.2rem; width: 3.6rem; height: 3.6rem; border-radius: 16px; display: flex;
  align-items: center; justify-content: center; flex-shrink: 0; }
.model-head h2 { font-size: 1.45rem; font-weight: 800; margin: 0; color: #1E1B4B; padding: 0; }
.model-head p { margin: .2rem 0 0 0; color: #5B5F7A; font-size: .95rem; }

/* résultat */
.result { border-radius: 18px; padding: 1.5rem 1.6rem; color: white; margin-top: .6rem;
  box-shadow: 0 14px 30px -14px rgba(30,27,75,.45); }
.result .label { font-size: .85rem; text-transform: uppercase; letter-spacing: .08em; opacity: .85; font-weight: 600; }
.result .value { font-size: 2.4rem; font-weight: 800; margin: .2rem 0 .1rem 0; letter-spacing: -0.02em; }
.result .sub { font-size: .95rem; opacity: .9; }
.proba { margin-top: .9rem; }
.proba .row { display: flex; align-items: center; gap: .7rem; margin: .38rem 0; font-size: .9rem; color: #1E1B4B; }
.proba .name { width: 120px; font-weight: 500; }
.proba .bar { flex: 1; background: #EEF0F7; border-radius: 999px; height: 10px; overflow: hidden; }
.proba .fill { height: 100%; border-radius: 999px; }
.proba .pct { width: 48px; text-align: right; font-weight: 600; }

/* formulaires, onglets, boutons */
[data-testid="stForm"] { background: white; border: 1px solid #E8EAF3; border-radius: 18px; padding: 1.4rem 1.4rem .8rem 1.4rem;
  box-shadow: 0 4px 18px -8px rgba(30,27,75,.12); }
.stTabs [data-baseweb="tab-list"] { gap: .4rem; background: #ECEEF8; padding: .35rem; border-radius: 14px; width: fit-content; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: .45rem 1.1rem; height: auto; font-weight: 600; }
.stTabs [aria-selected="true"] { background: white; box-shadow: 0 2px 8px -2px rgba(30,27,75,.2); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button { border-radius: 12px; font-weight: 600; padding: .55rem 1.3rem; }
[data-testid="stFileUploaderDropzone"] { border-radius: 16px; background: white; }
[data-testid="stMetric"] { background: white; border: 1px solid #E8EAF3; border-radius: 16px; padding: 1rem 1.2rem; }

/* barre latérale */
[data-testid="stSidebar"] { background: #1E1B4B; }
[data-testid="stSidebar"] * { color: #E0E7FF; }
[data-testid="stSidebar"] .side-title { font-size: 1.25rem; font-weight: 800; color: white; margin-bottom: .2rem; }
[data-testid="stSidebar"] .side-sub { font-size: .85rem; color: #A5B4FC; margin-bottom: 1.2rem; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: .55rem .7rem; border-radius: 10px; margin-bottom: .15rem; width: 100%; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.08); }
.side-foot { font-size: .78rem; color: #A5B4FC; margin-top: 2rem; line-height: 1.5; }
</style>
"""


# Charger les objets d'un modèle une seule fois (mis en cache)
@st.cache_resource
def load_objects(name):
  path = os.path.join(BASE_DIR, 'models', name)
  # modèle : model.joblib ou rf_model.joblib (nom utilisé dans les notebooks du cours)
  model_file = 'model.joblib' if os.path.exists(os.path.join(path, 'model.joblib')) else 'rf_model.joblib'
  model = jb.load(os.path.join(path, model_file))
  scaler = jb.load(os.path.join(path, 'scaler.joblib'))
  encoders = jb.load(os.path.join(path, 'encoders.joblib'))
  infos = jb.load(os.path.join(path, 'infos.joblib'))
  # encodeurs sauvegardés en liste (comme dans le cours) : les associer à leurs colonnes
  if isinstance(encoders, list):
    encoders = dict(zip(infos['encoder_columns'], encoders))
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
    proba = model.predict_proba(x_new) if hasattr(model, 'predict_proba') else None
    return classes[y_pred], proba
  return np.round(model.predict(x_new), 2), None


def unknown_values(df, infos):
  # valeurs catégorielles que le modèle n'a jamais vues
  errors = []
  for col, values in infos['categories'].items():
    bad = sorted(set(df[col].astype(str)) - set(values))
    if bad:
      errors.append(f"{LABELS.get(col, col)} : {', '.join(bad[:5])}")
  return errors


def fmt_number(value, unit):
  text = f'{value:,.2f}'.replace(',', ' ')
  if text.endswith('.00'):
    text = text[:-3]
  return f'{text} {unit}'.strip()


def class_color(name, meta):
  return CLASS_COLORS.get(name, meta['color'])


def result_card(meta, infos, y_pred, proba):
  target = LABELS.get(infos['target'], infos['target'])
  if infos['task'] == 'classification':
    name = str(y_pred[0])
    color = class_color(name, meta)
    sub = f'Confiance du modèle : {proba[0].max():.0%}' if proba is not None else ''
    st.markdown(f"""<div class="result" style="background: linear-gradient(135deg, {color}, {color}cc);">
      <div class="label">{target} prédit</div><div class="value">{name}</div><div class="sub">{sub}</div></div>""", unsafe_allow_html = True)
    if proba is not None:
      rows = ''
      for cls, p in sorted(zip(infos['classes'], proba[0]), key = lambda t: -t[1]):
        c = class_color(cls, meta)
        rows += f"""<div class="row"><div class="name">{cls}</div><div class="bar"><div class="fill" style="width:{p*100:.1f}%; background:{c};"></div></div><div class="pct">{p:.0%}</div></div>"""
      st.markdown(f'<div class="proba">{rows}</div>', unsafe_allow_html = True)
  else:
    st.markdown(f"""<div class="result" style="background: linear-gradient(135deg, {meta['color']}, {meta['color']}cc);">
      <div class="label">{target} estimé</div><div class="value">{fmt_number(y_pred[0], meta['unit'])}</div>
      <div class="sub">Estimation du modèle Random Forest</div></div>""", unsafe_allow_html = True)


def simple_interface(name, meta, model, scaler, encoders, infos):
  with st.form(f'simple_{name}', border = False):
    cols = st.columns(2)
    values = {}
    for i, col in enumerate(infos['features']):
      label = LABELS.get(col, col)
      with cols[i % 2]:
        if col in infos['categories']:
          values[col] = st.selectbox(label, infos['categories'][col], format_func = lambda v: VALUES.get(v, v), key = f'{name}_{col}')
        elif col in infos['integers']:
          values[col] = st.number_input(label, value = int(round(infos['defaults'][col])), step = 1, key = f'{name}_{col}')
        else:
          values[col] = st.number_input(label, value = float(round(infos['defaults'][col], 2)), key = f'{name}_{col}')
    submitted = st.form_submit_button('✨ Prédire', type = 'primary', width = 'stretch')
  if submitted:
    y_pred, proba = Pred_func(pd.DataFrame([values]), model, scaler, encoders, infos)
    result_card(meta, infos, y_pred, proba)


def multiple_interface(name, meta, model, scaler, encoders, infos):
  left, right = st.columns([3, 2])
  with left:
    st.markdown('**Colonnes attendues** : ' + ', '.join(f'`{c}`' for c in infos['features']))
  with right:
    exemple = os.path.join(BASE_DIR, 'models', name, 'exemple.csv')
    if os.path.exists(exemple):
      with open(exemple, 'rb') as f:
        st.download_button("⬇️ Fichier d'exemple", f.read(), file_name = f'exemple_{name}.csv', mime = 'text/csv', key = f'ex_{name}', width = 'stretch')
  file = st.file_uploader('Dépose ton fichier csv ici', type = 'csv', key = f'file_{name}')
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
  target = infos['target']
  df[target] = None
  if complete.any():
    y_pred, proba = Pred_func(df[complete], model, scaler, encoders, infos)
    df.loc[complete, target] = y_pred
    if proba is not None:
      df.loc[complete, 'Confiance'] = np.round(proba.max(axis = 1), 3)

  # résumé des prédictions
  done = df.loc[complete, target]
  if infos['task'] == 'classification':
    counts = done.value_counts()
    cols = st.columns(len(counts) + 1)
    cols[0].metric('Lignes prédites', len(done))
    for c, (cls, n) in zip(cols[1:], counts.items()):
      c.metric(str(cls), n, f'{n / len(done):.0%}', delta_color = 'off')
  else:
    done = done.astype(float)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Lignes prédites', len(done))
    c2.metric('Moyenne', fmt_number(done.mean(), meta['unit']))
    c3.metric('Minimum', fmt_number(done.min(), meta['unit']))
    c4.metric('Maximum', fmt_number(done.max(), meta['unit']))
  # la prédiction en premier pour qu'elle soit visible sans faire défiler
  first = [c for c in [target, 'Confiance'] if c in df.columns]
  df = df[first + [c for c in df.columns if c not in first]]
  st.dataframe(df.rename(columns = LABELS), width = 'stretch', hide_index = True)
  st.download_button('⬇️ Télécharger les prédictions', df.to_csv(index = False), file_name = f'predictions_{name}.csv',
                     mime = 'text/csv', key = f'dl_{name}', type = 'primary')


def home_page():
  st.markdown("""<div class="hero"><h1>Prédictions Machine Learning</h1>
    <p>Quatre modèles entraînés sur des données réelles : choisis-en un dans le menu, remplis le formulaire ou envoie un fichier csv, et obtiens une prédiction en un clic.</p>
    <div class="pills"><span class="pill">2 classifications</span><span class="pill">2 régressions</span><span class="pill">Prédiction simple &amp; multiple</span></div></div>""", unsafe_allow_html = True)
  names = list(MODELS)
  for row in (names[:2], names[2:]):
    cols = st.columns(2, gap = 'medium')
    for col, name in zip(cols, row):
      meta = MODELS[name]
      with col:
        st.markdown(f"""<div class="card"><div class="icon" style="background:{meta['color']}1f;">{meta['icon']}</div>
          <h3>{meta['name']}</h3><p>{meta['description']}</p>
          <span class="badge" style="background:{meta['color']}1f; color:{meta['color']};">{meta['type']}</span>
          <span class="badge" style="background:#EEF0F7; color:#1E1B4B;">{meta['score']}</span></div>""", unsafe_allow_html = True)
    st.write('')


def model_page(name):
  meta = MODELS[name]
  st.markdown(f"""<div class="model-head"><div class="icon" style="background:{meta['color']}1f;">{meta['icon']}</div>
    <div><h2>{meta['name']}</h2><p>{meta['question']} &nbsp;<span class="badge" style="background:{meta['color']}1f; color:{meta['color']};">{meta['type']}</span>
    <span class="badge" style="background:#EEF0F7; color:#1E1B4B;">{meta['score']}</span></p></div></div>""", unsafe_allow_html = True)
  if not os.path.exists(os.path.join(BASE_DIR, 'models', name, 'infos.joblib')):
    st.warning(f"Modèle introuvable : exécute d'abord le notebook qui crée models/{name}.")
    return
  model, scaler, encoders, infos = load_objects(name)
  simple, multiple = st.tabs(['🎯 Prédiction simple', '📂 Prédiction multiple'])
  with simple:
    simple_interface(name, meta, model, scaler, encoders, infos)
  with multiple:
    multiple_interface(name, meta, model, scaler, encoders, infos)


st.set_page_config(page_title = 'Prédictions ML', page_icon = '🤖', layout = 'wide')
st.markdown(CSS, unsafe_allow_html = True)

with st.sidebar:
  st.markdown('<div class="side-title">🤖 Prédictions ML</div><div class="side-sub">Classification &amp; régression</div>', unsafe_allow_html = True)
  pages = ['home'] + list(MODELS)
  page = st.radio('Navigation', pages, format_func = lambda p: '🏠  Accueil' if p == 'home' else f"{MODELS[p]['icon']}  {MODELS[p]['name']}", label_visibility = 'collapsed')
  st.markdown('<div class="side-foot">Modèles : régression logistique, régression linéaire, Ridge, Decision Tree et Random Forest, comparés sur les données de validation. Le meilleur est utilisé ici.</div>', unsafe_allow_html = True)

if page == 'home':
  home_page()
else:
  model_page(page)
