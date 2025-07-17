import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
from glob import glob
import requests
import zipfile

app = Flask(__name__)
CORS(app)

ufc_model = None
ufc_scaler = None
ufc_feature_cols = None
ufc_df_processed = None
ufc_full_df = None

tennis_model = None
tennis_scaler = None
tennis_feature_cols = None
tennis_df = None

UFC_DATASET = os.getenv("DATASET_PATH", "datas/Large set/large_dataset.csv")
TENNIS_DATA_DIR = "datas/tennis"
TENNIS_GITHUB_URL = "https://github.com/JeffSackmann/tennis_atp/archive/refs/heads/master.zip"

BUCKET = "ufc-fighters-img"
IMG_PREFIX = "images/"
DEFAULT_KEY = f"{IMG_PREFIX}default-img.png"
IMG_BASE = f"https://{BUCKET}.s3.amazonaws.com/"
HAS_IMAGE = {
    "Ciryl Gane", "Amir Albazi", "Brandon Royval", "Brandon Moreno", "Joshua Van",
    "Alexandre Pantoja", "Song Yadong", "Petr Yan", "Umar Nurmagomedov", "Sean O Malley",
    "Merab Dvalishvili", "Arnold Allen", "Brian Ortega", "Yair Rodriguez", "Diego Lopes",
    "Movsar Evloev", "Alexander Volkanovski", "Dustin Poirier", "Max Holloway", "Justin Gaethje",
    "Charles Oliveira", "Arman Tsarukyan", "Ilia Topuria", "Kamaru Usman", "Leon Edwards",
    "Shavkat Rakhmonov", "Sean Brady", "Belal Muhammad", "Jack Della Maddalena", "Robert Whittaker",
    "Israel Adesanya", "Khamzat Chimaev", "Sean Strickland", "Nassourdine Imavov", "Dricus du Plessis",
    "Jan Blachowicz", "Khalil Rountree", "Carlos Ulberg", "Jiri Prochazka", "Alex Pereira",
    "Magomed Ankalaev", "Jailton Almeida", "Curtis Blaydes", "Sergei Pavlovich", "Alexander Volkov",
    "Tom Aspinall"
}
MIN_FIGHTS = 3

import boto3
import botocore
from botocore import UNSIGNED
from botocore.client import Config

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

try:
    ufc_full_df = pd.read_csv(UFC_DATASET)
    fight_counts = ufc_full_df[["r_fighter", "b_fighter"]].stack().value_counts()
    valid_names = set(fight_counts[fight_counts >= MIN_FIGHTS].index)
    ufc_full_df = ufc_full_df[
        ufc_full_df["r_fighter"].isin(valid_names) &
        ufc_full_df["b_fighter"].isin(valid_names)
    ].reset_index(drop=True)
    print("✅ Données UFC chargées")
except Exception as e:
    print(f"⚠️ Erreur chargement UFC: {e}")

# ========== FONCTIONS UFC ==========

def slug(name: str) -> str:
    return name.lower().replace(" ", "-").replace("'", "").replace(".", "")

def object_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except botocore.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        return code not in ("404", "403", "NoSuchKey")

def img_url(name: str) -> str:
    key = f"{IMG_PREFIX}{slug(name)}.png"
    return f"{IMG_BASE}{key}"

def load_and_preprocess_ufc_data() -> pd.DataFrame:
    print("📦 Chargement et préparation du dataset UFC…")
    
    fights_df = pd.read_csv(UFC_DATASET)
    
    mask = fights_df["r_reach"].notna() & fights_df["r_height"].notna()
    if mask.sum() >= 50:
        reg = LinearRegression().fit(
            fights_df.loc[mask, ["r_height"]], fights_df.loc[mask, "r_reach"]
        )
        a, b = reg.coef_[0], reg.intercept_
    else:
        a, b = 1.02, 0.0
    
    for side in ["r", "b"]:
        h, r = f"{side}_height", f"{side}_reach"
        miss = fights_df[r].isna() & fights_df[h].notna()
        fights_df.loc[miss, r] = fights_df.loc[miss, h] * a + b
    
    fights_df.fillna({
        "reach_diff": fights_df["reach_diff"].mean(),
        "age_diff": fights_df["age_diff"].mean(),
        "height_diff": fights_df["height_diff"].mean(),
        "weight_diff": fights_df["weight_diff"].mean(),
        "r_age": fights_df["r_age"].mean(),
        "b_age": fights_df["b_age"].mean(),
        "r_reach": fights_df["r_reach"].mean(),
        "b_reach": fights_df["b_reach"].mean()
    }, inplace=True)
    
    fights_df["r_stance"].fillna("Unknown", inplace=True)
    fights_df["b_stance"].fillna("Unknown", inplace=True)
    
    fights_df["winner_encoded"] = fights_df["winner"].map({"Red": 1, "Blue": 0})
    
    drop_cols = [
        "winner", "r_fighter", "b_fighter", "event_name", "referee",
        "method", "time_sec", "finish_round", "total_rounds", "is_title_bout"
    ]
    fights_df.drop(columns=drop_cols, inplace=True, errors="ignore")
    
    drop_in_fight = [
        col for col in fights_df.columns
        if col in [
            "r_kd", "r_sig_str", "r_sig_str_att", "r_sig_str_acc",
            "r_str", "r_str_att", "r_str_acc", "r_td", "r_td_att",
            "r_td_acc", "r_sub_att", "r_rev", "r_ctrl_sec",
            "b_kd", "b_sig_str", "b_sig_str_att", "b_sig_str_acc",
            "b_str", "b_str_att", "b_str_acc", "b_td", "b_td_att",
            "b_td_acc", "b_sub_att", "b_rev", "b_ctrl_sec"
        ] or col.endswith("_diff")
    ]
    fights_df.drop(columns=drop_in_fight, inplace=True, errors="ignore")
    
    label_cols = ["weight_class", "gender"]
    for col in label_cols:
        le = LabelEncoder()
        fights_df[col] = le.fit_transform(fights_df[col])
    
    stance_le = LabelEncoder()
    fights_df["r_stance"] = stance_le.fit_transform(fights_df["r_stance"])
    fights_df["b_stance"] = stance_le.fit_transform(fights_df["b_stance"])
    
    print(f"[INFO] Dataset UFC prêt : {len(fights_df)} lignes – {fights_df.shape[1]} features")
    return fights_df

def train_ufc_model():
    global ufc_model, ufc_scaler, ufc_feature_cols, ufc_df_processed
    
    print("Entraînement du modèle UFC...")
    ufc_df_processed = load_and_preprocess_ufc_data()
    
    X = ufc_df_processed.drop('winner_encoded', axis=1)
    y = ufc_df_processed['winner_encoded']
    ufc_feature_cols = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    ufc_scaler = StandardScaler()
    X_train_scaled = ufc_scaler.fit_transform(X_train)
    X_test_scaled = ufc_scaler.transform(X_test)
    
    inputs = Input(shape=(X_train_scaled.shape[1],))
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    ufc_model = Model(inputs=inputs, outputs=outputs)
    ufc_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(zip(np.unique(y_train), weights))
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = ufc_model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=200,
        batch_size=64,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1
    )
    
    val_accuracy = history.history['val_accuracy'][-1]
    print(f"Modèle UFC entraîné - Accuracy validation : {val_accuracy:.2%}")
    return val_accuracy

def any_corner_stats_ufc(fighter: str) -> pd.Series:
    mask = (ufc_full_df["r_fighter"] == fighter) | (ufc_full_df["b_fighter"] == fighter)
    return ufc_full_df.loc[mask].select_dtypes("number").mean().fillna(0)

def build_ufc_fight_features(red_fighter: str, blue_fighter: str) -> pd.DataFrame:
    def stats_for(fighter, corner):
        mask = ufc_full_df[f"{corner}_fighter"] == fighter
        return ufc_full_df.loc[mask].select_dtypes("number").mean()
    
    red_stats_r = stats_for(red_fighter, "r")
    red_stats_b = stats_for(red_fighter, "b")
    blue_stats_r = stats_for(blue_fighter, "r")
    blue_stats_b = stats_for(blue_fighter, "b")
    
    red_stats = red_stats_r.where(red_stats_b.isna(), (red_stats_r + red_stats_b) / 2)
    blue_stats = blue_stats_r.where(blue_stats_b.isna(), (blue_stats_r + blue_stats_b) / 2)
    
    if red_stats.isna().all():
        red_stats = any_corner_stats_ufc(red_fighter)
    if blue_stats.isna().all():
        blue_stats = any_corner_stats_ufc(blue_fighter)
    
    feats = {}
    for col in ufc_feature_cols:
        if col.startswith("r_"):
            feats[col] = red_stats.get(col, 0)
        elif col.startswith("b_"):
            feats[col] = blue_stats.get(col, 0)
        else:
            feats[col] = (red_stats.get(col, 0) + blue_stats.get(col, 0)) / 2
    
    if "age_diff" in ufc_feature_cols:
        feats["age_diff"] = red_stats.get("r_age", 0) - blue_stats.get("b_age", 0)
    
    return pd.DataFrame([feats]).reindex(columns=ufc_feature_cols, fill_value=0)

# ========== FONCTIONS TENNIS ==========



def load_and_preprocess_tennis_data():
    global tennis_df
    
    print(" Chargement et préparation du dataset tennis...")

    
    
    pattern = os.path.join(TENNIS_DATA_DIR, "atp_matches_20*.csv")
    files = sorted(glob(pattern))
    
    dfs = []
    for f in files:
        try:
            df_temp = pd.read_csv(f)
            dfs.append(df_temp)
        except Exception as e:
            print(f"Erreur lecture {f}: {e}")
    
    if not dfs:
        raise ValueError("Aucun fichier tennis trouvé")
    
    df = pd.concat(dfs, ignore_index=True)
    
    np.random.seed(42)
    mask = np.random.rand(len(df)) > 0.5
    
    df['player_1'] = np.where(mask, df['winner_name'], df['loser_name'])
    df['player_2'] = np.where(mask, df['loser_name'], df['winner_name'])
    df['player_1_rank'] = np.where(mask, df['winner_rank'], df['loser_rank'])
    df['player_2_rank'] = np.where(mask, df['loser_rank'], df['winner_rank'])
    df['player_1_age'] = np.where(mask, df['winner_age'], df['loser_age'])
    df['player_2_age'] = np.where(mask, df['loser_age'], df['winner_age'])
    df['target'] = np.where(mask, 1, 0)
    
    for col in ['player_1_age', 'player_2_age']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    max_rank = max(df['player_1_rank'].max(), df['player_2_rank'].max())
    for col in ['player_1_rank', 'player_2_rank']:
        if col in df.columns:
            df[col] = df[col].fillna(max_rank + 1)
    
    df['rank_diff'] = df['player_1_rank'] - df['player_2_rank']
    df['age_diff'] = df['player_1_age'] - df['player_2_age']
    
    df = pd.get_dummies(df, columns=['surface', 'tourney_level'], drop_first=True)
    
    tennis_df = df
    print(f"ataset tennis préparé : {len(df)} matchs")
    return df

def train_tennis_model():
    global tennis_model, tennis_scaler, tennis_feature_cols
    
    print("🎾 Entraînement du modèle tennis...")
    df = load_and_preprocess_tennis_data()
    
    features = [
        'player_1_rank', 'player_2_rank', 'player_1_age', 'player_2_age',
        'rank_diff', 'age_diff'
    ] + [c for c in df.columns if c.startswith('surface_') or c.startswith('tourney_level_')]
    
    available_features = [f for f in features if f in df.columns]
    tennis_feature_cols = available_features
    
    X = df[tennis_feature_cols]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    tennis_scaler = StandardScaler()
    X_train_scaled = tennis_scaler.fit_transform(X_train)
    X_test_scaled = tennis_scaler.transform(X_test)
    
    tennis_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    tennis_model.fit(X_train_scaled, y_train)
    
    y_pred = tennis_model.predict(X_test_scaled)
    accuracy = (y_pred == y_test).mean()
    
    print(f"Modèle tennis entraîné - Accuracy : {accuracy:.3f}")
    return accuracy

def get_player_stats(player_name):
    global tennis_df
    
    if tennis_df is None:
        return None
    
    p1_matches = tennis_df[tennis_df['player_1'] == player_name]
    p2_matches = tennis_df[tennis_df['player_2'] == player_name]
    all_matches = pd.concat([p1_matches, p2_matches])
    
    if len(all_matches) == 0:
        return None
    
    stats = {
        'rank': all_matches[['player_1_rank', 'player_2_rank']].mean().mean(),
        'age': all_matches[['player_1_age', 'player_2_age']].mean().mean(),
        'matches_count': len(all_matches)
    }
    return stats

def build_tennis_features(player1, player2, surface="Hard", tourney_level="A"):
    global tennis_feature_cols
    
    p1_stats = get_player_stats(player1)
    p2_stats = get_player_stats(player2)
    
    if p1_stats is None:
        p1_rank, p1_age = 100, 25
    else:
        p1_rank, p1_age = p1_stats['rank'], p1_stats['age']
    
    if p2_stats is None:
        p2_rank, p2_age = 100, 25
    else:
        p2_rank, p2_age = p2_stats['rank'], p2_stats['age']
    
    features = {
        'player_1_rank': p1_rank,
        'player_2_rank': p2_rank,
        'player_1_age': p1_age,
        'player_2_age': p2_age,
        'rank_diff': p1_rank - p2_rank,
        'age_diff': p1_age - p2_age
    }
    
    for col in tennis_feature_cols:
        if col.startswith('surface_'):
            surface_name = col.replace('surface_', '')
            features[col] = 1 if surface == surface_name else 0
        elif col.startswith('tourney_level_'):
            level_name = col.replace('tourney_level_', '')
            features[col] = 1 if tourney_level == level_name else 0
        elif col not in features:
            features[col] = 0
    
    return pd.DataFrame([features])[tennis_feature_cols]

# ========== ENDPOINTS UFC ==========

@app.route('/train', methods=['POST'])
def train_ufc_endpoint():
    try:
        accuracy = train_ufc_model()
        return jsonify({
            'success': True,
            'message': 'Modèle UFC entraîné avec succès',
            'accuracy': float(accuracy)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict_ufc_endpoint():
    global ufc_model, ufc_scaler, ufc_feature_cols
    try:
        data = request.json
        red = data["red_fighter"]
        blue = data["blue_fighter"]
        
        if not ufc_model:
            return jsonify({"error": "model not trained"}), 400
        
        # Passe 1: A rouge
        f1 = build_ufc_fight_features(red, blue)
        f1_scaled = ufc_scaler.transform(f1.fillna(0))
        p1 = float(ufc_model.predict(f1_scaled, verbose=0)[0][0])
        
        # Passe 2: B rouge
        f2 = build_ufc_fight_features(blue, red)
        f2_scaled = ufc_scaler.transform(f2.fillna(0))
        p2 = float(ufc_model.predict(f2_scaled, verbose=0)[0][0])
        
        # Fusion symétrique
        p_red = 0.5 * (p1 + (1 - p2))
        p_blue = 1.0 - p_red
        winner = red if p_red > 0.5 else blue
        conf = max(p_red, p_blue)
        
        return jsonify({
            "red_fighter": red,
            "blue_fighter": blue,
            "red_probability": p_red,
            "blue_probability": p_blue,
            "predicted_winner": winner,
            "confidence": conf
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.get("/fighters")
def get_ufc_fighters():
    global ufc_full_df
    if ufc_full_df is None:
        ufc_full_df = pd.read_csv(UFC_DATASET)
    
    gender = request.args.get("gender")
    weight_class = request.args.get("weight_class")
    
    df = ufc_full_df
    if gender:
        df = df[df["gender"] == gender]
    if weight_class:
        df = df[df["weight_class"] == weight_class]
    
    names = df[["r_fighter", "b_fighter"]].stack().dropna().unique()
    
    fighters = []
    for n in sorted(names):
        has_img = n in HAS_IMAGE
        fighters.append({
            "name": n,
            "has_img": has_img,
            "img": f"{IMG_BASE}{IMG_PREFIX}{slug(n)}.png" if has_img else None
        })
    
    return jsonify({"fighters": fighters})

# ========== ENDPOINTS TENNIS ==========

@app.route('/tennis/train', methods=['POST'])
def train_tennis_endpoint():
    try:
        accuracy = train_tennis_model()
        return jsonify({
            'success': True,
            'message': 'Modèle tennis entraîné avec succès',
            'accuracy': float(accuracy)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/tennis/predict', methods=['POST'])
def predict_tennis_endpoint():
    global tennis_model, tennis_scaler, tennis_feature_cols
    
    try:
        data = request.json
        player1 = data['player1']
        player2 = data['player2']
        surface = data.get('surface', 'Hard')
        tourney_level = data.get('tourney_level', 'A')
        
        if not tennis_model:
            return jsonify({'error': 'Modèle tennis non entraîné'}), 400
        
        features = build_tennis_features(player1, player2, surface, tourney_level)
        features_scaled = tennis_scaler.transform(features)
        
        proba = tennis_model.predict_proba(features_scaled)[0]
        p1_proba = float(proba[1])
        p2_proba = float(proba[0])
        
        winner = player1 if p1_proba > 0.5 else player2
        confidence = max(p1_proba, p2_proba)
        
        return jsonify({
            'player1': player1,
            'player2': player2,
            'surface': surface,
            'tourney_level': tourney_level,
            'player1_probability': p1_proba,
            'player2_probability': p2_proba,
            'predicted_winner': winner,
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tennis/players', methods=['GET'])
def get_tennis_players():
    global tennis_df
    
    try:
        if tennis_df is None:
            load_and_preprocess_tennis_data()
        
        players = set()
        players.update(tennis_df['player_1'].dropna().unique())
        players.update(tennis_df['player_2'].dropna().unique())
        
        players_list = []
        for player in sorted(players):
            stats = get_player_stats(player)
            if stats['matches_count'] > 150:
                players_list.append({
                    'name': player,
                    'matches_count': stats['matches_count'] if stats else 0,
                    'avg_rank': round(stats['rank']) if stats else None
                })
        
        players_list.sort(key=lambda x: x['name'], reverse=True)
        
        return jsonify({
            'players': players_list[:500],
            'total': len(players_list)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tennis/surfaces', methods=['GET'])
def get_tennis_surfaces():
    try:
        surfaces = ['Hard', 'Clay', 'Grass', 'Carpet']
        levels = ['A', 'D', 'F', 'G', 'M']
        
        return jsonify({
            'surfaces': surfaces,
            'tourney_levels': levels
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========== ENDPOINTS COMMUNS ==========

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'ufc_model_trained': ufc_model is not None,
        'tennis_model_trained': tennis_model is not None
    })

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'ufc': {
            'model_trained': ufc_model is not None,
            'data_loaded': ufc_full_df is not None
        },
        'tennis': {
            'model_trained': tennis_model is not None,
            'data_loaded': tennis_df is not None
        }
    })

if __name__ == '__main__':
    print("Démarrage du serveur Sports Predictor unifié...")
    print("UFC disponible sur tous les endpoints existants")
    print("Tennis disponible sur /tennis/*")
    app.run(host='0.0.0.0', port=8000, debug=True)