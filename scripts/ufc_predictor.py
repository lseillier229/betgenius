import pandas as pd
import numpy as np
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

app = Flask(__name__)
CORS(app)

model = None
scaler = None
feature_cols = None
df_processed = None
full_df = None
DATASET = os.getenv("DATASET_PATH", "datas/Large set/large_dataset.csv")
full_df = pd.read_csv(DATASET)

# ─── Config images S3 ────────────────────────────────────────────────
BUCKET      = "ufc-fighters-img"                 
IMG_PREFIX  = "images/"                          
DEFAULT_KEY = f"{IMG_PREFIX}default-img.png"     
IMG_BASE    = f"https://{BUCKET}.s3.amazonaws.com/"
HAS_IMAGE = {
    "Ciryl Gane",
    "Amir Albazi",
    "Brandon Royval",
    "Brandon Moreno",
    "Joshua Van",
    "Alexandre Pantoja",
    "Song Yadong",
    "Petr Yan",
    "Umar Nurmagomedov",
    "Sean O Malley",
    "Merab Dvalishvili",
    "Arnold Allen",
    "Brian Ortega",
    "Yair Rodriguez",
    "Diego Lopes",
    "Movsar Evloev",
    "Alexander Volkanovski",
    "Dustin Poirier",
    "Max Holloway",
    "Justin Gaethje",
    "Charles Oliveira",
    "Arman Tsarukyan",
    "Ilia Topuria",
    "Kamaru Usman",
    "Leon Edwards",
    "Shavkat Rakhmonov",
    "Sean Brady",
    "Belal Muhammad",
    "Jack Della Maddalena",
    "Robert Whittaker",
    "Israel Adesanya",
    "Khamzat Chimaev",
    "Sean Strickland",
    "Nassourdine Imavov",
    "Dricus du Plessis",
    "Jan Blachowicz",
    "Khalil Rountree",
    "Carlos Ulberg",
    "Jiri Prochazka",
    "Alex Pereira",
    "Magomed Ankalaev",
    "Jailton Almeida",
    "Curtis Blaydes",
    "Sergei Pavlovich",
    "Alexander Volkov",
    "Tom Aspinall"
    
}
MIN_FIGHTS = 3
import boto3
import botocore 
from botocore import UNSIGNED
from botocore.client import Config

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

fight_counts = (
    full_df[["r_fighter", "b_fighter"]]
      .stack()
      .value_counts()
)

valid_names = set(fight_counts[fight_counts >= MIN_FIGHTS].index)

full_df = full_df[
    full_df["r_fighter"].isin(valid_names) &
    full_df["b_fighter"].isin(valid_names)
].reset_index(drop=True)

def slug(name: str) -> str:
    """'Tom Aspinall' → 'tom-aspinall'"""
    return (
        name.lower()
            .replace(" ", "-")
            .replace("'", "")
            .replace(".", "")
    )

def object_exists(key: str) -> bool:
    """Retourne True si l’objet <Bucket>/<Key> existe sur S3"""
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except botocore.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        return code not in ("404", "403", "NoSuchKey")

def img_url(name: str) -> str:
    key = f"{IMG_PREFIX}{slug(name)}.png"
    return f"{IMG_BASE}{key}"


def load_and_preprocess_data() -> pd.DataFrame:
    """Charge et prétraite le dataset UFC (logique Streamlit)"""
    print("📦 Chargement et préparation du dataset…")

    DATASET = os.getenv("DATASET_PATH", "datas/Large set/large_dataset.csv")
    fights_df = pd.read_csv(DATASET)

    mask = fights_df["r_reach"].notna() & fights_df["r_height"].notna()
    if mask.sum() >= 50:                                
        reg = LinearRegression().fit(
            fights_df.loc[mask, ["r_height"]], fights_df.loc[mask, "r_reach"]
        )
        a, b = reg.coef_[0], reg.intercept_
        print(f"📏 Reach ≈ {a:.3f} * height + {b:.1f}  (R²={reg.score(fights_df.loc[mask,['r_height']], fights_df.loc[mask,'r_reach']):.2f})")
    else:
        a, b = 1.02, 0.0                                

    for side in ["r", "b"]:
        h, r = f"{side}_height", f"{side}_reach"
        miss = fights_df[r].isna() & fights_df[h].notna()
        fights_df.loc[miss, r] = fights_df.loc[miss, h] * a + b

    fights_df.fillna({
        "reach_diff":  fights_df["reach_diff"].mean(),
        "age_diff":    fights_df["age_diff"].mean(),
        "height_diff": fights_df["height_diff"].mean(),
        "weight_diff": fights_df["weight_diff"].mean(),
        "r_age":  fights_df["r_age"].mean(),
        "b_age":  fights_df["b_age"].mean(),
        "r_reach":fights_df["r_reach"].mean(),
        "b_reach":fights_df["b_reach"].mean()
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

    print(f"[INFO] Dataset prêt : {len(fights_df)} lignes – {fights_df.shape[1]} features")
    return fights_df

def train_model_func():
    """Entraîne le modèle de prédiction"""
    global model, scaler, feature_cols, df_processed
    
    print("🧠 Entraînement du modèle...")
    df_processed = load_and_preprocess_data()
    
    X = df_processed.drop('winner_encoded', axis=1)
    y = df_processed['winner_encoded']
    feature_cols = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    inputs = Input(shape=(X_train_scaled.shape[1],))
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(zip(np.unique(y_train), weights))
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=200,
        batch_size=64,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1
    )
    
    val_accuracy = history.history['val_accuracy'][-1]
    print(f"🎯 Modèle entraîné - Accuracy validation : {val_accuracy:.2%}")
    
    return val_accuracy

def any_corner_stats(fighter: str) -> pd.Series:
    """Stats moyennes d’un combattant, qu’il ait été rouge OU bleu."""
    mask = (full_df["r_fighter"] == fighter) | (full_df["b_fighter"] == fighter)
    return (
        full_df.loc[mask]
               .select_dtypes("number")
               .mean()
               .fillna(0)                
    )

def weighted_merge(mean_a, n_a, mean_b, n_b):
    if n_a + n_b == 0:
        return pd.Series(dtype=float)   
    return (mean_a * n_a + mean_b * n_b) / (n_a + n_b)

def build_fight_features(red_fighter: str, blue_fighter: str) -> pd.DataFrame:
    """
    Tente d’abord de prendre les stats « dans le bon coin ».
    Si un combattant est totalement absent d’un coin,
    on bascule sur any_corner_stats(…).
    """
    def stats_for(fighter, corner):
        mask = full_df[f"{corner}_fighter"] == fighter
        return full_df.loc[mask].select_dtypes("number").mean()

    red_stats_r  = stats_for(red_fighter,  "r")
    red_stats_b  = stats_for(red_fighter,  "b")
    blue_stats_r = stats_for(blue_fighter, "r")
    blue_stats_b = stats_for(blue_fighter, "b")
    print(f"Stats A (rouge) : {red_stats_r.to_dict()}")
    
    # red_stats  = red_stats_r.add(red_stats_b, fill_value=0)
    # blue_stats = blue_stats_r.add(blue_stats_b, fill_value=0)

    red_stats  = red_stats_r.where(red_stats_b.isna(), (red_stats_r + red_stats_b) / 2)
    blue_stats = blue_stats_r.where(blue_stats_b.isna(), (blue_stats_r + blue_stats_b) / 2)


    if red_stats.isna().all():          
        red_stats = any_corner_stats(red_fighter)
    if blue_stats.isna().all():
        blue_stats = any_corner_stats(blue_fighter)

    feats = {}
    for col in feature_cols:
        if col.startswith("r_"):
            feats[col] = red_stats.get(col, 0)
        elif col.startswith("b_"):
            feats[col] = blue_stats.get(col, 0)
        else:
            feats[col] = (red_stats.get(col, 0) + blue_stats.get(col, 0)) / 2

    if "age_diff" in feature_cols:
        feats["age_diff"] = red_stats.get("r_age", 0) - blue_stats.get("b_age", 0)

    return pd.DataFrame([feats]).reindex(columns=feature_cols, fill_value=0)


@app.route('/train', methods=['POST'])
def train_endpoint():
    """Endpoint pour entraîner le modèle"""
    try:
        accuracy = train_model_func()
        return jsonify({
            'success': True,
            'message': 'Modèle entraîné avec succès',
            'accuracy': float(accuracy)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    global model, scaler, feature_cols
    try:
        data = request.json
        red  = data["red_fighter"]
        blue = data["blue_fighter"]
        print(f"🔍 Prédiction pour le combat : {red} vs {blue}")
        if not model:
            print("⚠️ Modèle non entraîné !")
            return jsonify({"error": "model not trained"}), 400

        # --- passe 1 : A rouge ---
        f1 = build_fight_features(red, blue)
        print(f"Stats A (rouge) : {f1.to_dict(orient='records')[0]}")
        f1_scaled = scaler.transform(f1.fillna(0))
        p1 = float(model.predict(f1_scaled, verbose=0)[0][0])  # proba rouge
        print(f"Proba A (rouge) : {p1:.2%} pour {red}")
        # --- passe 2 : B rouge ---
        f2 = build_fight_features(blue, red)
        f2_scaled = scaler.transform(f2.fillna(0))
        p2 = float(model.predict(f2_scaled, verbose=0)[0][0])  # proba rouge (de B)
        print(f"Proba B (rouge) : {p2:.2%} pour {blue}")
        # --- fusion symétrique ---
        p_red   = 0.5 * (p1 + (1 - p2))
        p_blue  = 1.0 - p_red
        winner  = red if p_red > 0.5 else blue
        conf    = max(p_red, p_blue)

        return jsonify({
            "red_fighter":  red,
            "blue_fighter": blue,
            "red_probability":  p_red,
            "blue_probability": p_blue,
            "predicted_winner": winner,
            "confidence": conf
        })

        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de santé"""
    return jsonify({
        'status': 'healthy',
        'model_trained': model is not None
    })

@app.get("/fighters")
def get_fighters():
    global full_df
    if full_df is None:
        full_df = pd.read_csv(DATASET)  
    gender       = request.args.get("gender")        # "M" | "F" | None
    weight_class = request.args.get("weight_class")  # ex. "Flyweight"…

    df = full_df
    if gender:
        df = df[df["gender"] == gender]
    if weight_class:
        df = df[df["weight_class"] == weight_class]

    names = (
        df[["r_fighter", "b_fighter"]]
        .stack()
        .dropna()
        .unique()
    )

    fighters = []
    for n in sorted(names):
        key = f"{IMG_PREFIX}{slug(n)}.png"
        has_img = n in HAS_IMAGE            
        fighters.append({
            "name": n,
            "has_img": has_img,
            "img": f"{IMG_BASE}{key}" if has_img else None
        })

    return jsonify({"fighters": fighters})

@app.get("/filters")
def list_filters():
    genders       = sorted(full_df["gender"].dropna().unique())
    weight_groups = sorted(full_df["weight_class"].dropna().unique())
    return jsonify({"genders": genders, "weight_classes": weight_groups})


if __name__ == '__main__':
    print("🚀 Démarrage du serveur UFC Predictor...")
    app.run(host='0.0.0.0', port=8000, debug=True)
