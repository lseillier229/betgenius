# UFC Fight Predictor

Une application web moderne pour prédire les résultats de combats UFC en utilisant l'intelligence artificielle.

## 🚀 Fonctionnalités

- **Interface moderne** : Interface utilisateur responsive avec Next.js et Tailwind CSS
- **Prédictions IA** : Modèle de machine learning avec TensorFlow pour prédire les vainqueurs
- **API Backend** : API REST pour gérer les prédictions et l'entraînement du modèle
- **Temps réel** : Prédictions en temps réel avec affichage des probabilités

## 🛠️ Installation

### 1. Installer les dépendances Node.js
\`\`\`bash
npm install
\`\`\`

### 2. Installer les dépendances Python
\`\`\`bash
cd scripts
pip install -r requirements.txt
\`\`\`

### 3. Démarrer le serveur Python
\`\`\`bash
npm run python-server
\`\`\`

### 4. Démarrer l'application Next.js
\`\`\`bash
npm run dev
\`\`\`

## 📊 Utilisation

1. **Entraîner le modèle** : Cliquez sur "Entraîner le Modèle" pour préparer l'IA
2. **Sélectionner les combattants** : Choisissez les combattants rouge et bleu
3. **Faire une prédiction** : Cliquez sur "Prédire le Vainqueur" pour obtenir les résultats

## 🏗️ Architecture

- **Frontend** : Next.js 14 avec TypeScript et Tailwind CSS
- **Backend** : API Routes Next.js + Serveur Flask Python
- **IA** : TensorFlow avec réseau de neurones dense
- **Données** : Preprocessing avec pandas et scikit-learn

## 📁 Structure du Projet

\`\`\`
ufc-predictor/
├── app/                    # Application Next.js
│   ├── api/               # API Routes
│   └── page.tsx           # Page principale
├── scripts/               # Scripts Python
│   ├── ufc_predictor.py   # Serveur Flask + ML
│   └── requirements.txt   # Dépendances Python
└── components/            # Composants UI
\`\`\`

## 🔧 Configuration

Le serveur Python s'exécute sur le port 8000 par défaut.
L'application Next.js s'exécute sur le port 3000.

## 📈 Modèle IA

- **Architecture** : Réseau de neurones dense avec dropout
- **Features** : Âge, portée, taille, poids, stance, historique des combats
- **Optimisation** : Adam optimizer avec early stopping
- **Métriques** : Accuracy et loss de validation
