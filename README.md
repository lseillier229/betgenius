# UFC Fight Predictor

Une application web moderne pour prédire les résultats de combats UFC en utilisant l'intelligence artificielle.

## Fonctionnalités

- **Interface moderne** : Interface utilisateur responsive avec Next.js et Tailwind CSS
- **Prédictions IA** : Modèle de machine learning avec TensorFlow pour prédire les vainqueurs
- **API Backend** : API REST pour gérer les prédictions et l'entraînement du modèle
- **Temps réel** : Prédictions en temps réel avec affichage des probabilités

## 🛠Installation

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
.venv\Scripts\Activate.ps1
python .\scripts\unified_sports_predictor.py
\`\`\`

### 4. Démarrer l'application Next.js
\`\`\`bash
npm run dev
\`\`\`

## Utilisation

2. **Sélectionner les combattants** : Choisissez les combattants rouge et bleu
3. **Faire une prédiction** : Cliquez sur "Prédire le Vainqueur" pour obtenir les résultats


## 🔧 Configuration

Le serveur Python s'exécute sur le port 8000 par défaut.
L'application Next.js s'exécute sur le port 3000.

## 📈 Modèle IA

- **Architecture** : Réseau de neurones dense avec dropout
- **Features** : Âge, portée, taille, poids, stance, historique des combats
- **Optimisation** : Adam optimizer avec early stopping
- **Métriques** : Accuracy et loss de validation
