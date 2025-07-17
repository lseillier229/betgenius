#!/usr/bin/env python3
"""
Script pour démarrer le serveur Sports Predictor unifié (UFC + Tennis)
"""

import subprocess
import sys
import os

def install_requirements():
    """Installe les dépendances Python"""
    print("📦 Installation des dépendances...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])

def start_server():
    """Démarre le serveur Flask unifié"""
    print("Démarrage du serveur Sports Predictor...")
    print("UFC Predictor disponible sur port 8000")
    print("Tennis Predictor disponible sur port 8000/tennis")
    os.system("python unified_sports_predictor.py")

if __name__ == "__main__":
    try:
        install_requirements()
        start_server()
    except KeyboardInterrupt:
        print("\nArrêt du serveur...")
    except Exception as e:
        print(f"Erreur: {e}")