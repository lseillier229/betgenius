#!/usr/bin/env python3
"""
Script pour démarrer le serveur Python UFC Predictor
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
    """Démarre le serveur Flask"""
    print("🚀 Démarrage du serveur UFC Predictor...")
    os.system("python ufc_predictor.py")

if __name__ == "__main__":
    try:
        install_requirements()
        start_server()
    except KeyboardInterrupt:
        print("\n👋 Arrêt du serveur...")
    except Exception as e:
        print(f"❌ Erreur: {e}")
