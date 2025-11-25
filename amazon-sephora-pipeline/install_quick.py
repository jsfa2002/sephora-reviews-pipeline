import subprocess
import sys
import os

def run_command(command):
    """Ejecuta un comando y muestra el resultado"""
    print(f"Ejecutando: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"Completado: {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error en {command}: {e}")
        return False

# Paquetes basicos esenciales
packages = [
    "pandas",
    "numpy", 
    "textblob",
    "vaderSentiment",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "plotly",
    "streamlit",
    "pyyaml",
    "scipy"
]

print("Instalando paquetes esenciales...")
for package in packages:
    run_command(f"pip install {package}")

print("\nCreando directorios necesarios...")
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("\nInstalacion completada! Ahora puedes ejecutar:")
print("python src/orchestrator.py")
print("streamlit run src/dashboard.py")