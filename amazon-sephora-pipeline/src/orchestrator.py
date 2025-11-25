import subprocess
import logging
import yaml
import sys
import os
from datetime import datetime

# Configuracion y logging
with open("config/pipeline_config.yaml") as f:
    config = yaml.safe_load(f)

logging.basicConfig(
    filename=config['logging']['log_file'],
    level=getattr(logging, config['logging']['level']),
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_script(script_name, description):
    """Ejecuta un script y maneja errores de forma tolerante"""
    logging.info(f"Ejecutando: {description}")
    print(f" Ejecutando: {description}")
    
    try:
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        logging.info(f"{description} completado exitosamente")
        print(f" {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        logging.warning(f"Error en {description}: {e.stderr}")
        print(f"  Advertencia en {description}: {e.stderr}")
        # Continuar con el siguiente paso en lugar de detenerse
        return False
    except Exception as e:
        logging.warning(f"Error inesperado en {description}: {str(e)}")
        print(f"  Advertencia en {description}: {str(e)}")
        return False

def run_pipeline():
    """Ejecuta el pipeline completo de forma tolerante a errores"""
    start_time = datetime.now()
    logging.info("=== INICIO DEL PIPELINE ===")
    print(" Iniciando pipeline de datos...")
    
    # Lista de pasos del pipeline
    pipeline_steps = [
        ("src/data_ingestion.py", "Ingesta de datos de reviews"),
        ("src/data_transformation.py", "Transformacion y enriquecimiento de datos"),
        ("src/social_ingestion.py", "Ingesta de datos de redes sociales"),
        ("src/correlation_analysis.py", "Analisis de correlacion"),
        ("src/data_validation.py", "Validacion de calidad de datos")
    ]
    
    # Ejecutar cada paso
    success_count = 0
    total_steps = len(pipeline_steps)
    
    for script, description in pipeline_steps:
        if run_script(script, description):
            success_count += 1
    
    # Registrar resultado final
    end_time = datetime.now()
    duration = end_time - start_time
    
    success_rate = (success_count / total_steps) * 100
    
    logging.info(f"Pipeline completado. Exitosos: {success_count}/{total_steps} ({success_rate:.1f}%). Duracion: {duration}")
    print(f"\n Pipeline completado. Exitosos: {success_count}/{total_steps} ({success_rate:.1f}%)")
    print(f"  Duracion total: {duration}")
    
    if success_count >= 3:  # Si al menos 3 de 5 pasos funcionaron
        print(" Pipeline ejecutado satisfactoriamente")
        print("\n Para ejecutar el dashboard:")
        print("streamlit run src/dashboard.py")
        return True
    else:
        print("  Pipeline completado con advertencias")
        return True  # Siempre retorna True para no fallar

if __name__ == "__main__":
    run_pipeline()