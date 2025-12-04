import pandas as pd
import yaml
import logging
import os

# Cargar configuracion
with open("config/pipeline_config.yaml") as f:
    config = yaml.safe_load(f)

logging.basicConfig(
    filename=config['logging']['log_file'],
    level=getattr(logging, config['logging']['level']),
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def validate_data(df):
    """Validaciones basicas de calidad de datos"""
    logging.info("Iniciando validacion de datos...")

    # Lista de resultados
    validation_results = []

    # 1. Validar que las columnas requeridas existan
    required_columns = ["review_text", "rating", "sentiment_score", "sentiment_label"]
    for col in required_columns:
        if col in df.columns:
            validation_results.append(f"Columna {col} presente: SI")
        else:
            validation_results.append(f"Columna {col} presente: NO")
            logging.warning(f"Columna faltante: {col}")

    # 2. Validar que no haya valores nulos en columnas criticas
    critical_columns = [col for col in required_columns if col in df.columns]
    for col in critical_columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            validation_results.append(f"Columna {col} tiene {null_count} valores nulos")
            logging.warning(f"Columna {col} tiene {null_count} valores nulos")
        else:
            validation_results.append(f"Columna {col} sin nulos: SI")

    # 3. Validar rango de rating
    if "rating" in df.columns:
        if not df["rating"].between(1, 5).all():
            invalid_ratings = df[~df["rating"].between(1, 5)]["rating"].unique()
            validation_results.append(f"Ratings fuera de rango: {invalid_ratings}")
            logging.warning(f"Se encontraron ratings fuera del rango [1,5]: {invalid_ratings}")
        else:
            validation_results.append("Todos los ratings en rango [1,5]: SI")

    # 4. Validar rango de sentimiento
    if "sentiment_score" in df.columns:
        if not df["sentiment_score"].between(-1, 1).all():
            invalid_sentiment = df[~df["sentiment_score"].between(-1, 1)]["sentiment_score"].unique()
            validation_results.append(f"Sentimientos fuera de rango: {invalid_sentiment}")
            logging.warning(f"Se encontraron sentimientos fuera del rango [-1,1]: {invalid_sentiment}")
        else:
            validation_results.append("Todos los sentimientos en rango [-1,1]: SI")

    # 5. Validar que las etiquetas de sentimiento sean consistentes
    if "sentiment_label" in df.columns:
        valid_labels = ['positive', 'negative', 'neutral']
        invalid_labels = df[~df['sentiment_label'].isin(valid_labels)]['sentiment_label'].unique()
        if len(invalid_labels) > 0:
            validation_results.append(f"Etiquetas de sentimiento invalidas: {invalid_labels}")
            logging.warning(f"Etiquetas de sentimiento invalidas: {invalid_labels}")
        else:
            validation_results.append("Etiquetas de sentimiento validas: SI")

    # 6. Validar que los textos no esten vacios
    if "review_text" in df.columns:
        empty_texts = df['review_text'].isna().sum()
        if empty_texts > 0:
            validation_results.append(f"Hay {empty_texts} textos de reseña vacios")
            logging.warning(f"Hay {empty_texts} textos de reseña vacios")
        else:
            validation_results.append("No hay textos vacios: SI")

    # 7. Validar que los textos tengan al menos 10 caracteres (si no estan vacios)
    if "review_text" in df.columns:
        short_texts = df[df['review_text'].str.len() < 10].shape[0]
        if short_texts > 0:
            validation_results.append(f"Hay {short_texts} textos con menos de 10 caracteres")
            logging.warning(f"Hay {short_texts} textos con menos de 10 caracteres")
        else:
            validation_results.append("Todos los textos tienen al menos 10 caracteres: SI")

    # 8. Validar que los clusters sean consistentes (si existen)
    if "user_cluster" in df.columns:
        cluster_counts = df['user_cluster'].value_counts()
        validation_results.append(f"Distribucion de clusters: {cluster_counts.to_dict()}")
        logging.info(f"Distribucion de clusters: {cluster_counts.to_dict()}")

    # 9. Metricas estadisticas basicas
    if "rating" in df.columns:
        rating_stats = {
            'mean': df['rating'].mean(),
            'std': df['rating'].std(),
            'min': df['rating'].min(),
            'max': df['rating'].max()
        }
        validation_results.append(f"Estadisticas de rating: {rating_stats}")

    if "sentiment_score" in df.columns:
        sentiment_stats = {
            'mean': df['sentiment_score'].mean(),
            'std': df['sentiment_score'].std(),
            'min': df['sentiment_score'].min(),
            'max': df['sentiment_score'].max()
        }
        validation_results.append(f"Estadisticas de sentimiento: {sentiment_stats}")

    logging.info("Validacion completada correctamente.")
    return validation_results

def generate_validation_report(validation_results, output_path="data/processed/validation_report.txt"):
    """Genera un reporte de validacion"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Reporte de Validacion de Datos\n")
        f.write("==============================\n\n")
        for result in validation_results:
            f.write(f"{result}\n")
    
    logging.info(f"Reporte de validacion guardado en {output_path}")

if __name__ == "__main__":
    processed_path = config["dataset"]["processed_path"]
    if os.path.exists(processed_path):
        df = pd.read_parquet(processed_path)
        results = validate_data(df)
        generate_validation_report(results)
        print("Validacion completada con exito. Ver reporte en data/processed/validation_report.txt")
    else:
        print("No se encontro el dataset procesado. Ejecuta primero la transformacion.")
        logging.error("No se encontro el dataset procesado para validacion.")