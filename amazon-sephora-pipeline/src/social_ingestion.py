# src/social_ingestion.py
import pandas as pd
import yaml
import logging
import os
from datetime import datetime, timedelta
import numpy as np

# Cargar configuracion
with open("config/pipeline_config.yaml") as f:
    config = yaml.safe_load(f)

# Configurar logging
logging.basicConfig(
    filename=config['logging']['log_file'],
    level=getattr(logging, config['logging']['level']),
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_realistic_social_data(reviews_df, hashtags):
    """Crea datos sociales realistas basados en las tendencias de reviews"""
    logging.info("Creando datos sociales realistas basados en tendencias de reviews...")
    
    # Asegurar que la columna de fecha es datetime
    reviews_df['review_date'] = pd.to_datetime(reviews_df['review_date'])
    
    # Agrupar reviews por fecha para obtener tendencias reales
    daily_reviews = reviews_df.groupby('review_date').agg({
        'review_text': 'count',
        'sentiment_score': 'mean',
        'rating': 'mean'
    }).reset_index()
    daily_reviews.columns = ['date', 'review_count', 'avg_sentiment', 'avg_rating']
    
    social_data = []
    
    # Definir factores por plataforma
    platform_factors = {
        'tiktok': {'volume_factor': 0.3, 'engagement_factor': 0.05, 'volatility': 0.4},
        'instagram': {'volume_factor': 0.2, 'engagement_factor': 0.03, 'volatility': 0.3},
        'youtube': {'volume_factor': 0.1, 'engagement_factor': 0.02, 'volatility': 0.2}
    }
    
    for platform, factors in platform_factors.items():
        for hashtag in hashtags:
            # Cada hashtag tiene una popularidad base diferente
            hashtag_popularity = np.random.uniform(0.5, 2.0)
            
            for _, row in daily_reviews.iterrows():
                # Base del volumen: correlacionado con el numero de reviews y el sentimiento
                base_volume = row['review_count'] * factors['volume_factor'] * hashtag_popularity
                
                # Añadir variabilidad y tendencia estacional
                day_of_week = row['date'].dayofweek
                seasonal_factor = 1 + 0.1 * np.sin(day_of_week * 2 * np.pi / 7)
                
                # Añadir algo de ruido
                noise = np.random.normal(1, factors['volatility'])
                
                mention_volume = max(10, int(base_volume * seasonal_factor * noise))
                
                # Engagement basado en el volumen y el sentimiento
                engagement_factor = factors['engagement_factor'] * (1 + row['avg_sentiment'])
                engagement = max(1, int(mention_volume * engagement_factor))
                
                social_data.append({
                    'date': row['date'],
                    'hashtag': f"#{hashtag}",
                    'mention_volume': mention_volume,
                    'engagement': engagement,
                    'platform': platform
                })
    
    return pd.DataFrame(social_data)

def save_social_data(df):
    """Guarda los datos sociales combinados"""
    output_path = config['dataset']['social_data_path']
    df.to_parquet(output_path, index=False)
    logging.info(f"Datos sociales guardados en {output_path}")

if __name__ == "__main__":
    try:
        logging.info("=== INICIANDO INGESTA DE REDES SOCIALES ===")
        
        # Cargar los datos de reviews para obtener el rango temporal
        reviews_path = config["dataset"]["processed_path"]
        reviews_df = pd.read_parquet(reviews_path)
        
        # Obtener hashtags de la configuracion
        hashtags = config['social_media']['tiktok_hashtags'] + config['social_media']['instagram_hashtags']
        hashtags = list(set(hashtags))  # Eliminar duplicados
        
        # Crear datos sociales realistas
        social_df = create_realistic_social_data(reviews_df, hashtags)
        
        # Guardar
        save_social_data(social_df)
        
        print("Ingesta de redes sociales completada exitosamente!")
        print(f"Hashtags monitoreados: {len(hashtags)}")
        print(f"Total de registros: {len(social_df)}")
        print(f"Rango de fechas: {social_df['date'].min()} a {social_df['date'].max()}")
        
    except Exception as e:
        logging.error(f"Error en ingesta de redes sociales: {str(e)}")
        print(f"Error en ingesta de redes sociales: {str(e)}")