import pandas as pd
import yaml
import logging
import os
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Importar las clases necesarias de Twitter (asumiendo que están en real_social_api)
# Para simplificar, copiaremos las clases de Twitter aquí, o puedes importarlas si usas módulos.
# Si quieres mantener las clases de Twitter separadas, avísame. Por ahora, las incluiremos.

# Cargar variables de entorno
load_dotenv()

# Cargar configuracion
with open("config/pipeline_config.yaml") as f:
    config = yaml.safe_load(f)

# Configurar logging
logging.basicConfig(
    filename=config['logging']['log_file'],
    level=getattr(logging, config['logging']['level']),
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ===== CLASE TWITTER COPIADA DE REAL_SOCIAL_API.PY (Necesaria para datos reales) =====
class TwitterCollector:
    """Recolector de datos de Twitter/X usando API oficial"""
    # *** Copia aquí la clase TwitterCollector COMPLETA de real_social_api.py ***
    
    def __init__(self):
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.enabled = False

        if not self.bearer_token:
            logging.warning("Twitter Bearer Token no configurado")
        else:
            try:
                import tweepy
                self.client = tweepy.Client(bearer_token=self.bearer_token)
                self.enabled = True
                logging.info("Twitter API inicializada correctamente")
            except Exception as e:
                logging.error(f"Error inicializando Twitter API: {e}")
                
    def search_tweets(self, query, max_results=100):
        # ... (Mantener la lógica search_tweets de real_social_api.py) ...
        if not self.enabled:
            logging.warning("Twitter API no habilitada")
            return []

        try:
            import tweepy
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=['created_at', 'public_metrics', 'author_id', 'lang'],
                expansions=['author_id'],
                user_fields=['username', 'verified']
            )

            results = []
            if tweets.data:
                for tweet in tweets.data:
                    results.append({
                        'id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'likes': tweet.public_metrics['like_count'],
                        'retweets': tweet.public_metrics['retweet_count'],
                        'replies': tweet.public_metrics['reply_count'],
                        'impressions': tweet.public_metrics.get('impression_count', 0),
                        'lang': tweet.lang
                    })

                logging.info(f"Twitter: {len(results)} tweets recolectados para '{query}'")
            return results
        except Exception as e:
            logging.error(f"Error buscando tweets: {e}")
            return []

    def collect_by_hashtags(self, hashtags, max_per_hashtag=50):
        all_tweets = []
        for hashtag in hashtags:
            query = f"#{hashtag} -is:retweet lang:en"
            tweets = self.search_tweets(query, max_results=max_per_hashtag)
            for tweet in tweets:
                tweet['hashtag'] = f"#{hashtag}"
                tweet['platform'] = 'twitter_real' # Etiqueta diferente para distinguir
                all_tweets.append(tweet)
        return pd.DataFrame(all_tweets)


# ===== FUNCIÓN DE SIMULACIÓN (DE social_ingestion.py) =====
def create_simulated_social_data(reviews_df, hashtags):
    """Crea datos sociales simulados basados en las tendencias de reviews"""
    logging.info("Creando datos sociales simulados basados en tendencias de reviews...")

    reviews_df['review_date'] = pd.to_datetime(reviews_df['review_date'])
    daily_reviews = reviews_df.groupby('review_date').agg({
        'review_text': 'count',
        'sentiment_score': 'mean',
        'rating': 'mean'
    }).reset_index()
    daily_reviews.columns = ['date', 'review_count', 'avg_sentiment', 'avg_rating']

    social_data = []

    platform_factors = {
        'tiktok_simulated': {'volume_factor': 0.3, 'engagement_factor': 0.05, 'volatility': 0.4},
        'instagram_simulated': {'volume_factor': 0.2, 'engagement_factor': 0.03, 'volatility': 0.3},
        'twitter_simulated': {'volume_factor': 0.1, 'engagement_factor': 0.02, 'volatility': 0.2} # Twitter simulado
    }

    for platform, factors in platform_factors.items():
        for hashtag in hashtags:
            hashtag_popularity = np.random.uniform(0.5, 2.0)
            
            for _, row in daily_reviews.iterrows():
                base_volume = row['review_count'] * factors['volume_factor'] * hashtag_popularity
                day_of_week = row['date'].dayofweek
                seasonal_factor = 1 + 0.1 * np.sin(day_of_week * 2 * np.pi / 7)
                noise = np.random.normal(1, factors['volatility'])
                
                mention_volume = max(10, int(base_volume * seasonal_factor * noise))
                
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


# ===== FUNCIÓN PRINCIPAL HÍBRIDA =====
def collect_hybrid_social_data():
    """Recolecta datos reales de Twitter y simula el resto de plataformas."""
    logging.info("=== INICIANDO INGESTA SOCIAL HÍBRIDA (Real + Simulada) ===")

    # Cargar datos de reviews para el rango temporal de la simulación
    reviews_path = config["dataset"]["processed_path"]
    try:
        reviews_df = pd.read_parquet(reviews_path)
    except FileNotFoundError:
        logging.error("No se encontró el dataset procesado. Ejecuta la transformación primero.")
        print("Error: No se encontró el dataset procesado. Ejecuta la transformación.")
        return None

    # Obtener hashtags
    hashtags = config['social_media']['tiktok_hashtags'] + config['social_media']['instagram_hashtags']
    hashtags = list(set([h.lower() for h in hashtags]))

    all_data_dfs = []

    # 1. Recolección REAL de Twitter
    twitter = TwitterCollector()
    if twitter.enabled:
        try:
            twitter_real_df = twitter.collect_by_hashtags(hashtags, max_per_hashtag=50)
            if not twitter_real_df.empty:
                # Transformar a formato común
                twitter_real_df['mention_volume'] = 1
                twitter_real_df['engagement'] = twitter_real_df['likes'] + twitter_real_df['retweets'] + twitter_real_df['replies']
                twitter_real_df['date'] = pd.to_datetime(twitter_real_df['created_at']).dt.date
                
                all_data_dfs.append(twitter_real_df[['date', 'hashtag', 'mention_volume', 'engagement', 'platform']])
                logging.info(f"Twitter REAL: {len(twitter_real_df)} registros agregados")
            else:
                logging.warning("Twitter REAL no devolvió registros. Se usará simulación para Twitter.")
        except Exception as e:
            logging.error(f"Error recolectando Twitter REAL: {e}")
            logging.warning("Se usará simulación para Twitter.")
    
    # 2. Recolección SIMULADA (incluye Twitter simulado, TikTok, Instagram)
    simulated_df = create_simulated_social_data(reviews_df, hashtags)
    
    # 3. Combinación (Priorizar los datos reales de Twitter)
    
    if all_data_dfs:
        # Si se recolectaron datos REALES de Twitter
        real_twitter_df = all_data_dfs[0]
        
        # Eliminar los datos simulados de Twitter del DF simulado
        simulated_df_filtered = simulated_df[simulated_df['platform'] != 'twitter_simulated'].copy()
        
        # Combinar: Datos Simulados (sin Twitter) + Datos Reales de Twitter
        final_df = pd.concat([simulated_df_filtered, real_twitter_df[['date', 'hashtag', 'mention_volume', 'engagement', 'platform']]], ignore_index=True)
    else:
        # Si Twitter REAL falló, usar solo los datos SIMULADOS (incluyendo el simulado de Twitter)
        final_df = simulated_df

    # 4. Guardar
    output_path = config['dataset']['social_data_path']
    final_df.to_parquet(output_path, index=False)
    logging.info(f"Datos sociales híbridos guardados: {len(final_df)} registros en {output_path}")

    print(" Ingesta Social Híbrida completada!")
    print(f"  - Total de registros: {len(final_df)}")
    print(f"  - Plataformas presentes: {final_df['platform'].unique().tolist()}")
    
    return final_df

if __name__ == "__main__":
    # Asegúrate de que las dependencias necesarias (tweepy) estén instaladas.
    try:
        import tweepy
    except ImportError:
        print("¡ADVERTENCIA! Parece que la librería 'tweepy' no está instalada.")
        print("Para que la recolección REAL de Twitter funcione, ejecuta: pip install tweepy")
        
    collect_hybrid_social_data()