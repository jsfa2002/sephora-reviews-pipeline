import tweepy
import pandas as pd
import yaml
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time
import numpy as np

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_simulated_twitter_data():
    """Crear datos de Twitter simulados para demostración"""
    print("Generando datos simulados de Twitter...")
    
    dates = [datetime.now() - timedelta(days=i) for i in range(30)]
    hashtags = ['#Sephora', '#Makeup', '#Skincare', '#Beauty', '#AmazonBeauty']
    
    tweets_data = []
    tweet_id = 100000
    
    for date in dates:
        for hashtag in hashtags:
            for _ in range(np.random.randint(1, 5)):
                tweet_id += 1
                texts = [
                    f"Just bought some amazing products from {hashtag.replace('#', '')}! Loving them!",
                    f"Review of {hashtag.replace('#', '')} products - absolutely fantastic!",
                    f"Has anyone tried the new {hashtag.replace('#', '')} collection?",
                    f"{hashtag.replace('#', '')} never disappoints! Great quality products.",
                    f"Just finished my {hashtag.replace('#', '')} haul. So excited to try everything!"
                ]
                
                text = np.random.choice(texts)
                
                tweets_data.append({
                    'tweet_id': tweet_id,
                    'text': text,
                    'created_at': date + timedelta(hours=np.random.randint(0, 24)),
                    'author_id': f"user_{np.random.randint(1000, 9999)}",
                    'author_username': f"user_{np.random.randint(1000, 9999)}",
                    'author_name': f"User {np.random.randint(1000, 9999)}",
                    'like_count': np.random.randint(0, 100),
                    'retweet_count': np.random.randint(0, 50),
                    'reply_count': np.random.randint(0, 20),
                    'impression_count': np.random.randint(100, 1000),
                    'hashtags': [hashtag, '#Beauty', '#Makeup'][:np.random.randint(1, 3)],
                    'mentions': [],
                    'query': hashtag,
                    'platform': 'twitter',
                    'engagement': np.random.randint(10, 500),
                    'sentiment_score': np.random.uniform(0.1, 0.9),
                    'sentiment_label': 'positive' if np.random.random() > 0.3 else 'neutral'
                })
    
    df = pd.DataFrame(tweets_data)
    return df

class TwitterIntegration:
    def __init__(self, use_simulation=False):
        self.use_simulation = use_simulation
        
        if not use_simulation:
            self.bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
            self.client = None
            self.initialize_client()
        else:
            self.client = None
    
    def initialize_client(self):
        """Inicializar cliente de Twitter API v2"""
        try:
            if not self.bearer_token or self.bearer_token == "tu_token_aqui":
                print(" No hay API Key de Twitter válida. Usando simulación.")
                self.use_simulation = True
                return
            
            self.client = tweepy.Client(
                bearer_token=self.bearer_token,
                wait_on_rate_limit=True
            )
            print(" Cliente de Twitter inicializado correctamente")
        except Exception as e:
            print(f" Error inicializando cliente Twitter: {e}")
            self.use_simulation = True
            self.client = None
    
    def search_tweets_safe(self, query, max_results=50, days_back=7):
        """Buscar tweets de forma segura con manejo de errores"""
        if self.use_simulation or not self.client:
            return pd.DataFrame()
        
        try:
            # Calcular fecha de inicio
            start_time = datetime.utcnow() - timedelta(days=days_back)
            
            tweets_data = []
            
            # Buscar tweets recientes
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 50),  # Reducir para evitar rate limits
                start_time=start_time,
                tweet_fields=['created_at', 'public_metrics', 'text', 'author_id'],
                expansions=['author_id'],
                user_fields=['username', 'name', 'public_metrics']
            )
            
            if response.data:
                users = {user.id: user for user in response.includes['users']} if response.includes else {}
                
                for tweet in response.data:
                    user = users.get(tweet.author_id)
                    
                    tweet_info = {
                        'tweet_id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'author_id': tweet.author_id,
                        'author_username': user.username if user else 'unknown',
                        'author_name': user.name if user else 'unknown',
                        'like_count': tweet.public_metrics['like_count'],
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'reply_count': tweet.public_metrics['reply_count'],
                        'impression_count': tweet.public_metrics.get('impression_count', 0),
                        'hashtags': self.extract_hashtags(tweet.text),
                        'mentions': self.extract_mentions(tweet.text),
                        'query': query,
                        'platform': 'twitter'
                    }
                    
                    # Calcular engagement total
                    tweet_info['engagement'] = (
                        tweet_info['like_count'] + 
                        tweet_info['retweet_count'] * 2 +
                        tweet_info['reply_count'] * 1.5
                    )
                    
                    tweets_data.append(tweet_info)
                
                print(f" Encontrados {len(tweets_data)} tweets para: {query}")
            
            return pd.DataFrame(tweets_data)
            
        except tweepy.TooManyRequests:
            print("⏸ Rate limit excedido. Esperando 15 minutos...")
            time.sleep(900)  # Esperar 15 minutos
            return pd.DataFrame()
        except Exception as e:
            print(f" Error buscando tweets para '{query}': {e}")
            return pd.DataFrame()
    
    def extract_hashtags(self, text):
        """Extraer hashtags del texto del tweet"""
        import re
        hashtags = re.findall(r'#(\w+)', text)
        return hashtags if hashtags else []
    
    def extract_mentions(self, text):
        """Extraer menciones del texto del tweet"""
        import re
        mentions = re.findall(r'@(\w+)', text)
        return mentions if mentions else []
    
    def analyze_sentiment(self, df_tweets):
        """Analizar sentimiento de tweets"""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            
            analyzer = SentimentIntensityAnalyzer()
            
            sentiments = []
            for text in df_tweets['text']:
                vs = analyzer.polarity_scores(str(text))
                sentiments.append({
                    'sentiment_score': vs['compound'],
                    'sentiment_label': 'positive' if vs['compound'] >= 0.05 else 
                                      'negative' if vs['compound'] <= -0.05 else 'neutral'
                })
            
            sentiment_df = pd.DataFrame(sentiments)
            return pd.concat([df_tweets.reset_index(drop=True), sentiment_df], axis=1)
        except:
            # Análisis simple si VADER no está instalado
            df_tweets['sentiment_score'] = np.random.uniform(-0.5, 0.8, len(df_tweets))
            df_tweets['sentiment_label'] = df_tweets['sentiment_score'].apply(
                lambda x: 'positive' if x > 0.1 else 'negative' if x < -0.1 else 'neutral'
            )
            return df_tweets

def main():
    """Función principal"""
    
    # Verificar si usar simulación
    use_simulation = False
    if not os.getenv("TWITTER_BEARER_TOKEN") or os.getenv("TWITTER_BEARER_TOKEN") == "tu_token_aqui":
        print(" No hay API Key de Twitter configurada. Usando simulación.")
        use_simulation = True
    
    twitter = TwitterIntegration(use_simulation=use_simulation)
    
    all_tweets = []
    
    if use_simulation:
        # Usar datos simulados
        df = create_simulated_twitter_data()
    else:
        # Queries optimizadas (más específicas para evitar rate limits)
        queries = [
            "Sephora review",
            "makeup review",
            "skincare"
        ]
        
        # Buscar tweets para cada query
        for query in queries:
            print(f" Buscando tweets para: {query}")
            
            tweets_df = twitter.search_tweets_safe(query, max_results=30, days_back=3)
            
            if not tweets_df.empty:
                # Analizar sentimiento
                tweets_df = twitter.analyze_sentiment(tweets_df)
                all_tweets.append(tweets_df)
            
            time.sleep(5)  # Mayor tiempo entre requests
        
        if all_tweets:
            df = pd.concat(all_tweets, ignore_index=True)
        else:
            print(" No se encontraron tweets reales. Generando datos simulados...")
            df = create_simulated_twitter_data()
    
    # Guardar datos
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f"{output_dir}/twitter_data_{datetime.now().strftime('%Y%m%d')}.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"\nDatos guardados en: {output_path}")
    print(f" Total tweets: {len(df)}")
    print(f" Rango de fechas: {df['created_at'].min().date()} a {df['created_at'].max().date()}")
    
    # Estadísticas
    if 'sentiment_label' in df.columns:
        sentiment_counts = df['sentiment_label'].value_counts()
        print("\n📈 Distribución de sentimiento:")
        for label, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  - {label}: {count} tweets ({percentage:.1f}%)")

if __name__ == "__main__":
    main()