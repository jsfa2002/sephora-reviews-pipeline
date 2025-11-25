import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yaml
import logging
import os
import re
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Cargar configuración
with open("config/pipeline_config.yaml") as f:
    config = yaml.safe_load(f)

# Configurar directorios y logging
os.makedirs("data/processed", exist_ok=True)
logging.basicConfig(
    filename=config['logging']['log_file'],
    level=getattr(logging, config['logging']['level']),
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataTransformer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def clean_text(self, text):
        """Limpieza avanzada de texto sin spaCy"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Eliminar URLs
        text = re.sub(r'http\S+', '', text)
        # Eliminar caracteres especiales pero mantener emojis
        text = re.sub(r'[^\w\s@#]', ' ', text)
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_features(self, text):
        """Extrae características del texto sin spaCy"""
        if pd.isna(text) or text == "":
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'exclamation_count': 0,
                'question_count': 0
            }
        
        # Métodos simples sin spaCy
        words = text.split()
        word_count = len(words)
        
        # Contar oraciones aproximadas
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        if sentence_count == 0 and word_count > 0:
            sentence_count = 1
            
        avg_word_length = np.mean([len(word) for word in words]) if word_count > 0 else 0
        
        features = {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?')
        }
        return features

def clean_data(df):
    """Limpieza completa de los datos"""
    logging.info("Iniciando limpieza de datos...")
    
    transformer = DataTransformer()
    
    # Identificar columnas dinámicamente
    text_col = next((c for c in df.columns if "review_text" in c.lower() or "text" in c.lower()), None)
    rating_col = next((c for c in df.columns if "rating" in c.lower()), None)
    date_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None)

    if not text_col or not rating_col:
        available_cols = ", ".join(df.columns)
        raise KeyError(f"Columnas requeridas no encontradas. Columnas disponibles: {available_cols}")

    # Renombrar columnas para estandarizar
    column_mapping = {
        text_col: "review_text",
        rating_col: "rating"
    }
    if date_col:
        column_mapping[date_col] = "review_date"
    
    df = df.rename(columns=column_mapping)
    
    # Limpiar texto
    df['cleaned_text'] = df['review_text'].apply(transformer.clean_text)
    
    # Eliminar reseñas sin texto o sin rating
    initial_count = len(df)
    df = df.dropna(subset=['cleaned_text', 'rating'])
    df = df[df['cleaned_text'].str.len() > 5]  # Reducir mínimo de caracteres
    cleaned_count = len(df)
    logging.info(f"Reseñas después de limpieza: {cleaned_count}/{initial_count} ({cleaned_count/initial_count*100:.1f}%)")
    
    # Convertir y validar rating
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df[df['rating'].between(1, 5)]
    df['rating'] = df['rating'].round(1)
    
    # Procesar fecha
    if 'review_date' in df.columns:
        df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
        df = df.dropna(subset=['review_date'])
    
    # Limpiar datos demográficos
    demographic_cols = ['skin_tone', 'eye_color', 'skin_type', 'hair_color']
    for col in demographic_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').str.lower().str.strip()
    
    logging.info(f"Datos limpios: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df

def create_diverse_categories(df):
    """Crea categorías diversas basadas en el contenido de las reviews"""
    logging.info("Creando categorías diversas...")
    
    # Categorías basadas en keywords en el texto
    category_keywords = {
        'Skincare': ['skin', 'face', 'moisturizer', 'serum', 'cream', 'cleanser', 'acne', 'wrinkle'],
        'Makeup': ['makeup', 'lipstick', 'foundation', 'mascara', 'eyeshadow', 'blush', 'concealer'],
        'Hair Care': ['hair', 'shampoo', 'conditioner', 'styling', 'color', 'treatment'],
        'Fragrance': ['perfume', 'fragrance', 'scent', 'cologne', 'smell'],
        'Body Care': ['body', 'lotion', 'bath', 'shower', 'hand', 'foot'],
        'Sunscreen': ['sunscreen', 'sunblock', 'spf', 'uv', 'protection'],
        'Anti-Aging': ['anti-aging', 'wrinkle', 'firming', 'age', 'mature'],
        'Natural': ['natural', 'organic', 'clean', 'chemical-free', 'vegan']
    }
    
    def categorize_text(text):
        text_lower = str(text).lower()
        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        return 'Other'
    
    # Aplicar categorización
    df['primary_category'] = df['cleaned_text'].apply(categorize_text)
    
    # Si hay marca, usar como categoría secundaria
    if 'brand_name' in df.columns:
        df['secondary_category'] = df['brand_name']
    else:
        df['secondary_category'] = 'General'
    
    logging.info(f"Distribución de categorías: {df['primary_category'].value_counts().to_dict()}")
    return df

def advanced_sentiment_analysis(df):
    """Análisis de sentimiento usando VADER y TextBlob"""
    logging.info("Realizando análisis de sentimiento...")
    
    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        try:
            # Usar VADER para análisis de sentimiento
            vader_score = analyzer.polarity_scores(str(text))['compound']
            # Usar TextBlob como respaldo
            textblob_score = TextBlob(str(text)).sentiment.polarity
            # Combinar ambos scores
            combined_score = (vader_score + textblob_score) / 2
            return combined_score, textblob_score, vader_score
        except:
            return 0.0, 0.0, 0.0
    
    sentiment_results = df['cleaned_text'].apply(get_sentiment)
    df['sentiment_score'] = sentiment_results.apply(lambda x: x[0])
    df['textblob_sentiment'] = sentiment_results.apply(lambda x: x[1])
    df['vader_compound'] = sentiment_results.apply(lambda x: x[2])
    
    # Clasificar sentimiento
    conditions = [
        df['sentiment_score'] > 0.1,
        df['sentiment_score'] < -0.1
    ]
    choices = ['positive', 'negative']
    df['sentiment_label'] = np.select(conditions, choices, default='neutral')
    
    # Intensidad del sentimiento
    df['sentiment_intensity'] = df['sentiment_score'].abs()
    
    logging.info(f"Distribución de sentimiento: {df['sentiment_label'].value_counts().to_dict()}")
    return df

def extract_keyword_insights(df):
    """Extrae insights usando keywords simples sin NER"""
    logging.info("Extrayendo insights por keywords...")
    
    # Términos relacionados con tipo de piel
    skin_keywords = {
        'dry': ['dry', 'flaky', 'dehydrated', 'tight'],
        'oily': ['oily', 'greasy', 'shiny', 'sebum'],
        'combination': ['combination', 't-zone', 'oily t-zone'],
        'sensitive': ['sensitive', 'irritated', 'redness', 'reactive'],
        'normal': ['normal', 'balanced'],
        'mature': ['mature', 'aging', 'wrinkles', 'fine lines', 'anti-aging']
    }
    
    # Términos de preocupaciones
    concern_keywords = {
        'acne': ['acne', 'pimples', 'breakout', 'blemish'],
        'hydration': ['hydration', 'moisture', 'dryness'],
        'anti-aging': ['wrinkle', 'aging', 'firmness'],
        'brightening': ['brighten', 'glow', 'dull', 'dark spot'],
        'pores': ['pore', 'large pores', 'blackhead']
    }
    
    def extract_skin_info(text):
        text_lower = str(text).lower()
        skin_type = 'unknown'
        concerns = []
        
        for skin_type_key, keywords in skin_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                skin_type = skin_type_key
                break
        
        for concern, keywords in concern_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                concerns.append(concern)
        
        return skin_type, concerns[:2]  # Máximo 2 preocupaciones
    
    results = df['cleaned_text'].apply(extract_skin_info)
    df['inferred_skin_type'] = results.apply(lambda x: x[0])
    df['main_concerns'] = results.apply(lambda x: x[1])
    
    logging.info("Insights por keywords extraídos exitosamente")
    return df

def advanced_customer_clustering(df):
    """Clustering avanzado con PCA"""
    logging.info("Realizando clustering avanzado...")
    
    try:
        # Crear características para clustering
        clustering_features = []
        
        # Características básicas
        basic_features = ['text_length', 'rating', 'sentiment_score', 'sentiment_intensity']
        for feature in basic_features:
            if feature in df.columns:
                clustering_features.append(feature)
        
        # Características de texto
        text_features = ['text_word_count', 'text_sentence_count', 'text_avg_word_length']
        for feature in text_features:
            if feature in df.columns:
                clustering_features.append(feature)
        
        if len(clustering_features) < 3:
            logging.warning("No hay suficientes características para clustering avanzado")
            df['user_cluster'] = 0
            return df
        
        # Crear dataset para clustering
        cluster_data = df[clustering_features].copy().fillna(0)
        
        # Normalizar características
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(cluster_data)
        
        # Aplicar PCA para reducción dimensional
        n_samples = len(features_scaled)
        n_components = min(3, len(clustering_features), n_samples)
        
        if n_components >= 2:
            pca = PCA(n_components=n_components, random_state=42)
            features_pca = pca.fit_transform(features_scaled)
            
            # Explicar varianza
            variance_explained = pca.explained_variance_ratio_.sum()
            logging.info(f"Varianza explicada por PCA: {variance_explained:.3f}")
        else:
            features_pca = features_scaled
        
        # Aplicar K-means en componentes PCA
        n_clusters = min(4, n_samples)
        
        if n_samples >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['user_cluster'] = kmeans.fit_predict(features_pca)
            
            # Calcular métricas de cluster
            cluster_centers = kmeans.cluster_centers_
            logging.info(f"Clustering avanzado completado: {n_clusters} clusters creados")
            
            # Guardar información del clustering
            cluster_info = {
                'n_clusters': n_clusters,
                'features_used': clustering_features,
                'variance_explained': variance_explained if 'variance_explained' in locals() else 0,
                'cluster_sizes': df['user_cluster'].value_counts().to_dict()
            }
            
            return df, cluster_info
        else:
            logging.warning("No hay suficientes muestras para clustering")
            df['user_cluster'] = 0
            return df, None
        
    except Exception as e:
        logging.warning(f"Error en clustering avanzado: {e}. Asignando cluster unico.")
        df['user_cluster'] = 0
        return df, None

def temporal_analysis(df):
    """Análisis de series temporales para tendencias"""
    logging.info("Realizando análisis temporal...")
    
    if 'review_date' not in df.columns:
        logging.warning("No hay columna 'review_date' para análisis temporal")
        return None
    
    try:
        # Métricas diarias
        daily_metrics = df.groupby('review_date').agg({
            'rating': ['mean', 'count'],
            'sentiment_score': 'mean'
        }).round(3)
        
        daily_metrics.columns = ['avg_rating', 'review_count', 'avg_sentiment']
        daily_metrics = daily_metrics.reset_index()
        
        # Métricas móviles (7 días)
        if len(daily_metrics) >= 7:
            daily_metrics['rating_7d_avg'] = daily_metrics['avg_rating'].rolling(7, min_periods=1).mean()
            daily_metrics['sentiment_7d_avg'] = daily_metrics['avg_sentiment'].rolling(7, min_periods=1).mean()
            daily_metrics['volume_7d_avg'] = daily_metrics['review_count'].rolling(7, min_periods=1).mean()
        
        # Calcular tendencias
        if len(daily_metrics) >= 14:
            # Tendencia de 14 días
            daily_metrics['rating_trend'] = daily_metrics['avg_rating'].rolling(14).mean()
            daily_metrics['volume_trend'] = daily_metrics['review_count'].rolling(14).mean()
        
        logging.info(f"Análisis temporal completado: {len(daily_metrics)} días analizados")
        return daily_metrics
        
    except Exception as e:
        logging.error(f"Error en análisis temporal: {e}")
        return None

def product_analysis(df):
    """Análisis detallado por producto"""
    logging.info("Realizando análisis de productos...")
    
    if 'product_id' not in df.columns:
        logging.warning("No hay columna 'product_id' para análisis de productos")
        return None
    
    try:
        product_metrics = df.groupby('product_id').agg({
            'rating': ['mean', 'count', 'std'],
            'sentiment_score': 'mean',
            'user_id': 'nunique'
        }).round(3)
        
        product_metrics.columns = ['avg_rating', 'review_count', 'rating_std', 'avg_sentiment', 'unique_users']
        product_metrics = product_metrics.reset_index()
        
        # Clasificar productos por performance
        conditions = [
            (product_metrics['avg_rating'] >= 4.0) & (product_metrics['review_count'] >= 5) & (product_metrics['rating_std'] <= 1.0),
            (product_metrics['avg_rating'] <= 2.0) & (product_metrics['review_count'] >= 3),
            (product_metrics['avg_rating'] >= 3.0) & (product_metrics['avg_rating'] < 4.0) & (product_metrics['review_count'] >= 2)
        ]
        choices = ['high_performer', 'low_performer', 'average_performer']
        product_metrics['performance_category'] = np.select(conditions, choices, default='new_or_low_volume')
        
        # Calcular score de producto (combinación de rating y volumen)
        product_metrics['product_score'] = (
            product_metrics['avg_rating'] * np.log1p(product_metrics['review_count'])
        )
        
        logging.info(f"Análisis de productos completado: {len(product_metrics)} productos analizados")
        return product_metrics
        
    except Exception as e:
        logging.error(f"Error en análisis de productos: {e}")
        return None

def create_advanced_aggregations(df, daily_metrics, product_metrics):
    """Crea agregaciones avanzadas para el dashboard"""
    logging.info("Creando agregaciones avanzadas para dashboard...")
    
    try:
        # Resumen por marca si existe
        if 'brand_name' in df.columns:
            brand_summary = df.groupby('brand_name').agg({
                'rating': ['mean', 'count', 'std'],
                'sentiment_score': 'mean',
                'product_id': 'nunique',
                'user_id': 'nunique'
            }).round(3).reset_index()
            
            brand_summary.columns = ['brand_name', 'avg_rating', 'review_count', 'rating_std', 
                                   'avg_sentiment', 'product_count', 'unique_users']
            brand_summary.to_parquet("data/processed/brand_summary.parquet", index=False)
            logging.info("Resumen por marca guardado")
        
        # Resumen por categoría primaria
        if 'primary_category' in df.columns:
            category_summary = df.groupby('primary_category').agg({
                'rating': 'mean',
                'sentiment_score': 'mean',
                'product_id': 'nunique',
                'review_text': 'count'
            }).round(3).reset_index()
            
            category_summary.columns = ['category', 'avg_rating', 'avg_sentiment', 'product_count', 'review_count']
            category_summary.to_parquet("data/processed/category_summary.parquet", index=False)
            logging.info("Resumen por categoría guardado")
        
        # Resumen por cluster
        if 'user_cluster' in df.columns:
            cluster_summary = df.groupby('user_cluster').agg({
                'rating': ['mean', 'count'],
                'sentiment_score': 'mean',
                'text_word_count': 'mean',
                'sentiment_intensity': 'mean'
            }).round(3).reset_index()
            
            cluster_summary.columns = ['cluster', 'avg_rating', 'review_count', 'avg_sentiment', 
                                     'avg_word_count', 'avg_sentiment_intensity']
            cluster_summary.to_parquet("data/processed/cluster_summary.parquet", index=False)
            logging.info("Resumen por cluster guardado")
        
        # Guardar métricas temporales
        if daily_metrics is not None:
            daily_metrics.to_parquet("data/processed/temporal_metrics.parquet", index=False)
            logging.info("Métricas temporales guardadas")
        
        # Guardar métricas de productos
        if product_metrics is not None:
            product_metrics.to_parquet("data/processed/product_metrics.parquet", index=False)
            logging.info("Métricas de productos guardadas")
            
    except Exception as e:
        logging.error(f"Error creando agregaciones: {e}")

def save_processed_data(df):
    """Guarda el dataset procesado principal"""
    output_path = config["dataset"]["processed_path"]
    
    try:
        # Seleccionar columnas finales
        final_columns = [
            'review_text', 'cleaned_text', 'rating', 'sentiment_score',
            'sentiment_label', 'sentiment_intensity', 'textblob_sentiment', 
            'vader_compound', 'inferred_skin_type', 'main_concerns', 'user_cluster'
        ]
        
        # Añadir características de texto
        text_feature_cols = [col for col in df.columns if col.startswith('text_')]
        final_columns.extend(text_feature_cols)
        
        # Añadir columnas opcionales
        optional_cols = ['review_date', 'product_id', 'product_name', 'brand_name', 
                        'primary_category', 'secondary_category', 'user_id', 'skin_tone', 'eye_color', 
                        'skin_type', 'hair_color']
        
        for col in optional_cols:
            if col in df.columns:
                final_columns.append(col)
        
        # Seleccionar solo columnas existentes
        existing_columns = [col for col in final_columns if col in df.columns]
        df_final = df[existing_columns]
        
        # Guardar
        df_final.to_parquet(output_path, index=False)
        logging.info(f"Dataset procesado guardado en {output_path}")
        
        return df_final
        
    except Exception as e:
        logging.error(f"Error guardando datos procesados: {e}")
        # Guardar versión mínima como respaldo
        df_minimal = df[['review_text', 'cleaned_text', 'rating', 'sentiment_score', 'sentiment_label']]
        df_minimal.to_parquet(output_path, index=False)
        logging.info("Dataset mínimo guardado como respaldo")
        return df_minimal

if __name__ == "__main__":
    try:
        logging.info("=== INICIANDO TRANSFORMACIÓN DE DATOS ===")
        
        # Cargar datos
        df = pd.read_csv(config["dataset"]["raw_path"], low_memory=False)
        logging.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Pipeline de transformación
        df = clean_data(df)
        df = create_diverse_categories(df)  # NUEVO: Crear categorías diversas
        df = advanced_sentiment_analysis(df)
        df = extract_keyword_insights(df)
        
        # Extraer características de texto
        transformer = DataTransformer()
        text_features = df['cleaned_text'].apply(transformer.extract_features)
        for feature in ['word_count', 'sentence_count', 'avg_word_length', 'exclamation_count', 'question_count']:
            df[f'text_{feature}'] = text_features.apply(lambda x: x[feature])
        
        # Característica adicional: longitud del texto
        df['text_length'] = df['cleaned_text'].str.len()
        
        # Clustering avanzado
        df, cluster_info = advanced_customer_clustering(df)
        
        # Análisis adicionales
        daily_metrics = temporal_analysis(df)
        product_metrics = product_analysis(df)
        
        # Crear agregaciones avanzadas
        create_advanced_aggregations(df, daily_metrics, product_metrics)
        
        # Guardar datos procesados
        final_df = save_processed_data(df)
        
        print("Transformación completada exitosamente!")
        print(f"Datos finales: {final_df.shape[0]} reseñas, {final_df.shape[1]} características")
        if 'primary_category' in final_df.columns:
            category_dist = final_df['primary_category'].value_counts()
            print(f"Distribución de categorías: {category_dist.to_dict()}")
        if 'sentiment_label' in final_df.columns:
            sentiment_dist = final_df['sentiment_label'].value_counts()
            print(f"Distribución de sentimiento: {sentiment_dist.to_dict()}")
        if cluster_info:
            print(f"Clusters creados: {cluster_info['n_clusters']}")
            print(f"Tamaños de clusters: {cluster_info['cluster_sizes']}")
        print("Métricas guardadas en data/processed/")
        
    except Exception as e:
        logging.error(f"Error en transformación: {str(e)}")
        print(f"Error en transformación: {str(e)}")
        raise