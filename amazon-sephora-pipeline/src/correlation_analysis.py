import pandas as pd
import numpy as np
import yaml
import logging
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

# Cargar configuracion
with open("config/pipeline_config.yaml") as f:
    config = yaml.safe_load(f)

# Configurar logging
logging.basicConfig(
    filename=config['logging']['log_file'],
    level=getattr(logging, config['logging']['level']),
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data():
    """Carga los datos de reviews y redes sociales"""
    reviews_path = config["dataset"]["processed_path"]
    social_path = config["dataset"]["social_data_path"]
    
    reviews_df = pd.read_parquet(reviews_path)
    social_df = pd.read_parquet(social_path)
    
    return reviews_df, social_df

def preprocess_for_correlation(reviews_df, social_df):
    """Preprocesa los datos para analisis de correlacion"""
    
    # Agregar reviews por dia
    reviews_df['review_date'] = pd.to_datetime(reviews_df['review_date'])
    daily_reviews = reviews_df.groupby('review_date').agg({
        'rating': 'mean',
        'sentiment_score': 'mean',
        'product_id': 'count'
    }).reset_index()
    daily_reviews.columns = ['date', 'avg_rating', 'avg_sentiment', 'review_count']
    
    # Agregar social data por dia y plataforma
    social_df['date'] = pd.to_datetime(social_df['date'])
    daily_social = social_df.groupby(['date', 'platform']).agg({
        'mention_volume': 'sum',
        'engagement': 'sum'
    }).reset_index()
    
    return daily_reviews, daily_social

def calculate_correlations(daily_reviews, daily_social):
    """Calcula correlaciones entre reviews y actividad en redes sociales"""
    
    results = {}
    
    # Combinar datos por fecha
    for platform in daily_social['platform'].unique():
        platform_data = daily_social[daily_social['platform'] == platform]
        merged_data = daily_reviews.merge(platform_data, on='date', how='inner')
        
        if len(merged_data) < 10:
            logging.warning(f"No hay suficientes datos para correlacion con {platform}: {len(merged_data)} puntos")
            continue
        
        # Correlacion: Volume de reviews vs Mention volume
        if 'review_count' in merged_data.columns and 'mention_volume' in merged_data.columns:
            corr_spearman, p_spearman = spearmanr(merged_data['review_count'], merged_data['mention_volume'])
            corr_pearson, p_pearson = pearsonr(merged_data['review_count'], merged_data['mention_volume'])
            
            results[f'review_count_vs_mention_volume_{platform}'] = {
                'spearman_correlation': corr_spearman,
                'spearman_p_value': p_spearman,
                'pearson_correlation': corr_pearson,
                'pearson_p_value': p_pearson,
                'sample_size': len(merged_data)
            }
        
        # Correlacion: Sentimiento vs Engagement
        if 'avg_sentiment' in merged_data.columns and 'engagement' in merged_data.columns:
            corr_spearman, p_spearman = spearmanr(merged_data['avg_sentiment'], merged_data['engagement'])
            corr_pearson, p_pearson = pearsonr(merged_data['avg_sentiment'], merged_data['engagement'])
            
            results[f'sentiment_vs_engagement_{platform}'] = {
                'spearman_correlation': corr_spearman,
                'spearman_p_value': p_spearman,
                'pearson_correlation': corr_pearson,
                'pearson_p_value': p_pearson,
                'sample_size': len(merged_data)
            }
        
        # Correlacion: Rating promedio vs Engagement
        if 'avg_rating' in merged_data.columns and 'engagement' in merged_data.columns:
            corr_spearman, p_spearman = spearmanr(merged_data['avg_rating'], merged_data['engagement'])
            corr_pearson, p_pearson = pearsonr(merged_data['avg_rating'], merged_data['engagement'])
            
            results[f'rating_vs_engagement_{platform}'] = {
                'spearman_correlation': corr_spearman,
                'spearman_p_value': p_spearman,
                'pearson_correlation': corr_pearson,
                'pearson_p_value': p_pearson,
                'sample_size': len(merged_data)
            }
    
    return results

def analyze_hashtag_impact(reviews_df, social_df):
    """Analiza el impacto de hashtags especificos en las reviews"""
    
    results = {}
    
    # Agrupar social data por hashtag
    hashtag_performance = social_df.groupby('hashtag').agg({
        'mention_volume': 'mean',
        'engagement': 'mean'
    }).reset_index()
    
    # Ordenar por engagement
    top_hashtags = hashtag_performance.nlargest(10, 'engagement')['hashtag'].tolist()
    
    results['top_hashtags_by_engagement'] = top_hashtags
    
    # Buscar coincidencias entre hashtags y terminos en reviews
    for hashtag in top_hashtags:
        # Buscar el hashtag (sin #) en las reviews
        keyword = hashtag.replace('#', '').lower()
        contains_keyword = reviews_df['cleaned_text'].str.contains(keyword, case=False, na=False)
        
        if contains_keyword.sum() > 0:
            # Comparar metricas de reviews que contienen vs no contienen el keyword
            with_keyword = reviews_df[contains_keyword]
            without_keyword = reviews_df[~contains_keyword]
            
            results[f'hashtag_impact_{hashtag}'] = {
                'reviews_with_keyword': len(with_keyword),
                'avg_rating_with_keyword': with_keyword['rating'].mean(),
                'avg_sentiment_with_keyword': with_keyword['sentiment_score'].mean(),
                'avg_rating_without_keyword': without_keyword['rating'].mean(),
                'avg_sentiment_without_keyword': without_keyword['sentiment_score'].mean(),
                'rating_difference': with_keyword['rating'].mean() - without_keyword['rating'].mean(),
                'sentiment_difference': with_keyword['sentiment_score'].mean() - without_keyword['sentiment_score'].mean()
            }
    
    return results

def advanced_time_series_analysis(daily_reviews, daily_social):
    """Analisis de series temporales avanzado"""
    results = {}
    
    try:
        # Cross-correlation con diferentes lags
        for platform in daily_social['platform'].unique():
            platform_data = daily_social[daily_social['platform'] == platform]
            merged_data = daily_reviews.merge(platform_data, on='date', how='inner')
            
            if len(merged_data) < 30:
                continue
            
            # Analizar diferentes lags (0-14 días)
            lag_correlations = {}
            for lag in range(0, 15):
                shifted_reviews = merged_data['review_count'].shift(lag)
                valid_data = pd.concat([shifted_reviews, merged_data['mention_volume']], axis=1).dropna()
                
                if len(valid_data) >= 10:
                    corr, p_value = spearmanr(valid_data['review_count'], valid_data['mention_volume'])
                    lag_correlations[lag] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'lag_days': lag
                    }
            
            # Encontrar el mejor lag
            if lag_correlations:
                best_lag = max(lag_correlations.items(), key=lambda x: abs(x[1]['correlation']))
                results[f'optimal_lag_{platform}'] = {
                    'best_lag': best_lag[0],
                    'best_correlation': best_lag[1]['correlation'],
                    'all_lags': lag_correlations
                }
        
    except Exception as e:
        logging.warning(f"Error en analisis de series temporales: {e}")
    
    return results

def generate_correlation_report(correlation_results, hashtag_results, time_series_results, output_path="data/processed/correlation_report.txt"):
    """Genera un reporte de correlacion completo"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Reporte de Correlacion: Reviews vs Redes Sociales\n")
        f.write("=================================================\n\n")
        
        f.write("RESUMEN EJECUTIVO:\n")
        f.write("------------------\n")
        
        # Resumen de correlaciones significativas
        strong_correlations = []
        for key, result in correlation_results.items():
            if abs(result['spearman_correlation']) > 0.5 and result['spearman_p_value'] < 0.05:
                strong_correlations.append((key, result['spearman_correlation']))
        
        if strong_correlations:
            f.write("Correlaciones significativas encontradas (|r| > 0.5, p < 0.05):\n")
            for key, corr in strong_correlations:
                f.write(f"- {key}: {corr:.3f}\n")
        else:
            f.write("No se encontraron correlaciones fuertes significativas.\n")
        
        f.write("\nCORRELACIONES POR PLATAFORMA:\n")
        f.write("-----------------------------\n")
        
        for key, result in correlation_results.items():
            f.write(f"\n{key}:\n")
            f.write(f"  Spearman: {result['spearman_correlation']:.3f} (p-value: {result['spearman_p_value']:.3f})\n")
            f.write(f"  Pearson:  {result['pearson_correlation']:.3f} (p-value: {result['pearson_p_value']:.3f})\n")
            f.write(f"  Muestra:  {result['sample_size']} puntos\n")
            
            # Interpretacion
            corr_value = result['spearman_correlation']
            p_value = result['spearman_p_value']
            
            if p_value < 0.05:
                if abs(corr_value) > 0.7:
                    strength = "FUERTE"
                elif abs(corr_value) > 0.5:
                    strength = "MODERADA"
                elif abs(corr_value) > 0.3:
                    strength = "DEBIL"
                else:
                    strength = "MUY DEBIL"
                
                direction = "positiva" if corr_value > 0 else "negativa"
                f.write(f"  Interpretacion: Correlacion {strength} {direction} (ESTADISTICAMENTE SIGNIFICATIVA)\n")
            else:
                f.write(f"  Interpretacion: Correlacion NO significativa (p > 0.05)\n")
        
        f.write("\nIMPACTO DE HASHTAGS:\n")
        f.write("--------------------\n")
        
        if 'top_hashtags_by_engagement' in hashtag_results:
            f.write(f"Top hashtags por engagement: {', '.join(hashtag_results['top_hashtags_by_engagement'])}\n\n")
            
            for key, result in hashtag_results.items():
                if key.startswith('hashtag_impact_'):
                    hashtag = key.replace('hashtag_impact_', '')
                    f.write(f"Hashtag: {hashtag}\n")
                    f.write(f"  Reviews que mencionan: {result['reviews_with_keyword']}\n")
                    f.write(f"  Rating promedio (con keyword): {result['avg_rating_with_keyword']:.2f}\n")
                    f.write(f"  Rating promedio (sin keyword): {result['avg_rating_without_keyword']:.2f}\n")
                    f.write(f"  Diferencia en rating: {result['rating_difference']:+.2f}\n")
                    f.write(f"  Sentimiento promedio (con keyword): {result['avg_sentiment_with_keyword']:.3f}\n")
                    f.write(f"  Sentimiento promedio (sin keyword): {result['avg_sentiment_without_keyword']:.3f}\n")
                    f.write(f"  Diferencia en sentimiento: {result['sentiment_difference']:+.3f}\n\n")
        
        f.write("\nANALISIS DE SERIES TEMPORALES:\n")
        f.write("------------------------------\n")
        
        for key, result in time_series_results.items():
            if key.startswith('optimal_lag_'):
                platform = key.replace('optimal_lag_', '')
                f.write(f"Plataforma: {platform}\n")
                f.write(f"  Mejor lag: {result['best_lag']} dias\n")
                f.write(f"  Correlacion en mejor lag: {result['best_correlation']:.3f}\n")
        
        f.write("\nRECOMENDACIONES:\n")
        f.write("----------------\n")
        
        # Recomendaciones basadas en los resultados
        strong_corr_count = len([r for r in correlation_results.values() 
                               if abs(r['spearman_correlation']) > 0.5 and r['spearman_p_value'] < 0.05])
        
        if strong_corr_count > 0:
            f.write(f"- Se encontraron {strong_corr_count} correlaciones fuertes significativas\n")
            f.write("- Considera monitorear estas metricas para prediccion de ventas\n")
        else:
            f.write("- No se encontraron correlaciones fuertes consistentes\n")
            f.write("- Considera revisar la estrategia de redes sociales\n")
        
        # Recomendaciones especificas por plataforma
        for platform in ['tiktok', 'instagram', 'youtube']:
            platform_corrs = [k for k in correlation_results.keys() if platform in k]
            if platform_corrs:
                avg_corr = np.mean([abs(correlation_results[k]['spearman_correlation']) for k in platform_corrs])
                if avg_corr > 0.4:
                    f.write(f"- {platform.capitalize()} muestra correlaciones moderadas, buena plataforma para campañas\n")
    
    logging.info(f"Reporte de correlacion guardado en {output_path}")

if __name__ == "__main__":
    try:
        logging.info("=== INICIANDO ANALISIS DE CORRELACION ===")
        
        # Cargar datos
        reviews_df, social_df = load_data()
        
        # Preprocesar
        daily_reviews, daily_social = preprocess_for_correlation(reviews_df, social_df)
        
        # Calcular correlaciones basicas
        correlation_results = calculate_correlations(daily_reviews, daily_social)
        
        # Analizar impacto de hashtags
        hashtag_results = analyze_hashtag_impact(reviews_df, social_df)
        
        # Analisis de series temporales avanzado
        time_series_results = advanced_time_series_analysis(daily_reviews, daily_social)
        
        # Generar reporte
        generate_correlation_report(correlation_results, hashtag_results, time_series_results)
        
        print("Analisis de correlacion completado exitosamente!")
        print("Ver reporte en data/processed/correlation_report.txt")
        
        # Mostrar resumen en consola
        print("\n=== RESUMEN DE CORRELACIONES ===")
        for key, result in correlation_results.items():
            corr = result['spearman_correlation']
            p_val = result['spearman_p_value']
            sig = "**" if p_val < 0.05 else ""
            print(f"{key}: {corr:.3f} {sig}(p={p_val:.3f}){sig}")
        
    except Exception as e:
        logging.error(f"Error en analisis de correlacion: {str(e)}")
        print(f"Error en analisis de correlacion: {str(e)}")