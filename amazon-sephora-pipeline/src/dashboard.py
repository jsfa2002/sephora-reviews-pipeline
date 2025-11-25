import streamlit as st
import pandas as pd
import numpy as np
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
import re
from scipy.stats import f_oneway, spearmanr
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Agregar el directorio raíz al path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar pagina
st.set_page_config(
    page_title="Sephora Reviews Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar configuracion
try:
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "pipeline_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
except FileNotFoundError as e:
    st.error(f"Archivo de configuracion no encontrado: {e}")
    config = {}
except Exception as e:
    st.error(f"Error cargando configuracion: {e}")
    config = {}

@st.cache_data(ttl=3600)
def load_data():
    """Carga los datos procesados de forma tolerante a errores"""
    try:
        processed_path = config.get('dataset', {}).get('processed_path', 'data/processed/reviews_processed.parquet')
        
        if not os.path.exists(processed_path):
            st.error(f"Archivo no encontrado: {processed_path}")
            processed_dir = "data/processed"
            if os.path.exists(processed_dir):
                files = os.listdir(processed_dir)
                st.write(f"Archivos disponibles: {files}")
            return None
        
        df = pd.read_parquet(processed_path)
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None

@st.cache_data(ttl=3600)
def load_social_data():
    """Carga los datos de redes sociales"""
    try:
        social_path = config.get('dataset', {}).get('social_data_path', 'data/processed/social_media_data.parquet')
        if not os.path.exists(social_path):
            return None
        
        social_df = pd.read_parquet(social_path)
        return social_df
    except Exception as e:
        return None

def detect_categories(df):
    """Detecta automaticamente las columnas de categoria disponibles"""
    category_columns = []
    
    possible_category_cols = [
        'primary_category', 'secondary_category', 'category', 
        'product_type', 'brand_name', 'product_name'
    ]
    
    for col in possible_category_cols:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            if 1 < len(unique_vals) <= 50:
                category_columns.append((col, len(unique_vals)))
    
    if not category_columns:
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_vals = df[col].dropna().unique()
                if 1 < len(unique_vals) <= 20:
                    category_columns.append((col, len(unique_vals)))
    
    return category_columns

def calculate_temporal_metrics(filtered_df):
    """Calcula metricas temporales basadas en los datos filtrados"""
    if 'review_date' not in filtered_df.columns or len(filtered_df) == 0:
        return None
    
    try:
        filtered_df = filtered_df.copy()
        filtered_df['review_date'] = pd.to_datetime(filtered_df['review_date'])
        
        daily_metrics = filtered_df.groupby('review_date').agg({
            'rating': ['mean', 'count'],
            'sentiment_score': 'mean'
        }).round(3)
        
        daily_metrics.columns = ['avg_rating', 'review_count', 'avg_sentiment']
        daily_metrics = daily_metrics.reset_index()
        
        daily_metrics = daily_metrics.sort_values('review_date')
        
        if len(daily_metrics) >= 7:
            daily_metrics['rating_7d_avg'] = daily_metrics['avg_rating'].rolling(7, min_periods=1).mean()
            daily_metrics['sentiment_7d_avg'] = daily_metrics['avg_sentiment'].rolling(7, min_periods=1).mean()
            daily_metrics['volume_7d_avg'] = daily_metrics['review_count'].rolling(7, min_periods=1).mean()
        
        return daily_metrics
        
    except Exception as e:
        st.error(f"Error calculando metricas temporales: {e}")
        return None

def calculate_brand_summary(filtered_df):
    """Calcula resumen por marca basado en datos filtrados"""
    if 'brand_name' not in filtered_df.columns or len(filtered_df) == 0:
        return None
    
    try:
        brand_summary = filtered_df.groupby('brand_name').agg({
            'rating': ['mean', 'count'],
            'sentiment_score': 'mean',
            'product_id': 'nunique'
        }).round(3).reset_index()
        
        brand_summary.columns = ['brand_name', 'avg_rating', 'review_count', 'avg_sentiment', 'product_count']
        return brand_summary
        
    except Exception as e:
        st.error(f"Error calculando resumen de marcas: {e}")
        return None

def calculate_product_metrics(filtered_df):
    """Calcula metricas de productos basadas en datos filtrados"""
    if 'product_id' not in filtered_df.columns or len(filtered_df) == 0:
        return None
    
    try:
        product_metrics = filtered_df.groupby('product_id').agg({
            'rating': ['mean', 'count'],
            'sentiment_score': 'mean',
        }).round(3)
        
        product_metrics.columns = ['avg_rating', 'review_count', 'avg_sentiment']
        product_metrics = product_metrics.reset_index()
        
        if 'product_name' in filtered_df.columns:
            product_names = filtered_df.groupby('product_id')['product_name'].first()
            product_metrics = product_metrics.merge(product_names, on='product_id', how='left')
        else:
            product_metrics['product_name'] = product_metrics['product_id']
        
        conditions = [
            (product_metrics['avg_rating'] >= 4.0) & (product_metrics['review_count'] >= 2),
            (product_metrics['avg_rating'] <= 2.0) & (product_metrics['review_count'] >= 1),
            (product_metrics['avg_rating'] >= 3.0) & (product_metrics['avg_rating'] < 4.0)
        ]
        choices = ['alto_desempeno', 'bajo_desempeno', 'desempeno_promedio']
        product_metrics['performance_category'] = np.select(conditions, choices, default='nuevo_o_bajo_volumen')
        
        return product_metrics
        
    except Exception as e:
        st.error(f"Error calculando metricas de productos: {e}")
        return None

def filter_social_data(social_df, date_range=None, hashtags=None, platforms=None):
    """Filtra los datos de redes sociales según los criterios seleccionados"""
    if social_df is None:
        return None
    
    filtered_social = social_df.copy()
    
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        filtered_social = filtered_social[
            (filtered_social['date'] >= start_date) & 
            (filtered_social['date'] <= end_date)
        ]
    
    if hashtags and 'Todas' not in hashtags:
        filtered_social = filtered_social[filtered_social['hashtag'].isin(hashtags)]
    
    if platforms and 'Todas' not in platforms:
        filtered_social = filtered_social[filtered_social['platform'].isin(platforms)]
    
    return filtered_social

def calculate_social_metrics(filtered_social):
    """Calcula métricas de redes sociales basadas en datos filtrados"""
    if filtered_social is None or len(filtered_social) == 0:
        return None
    
    try:
        platform_metrics = filtered_social.groupby('platform').agg({
            'mention_volume': ['sum', 'mean'],
            'engagement': ['sum', 'mean']
        }).round(0)
        
        platform_metrics.columns = ['total_mentions', 'avg_mentions', 'total_engagement', 'avg_engagement']
        platform_metrics = platform_metrics.reset_index()
        
        hashtag_metrics = filtered_social.groupby('hashtag').agg({
            'mention_volume': 'sum',
            'engagement': 'sum'
        }).round(0).reset_index()
        
        temporal_social = filtered_social.groupby('date').agg({
            'mention_volume': 'sum',
            'engagement': 'sum'
        }).reset_index()
        
        return {
            'platform_metrics': platform_metrics,
            'hashtag_metrics': hashtag_metrics,
            'temporal_metrics': temporal_social
        }
        
    except Exception as e:
        st.error(f"Error calculando métricas sociales: {e}")
        return None

def enhanced_temporal_analysis(temporal_metrics):
    """Analisis de series temporales avanzado basado en datos filtrados"""
    st.header("Analisis de Series Temporales Avanzado")
    
    if temporal_metrics is not None and len(temporal_metrics) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patrones de Estacionalidad")
            fig_seasonal = go.Figure()
            
            if 'rating_7d_avg' in temporal_metrics.columns:
                fig_seasonal.add_trace(go.Scatter(
                    x=temporal_metrics['review_date'],
                    y=temporal_metrics['rating_7d_avg'],
                    name="Rating (Media 7 dias)",
                    line=dict(color='#FF6B9D')
                ))
            
            if 'sentiment_7d_avg' in temporal_metrics.columns:
                fig_seasonal.add_trace(go.Scatter(
                    x=temporal_metrics['review_date'],
                    y=temporal_metrics['sentiment_7d_avg'],
                    name="Sentimiento (Media 7 dias)",
                    line=dict(color='#36A2EB')
                ))
            
            fig_seasonal.update_layout(
                title="Tendencias y Estacionalidad",
                xaxis_title="Fecha",
                yaxis_title="Valor"
            )
            st.plotly_chart(fig_seasonal, use_container_width=True)
        
        with col2:
            st.subheader("Deteccion de Anomalias")
            
            if 'review_count' in temporal_metrics.columns:
                temporal_metrics['volume_zscore'] = (
                    (temporal_metrics['review_count'] - temporal_metrics['review_count'].mean()) / 
                    temporal_metrics['review_count'].std()
                )
                
                anomalies = temporal_metrics[temporal_metrics['volume_zscore'].abs() > 2]
                
                fig_anomalies = go.Figure()
                fig_anomalies.add_trace(go.Scatter(
                    x=temporal_metrics['review_date'],
                    y=temporal_metrics['review_count'],
                    name="Volumen Reviews",
                    line=dict(color='#2E86AB')
                ))
                
                if len(anomalies) > 0:
                    fig_anomalies.add_trace(go.Scatter(
                        x=anomalies['review_date'],
                        y=anomalies['review_count'],
                        mode='markers',
                        name="Anomalias (Z-score > 2)",
                        marker=dict(color='red', size=10, symbol='x')
                    ))
                
                fig_anomalies.update_layout(
                    title="Deteccion de Anomalias en Volumen",
                    xaxis_title="Fecha",
                    yaxis_title="Volumen de Reviews"
                )
                st.plotly_chart(fig_anomalies, use_container_width=True)
                
                if len(anomalies) > 0:
                    st.warning(f"Se detectaron {len(anomalies)} anomalias en el volumen de reviews")
                    
                    with st.expander("Ver detalles de anomalias"):
                        for _, anomaly in anomalies.iterrows():
                            st.write(f"- {anomaly['review_date'].date()}: {anomaly['review_count']} reviews (Z-score: {anomaly['volume_zscore']:.2f})")
                else:
                    st.success("No se detectaron anomalias significativas en el volumen")
            
            st.subheader("Metricas de Variacion")
            if 'review_count' in temporal_metrics.columns:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    vol_variance = temporal_metrics['review_count'].var()
                    st.metric("Varianza Volumen", f"{vol_variance:.1f}")
                
                with col2:
                    vol_cv = (temporal_metrics['review_count'].std() / temporal_metrics['review_count'].mean()) * 100
                    st.metric("Coef. Variacion", f"{vol_cv:.1f}%")
                
                with col3:
                    max_vol = temporal_metrics['review_count'].max()
                    min_vol = temporal_metrics['review_count'].min()
                    st.metric("Rango Volumen", f"{min_vol}-{max_vol}")
    else:
        st.info("No hay suficientes datos temporales para el analisis")

def statistical_analysis_section(filtered_df):
    """Analisis estadistico avanzado basado en datos filtrados"""
    st.header("Analisis Estadistico Avanzado")
    
    if len(filtered_df) == 0:
        st.warning("No hay datos para analisis estadistico")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ANOVA entre Categorias")
        
        if 'brand_name' in filtered_df.columns:
            top_brands = filtered_df['brand_name'].value_counts().head(5).index
            anova_data = []
            
            for brand in top_brands:
                brand_ratings = filtered_df[filtered_df['brand_name'] == brand]['rating']
                anova_data.append(brand_ratings)
            
            if len(anova_data) >= 2:
                f_stat, p_value = f_oneway(*anova_data)
                
                st.metric("F-statistic", f"{f_stat:.3f}")
                st.metric("P-value", f"{p_value:.3f}")
                
                if p_value < 0.05:
                    st.success("Diferencias significativas entre marcas (p < 0.05)")
                else:
                    st.info("No hay diferencias significativas entre marcas")
            else:
                st.info("No hay suficientes marcas para ANOVA")
        else:
            st.info("Se necesita columna 'brand_name' para ANOVA")
    
    with col2:
        st.subheader("Correlacion Spearman")
        
        numeric_cols = ['rating', 'sentiment_score', 'sentiment_intensity']
        available_numeric = [col for col in numeric_cols if col in filtered_df.columns]
        
        if len(available_numeric) >= 2:
            corr_matrix = filtered_df[available_numeric].corr(method='spearman')
            
            fig_corr = px.imshow(
                corr_matrix,
                title="Matriz de Correlacion Spearman",
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No hay suficientes variables numericas para correlacion")

def enhanced_customer_segmentation(filtered_df):
    """Segmentacion avanzada de consumidores basada en datos filtrados"""
    st.header("Segmentacion Avanzada de Consumidores")
    
    if len(filtered_df) == 0:
        st.warning("No hay datos para segmentacion")
        return
    
    try:
        if 'user_id' in filtered_df.columns and 'review_date' in filtered_df.columns:
            max_date = filtered_df['review_date'].max()
            
            rfm = filtered_df.groupby('user_id').agg({
                'review_date': lambda x: (max_date - x.max()).days,
                'user_id': 'count',
                'rating': 'mean',
                'sentiment_score': 'mean'
            }).rename(columns={
                'review_date': 'recency',
                'user_id': 'frequency',
                'rating': 'monetary',
                'sentiment_score': 'avg_sentiment'
            })
            
            rfm['r_quartile'] = pd.qcut(rfm['recency'], 4, labels=[4, 3, 2, 1])
            rfm['f_quartile'] = pd.qcut(rfm['frequency'], 4, labels=[1, 2, 3, 4])
            rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, labels=[1, 2, 3, 4])
            
            rfm['rfm_score'] = rfm['r_quartile'].astype(str) + rfm['f_quartile'].astype(str) + rfm['m_quartile'].astype(str)
            
            segment_map = {
                r'4[4-9][4-9]': 'Champions',
                r'[3-4][3-4][3-4]': 'Loyal Customers',
                r'[2-3][2-3][2-3]': 'Potential Loyalists',
                r'[3-4][1-2][1-2]': 'New Customers',
                r'[1-2][3-4][3-4]': 'At Risk',
                r'[1-2][1-2][1-2]': 'Lost Customers'
            }
            
            rfm['segment'] = rfm['rfm_score']
            for pattern, segment_name in segment_map.items():
                rfm.loc[rfm['rfm_score'].str.match(pattern), 'segment'] = segment_name
            
            rfm['segment'] = rfm['segment'].fillna('Others')
            
            col1, col2 = st.columns(2)
            
            with col1:
                segment_counts = rfm['segment'].value_counts()
                fig_segments = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title="Distribucion de Segmentos RFM"
                )
                st.plotly_chart(fig_segments, use_container_width=True)
            
            with col2:
                segment_metrics = rfm.groupby('segment').agg({
                    'recency': 'mean',
                    'frequency': 'mean',
                    'monetary': 'mean',
                    'avg_sentiment': 'mean'
                }).round(3)
                
                st.dataframe(segment_metrics, use_container_width=True)
        
        else:
            if 'user_cluster' in filtered_df.columns:
                st.subheader("Analisis de Clusters")
                
                cluster_analysis = filtered_df.groupby('user_cluster').agg({
                    'rating': ['mean', 'count'],
                    'sentiment_score': 'mean',
                }).round(3)
                
                cluster_analysis.columns = ['_'.join(col).strip() for col in cluster_analysis.columns.values]
                st.dataframe(cluster_analysis, use_container_width=True)
                
                st.subheader("Interpretacion de Clusters")
                st.info("""
                - Cluster 0: Usuarios promedio
                - Cluster 1: Usuarios muy positivos 
                - Cluster 2: Usuarios criticos/negativos
                - Cluster 3: Usuarios detallistas (reviews largas)
                """)
            else:
                st.info("No hay datos suficientes para segmentacion avanzada")
            
    except Exception as e:
        st.warning(f"Segmentacion avanzada no disponible: {str(e)}")

def enhanced_social_correlation(filtered_df, filtered_social):
    """Correlacion avanzada con redes sociales"""
    st.header("Correlacion Avanzada con Redes Sociales")
    
    try:
        if filtered_social is not None and len(filtered_social) > 0 and len(filtered_df) > 0:
            # Preparar datos para correlacion
            reviews_daily = filtered_df.groupby('review_date').agg({
                'rating': 'mean',
                'sentiment_score': 'mean',
                'product_id': 'count'
            }).reset_index()
            reviews_daily.columns = ['date', 'avg_rating', 'avg_sentiment', 'review_count']
            
            social_daily = filtered_social.groupby(['date', 'platform']).agg({
                'mention_volume': 'sum',
                'engagement': 'sum'
            }).reset_index()
            
            # Calcular correlaciones
            correlation_results = {}
            
            for platform in social_daily['platform'].unique():
                platform_social = social_daily[social_daily['platform'] == platform]
                merged_data = reviews_daily.merge(platform_social, on='date', how='inner')
                
                if len(merged_data) >= 5:
                    if 'review_count' in merged_data.columns and 'mention_volume' in merged_data.columns:
                        corr, p_value = spearmanr(merged_data['review_count'], merged_data['mention_volume'])
                        correlation_results[f'{platform}_reviews_vs_mentions'] = {
                            'correlation': corr,
                            'p_value': p_value,
                            'sample_size': len(merged_data)
                        }
                    
                    if 'avg_sentiment' in merged_data.columns and 'engagement' in merged_data.columns:
                        corr, p_value = spearmanr(merged_data['avg_sentiment'], merged_data['engagement'])
                        correlation_results[f'{platform}_sentiment_vs_engagement'] = {
                            'correlation': corr,
                            'p_value': p_value,
                            'sample_size': len(merged_data)
                        }
            
            # Mostrar resultados
            if correlation_results:
                corr_data = []
                for key, result in correlation_results.items():
                    corr_data.append({
                        'platform': key,
                        'correlation': result['correlation'],
                        'p_value': result['p_value'],
                        'sample_size': result['sample_size']
                    })
                
                corr_df = pd.DataFrame(corr_data)
                
                fig_corr_bars = px.bar(
                    corr_df,
                    x='platform',
                    y='correlation',
                    color='correlation',
                    title="Correlacion por Plataforma",
                    color_continuous_scale='RdYlGn',
                    range_color=[-1, 1]
                )
                st.plotly_chart(fig_corr_bars, use_container_width=True)
                
                st.subheader("Interpretacion de Correlaciones")
                
                # Crear una tabla bonita para las correlaciones
                for _, row in corr_df.iterrows():
                    corr_val = row['correlation']
                    platform = row['platform']
                    p_val = row['p_value']
                    
                    # Determinar color y simbolo
                    if p_val < 0.05:
                        if abs(corr_val) > 0.7:
                            color = "🟢"  # Verde para fuerte
                            strength = "FUERTE"
                        elif abs(corr_val) > 0.5:
                            color = "🟡"  # Amarillo para moderada
                            strength = "MODERADA"
                        elif abs(corr_val) > 0.3:
                            color = "🟠"  # Naranja para debil
                            strength = "DEBIL"
                        else:
                            color = "🔴"  # Rojo para muy debil
                            strength = "MUY DEBIL"
                        significance = "✓"  # Check para significativo
                    else:
                        color = "⚫"  # Negro para no significativo
                        strength = "NO SIGNIFICATIVA"
                        significance = "✗"  # X para no significativo
                    
                    # Mostrar en columnas para mejor formato
                    col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
                    
                    with col1:
                        st.write(f"{color} {significance}")
                    
                    with col2:
                        st.write(f"**{platform}**")
                    
                    with col3:
                        st.write(f"{strength} ({corr_val:.3f})")
                    
                    with col4:
                        if p_val < 0.05:
                            st.write(f"p = {p_val:.3f}")
                        else:
                            st.write(f"p = {p_val:.3f}")
                
                # Agregar leyenda
                with st.expander("Leyenda de interpretación"):
                    st.write("""
                    **Símbolos y colores:**
                    - 🟢 ✓ : Correlación FUERTE y significativa
                    - 🟡 ✓ : Correlación MODERADA y significativa  
                    - 🟠 ✓ : Correlación DÉBIL y significativa
                    - 🔴 ✓ : Correlación MUY DÉBIL pero significativa
                    - ⚫ ✗ : Correlación NO SIGNIFICATIVA
                    
                    **Criterios:**
                    - Fuerte: |r| > 0.7
                    - Moderada: 0.5 < |r| ≤ 0.7
                    - Débil: 0.3 < |r| ≤ 0.5
                    - Muy débil: |r| ≤ 0.3
                    - Significativa: p < 0.05
                    """)
                    
            else:
                st.info("No hay suficientes datos superpuestos para calcular correlaciones")
        
        # Mostrar reporte de correlacion si existe
        correlation_path = "data/processed/correlation_report.txt"
        if os.path.exists(correlation_path):
            with st.expander("Ver Reporte Completo de Correlacion"):
                with open(correlation_path, "r", encoding='utf-8') as f:
                    correlation_report = f.read()
                st.text(correlation_report)
        else:
            st.info("Ejecuta el analisis de correlacion para generar el reporte completo")
            
    except Exception as e:
        st.error(f"Error en analisis de correlacion: {e}")

def data_quality_metrics(filtered_df):
    """Metricas de calidad de datos basadas en datos filtrados"""
    st.header("Metricas de Calidad de Datos")
    
    if len(filtered_df) == 0:
        st.warning("No hay datos para calcular metricas de calidad")
        return
    
    try:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            try:
                null_count = 0
                total_cells = 0
                for col in filtered_df.columns:
                    null_count += filtered_df[col].isnull().sum()
                    total_cells += len(filtered_df)
                completeness = (1 - null_count / total_cells) * 100 if total_cells > 0 else 0
                st.metric("Completitud", f"{completeness:.1f}%")
            except:
                st.metric("Completitud", "N/A")
        
        with col2:
            try:
                basic_cols = ['rating', 'sentiment_score', 'sentiment_label']
                available_basic = [col for col in basic_cols if col in filtered_df.columns]
                
                if available_basic:
                    basic_df = filtered_df[available_basic].copy()
                    duplicates = basic_df.duplicated().sum()
                    st.metric("Duplicados", duplicates)
                else:
                    st.metric("Duplicados", "N/A")
            except:
                st.metric("Duplicados", "N/A")
        
        with col3:
            try:
                if 'sentiment_label' in filtered_df.columns and 'rating' in filtered_df.columns:
                    positive_consistent = ((filtered_df['sentiment_label'] == 'positive') & (filtered_df['rating'] >= 4)).sum()
                    total = len(filtered_df)
                    consistency = (positive_consistent / total) * 100 if total > 0 else 0
                    st.metric("Consistencia Basica", f"{consistency:.1f}%")
                else:
                    st.metric("Consistencia Basica", "N/A")
            except:
                st.metric("Consistencia Basica", "Error")
        
        with col4:
            try:
                if 'rating' in filtered_df.columns:
                    avg_rating = filtered_df['rating'].mean()
                    st.metric("Rating Promedio", f"{avg_rating:.2f}")
                else:
                    st.metric("Rating Promedio", "N/A")
            except:
                st.metric("Rating Promedio", "Error")
        
        if 'cleaned_text' in filtered_df.columns:
            try:
                st.subheader("Calidad de Textos")
                text_stats = filtered_df['cleaned_text'].str.len().describe()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Longitud promedio", f"{text_stats['mean']:.1f}")
                    st.metric("Textos cortos (<10)", (filtered_df['cleaned_text'].str.len() < 10).sum())
                with col2:
                    st.metric("Longitud maxima", f"{text_stats['max']:.0f}")
                    st.metric("Textos largos (>1000)", (filtered_df['cleaned_text'].str.len() > 1000).sum())
            except:
                st.info("Analisis de textos no disponible")
    
    except Exception as e:
        st.error(f"Error en metricas de calidad: {str(e)}")

def simple_forecasting(temporal_metrics):
    """Forecasting basico basado en datos filtrados"""
    st.header("Forecasting Basico")
    
    if temporal_metrics is not None and len(temporal_metrics) > 10:
        try:
            if 'review_count' in temporal_metrics.columns:
                temporal_metrics = temporal_metrics.sort_values('review_date')
                temporal_metrics['forecast_7d'] = temporal_metrics['review_count'].rolling(7, min_periods=1).mean()
                
                if len(temporal_metrics) >= 14:
                    x = np.arange(len(temporal_metrics))
                    y = temporal_metrics['review_count'].values
                    z = np.polyfit(x, y, 1)
                    trend_slope = z[0]
                    
                    last_date = temporal_metrics['review_date'].max()
                    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
                    future_reviews = [temporal_metrics['review_count'].iloc[-1] + trend_slope * i for i in range(1, 8)]
                
                last_7d_avg = temporal_metrics['review_count'].tail(7).mean()
                last_30d_avg = temporal_metrics['review_count'].tail(30).mean() if len(temporal_metrics) >= 30 else last_7d_avg
                
                fig_forecast = go.Figure()
                
                fig_forecast.add_trace(go.Scatter(
                    x=temporal_metrics['review_date'],
                    y=temporal_metrics['review_count'],
                    name="Volumen Real",
                    line=dict(color='#2E86AB'),
                    mode='lines+markers'
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=temporal_metrics['review_date'],
                    y=temporal_metrics['forecast_7d'],
                    name="Media Movil 7d",
                    line=dict(color='#A23B72', dash='dash')
                ))
                
                if len(temporal_metrics) >= 14 and 'future_dates' in locals():
                    fig_forecast.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_reviews,
                        name="Proyeccion 7 dias",
                        line=dict(color='#F18F01', dash='dot'),
                        mode='lines+markers'
                    ))
                
                fig_forecast.update_layout(
                    title="Forecasting de Volumen de Reviews",
                    xaxis_title="Fecha",
                    yaxis_title="Numero de Reviews"
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Promedio ultimos 7 dias", f"{last_7d_avg:.0f} reviews/dia")
                
                with col2:
                    st.metric("Promedio ultimos 30 dias", f"{last_30d_avg:.0f} reviews/dia")
                
                with col3:
                    if len(temporal_metrics) >= 14:
                        trend_direction = "Aumentando" if trend_slope > 0 else "Disminuyendo" if trend_slope < 0 else "Estable"
                        st.metric("Tendencia", trend_direction)
                    else:
                        st.metric("Tendencia", "Necesita mas datos")
                
                st.subheader("Insights del Forecasting")
                
                if last_7d_avg > last_30d_avg * 1.2:
                    st.success("Tendencia positiva: El volumen reciente es mayor que el promedio historico. Considera aumentar la capacidad de atencion al cliente.")
                elif last_7d_avg < last_30d_avg * 0.8:
                    st.warning("Tendencia negativa: El volumen reciente es menor que el promedio historico. Revisa posibles causas.")
                else:
                    st.info("Tendencia estable: El volumen se mantiene dentro del rango esperado.")
                    
        except Exception as e:
            st.warning(f"Forecasting no disponible: {str(e)}")
    else:
        st.info("Se necesitan al menos 10 puntos de datos para forecasting")

def create_alerts_section(filtered_df):
    """Sistema de alertas basado en thresholds"""
    st.header("Sistema de Alertas")
    
    if len(filtered_df) == 0:
        st.warning("No hay datos para mostrar alertas")
        return
    
    negative_review_pct = (filtered_df['sentiment_label'] == 'negative').mean()
    low_rating_pct = (filtered_df['rating'] <= 2).mean()
    
    alerts = []
    
    if negative_review_pct > 0.3:
        alerts.append(f"CRITICO: Sentimiento negativo > 30% ({negative_review_pct:.1%})")
    
    if low_rating_pct > 0.1:
        alerts.append(f"ALTA: Reviews con rating <= 2 > 10% ({low_rating_pct:.1%})")
    
    try:
        with open("data/processed/correlation_report.txt", "r") as f:
            report = f.read()
            if "Correlacion muy debil o nula" in report:
                alerts.append("MEDIA: Correlacion con redes sociales muy baja")
    except:
        alerts.append("INFO: Reporte de correlacion no disponible")
    
    if alerts:
        for alert in alerts:
            if alert.startswith("CRITICO"):
                st.error(alert)
            elif alert.startswith("ALTA"):
                st.warning(alert)
            else:
                st.info(alert)
    else:
        st.success("Todas las metricas dentro de rangos normales")

def display_social_media_analysis(filtered_social, social_metrics):
    """Muestra el análisis de redes sociales integrado con los filtros"""
    st.header("Analisis de Redes Sociales")
    
    if filtered_social is None or len(filtered_social) == 0:
        st.warning("No hay datos de redes sociales para mostrar con los filtros seleccionados")
        return
    
    if social_metrics is None:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_mentions = filtered_social['mention_volume'].sum()
    total_engagement = filtered_social['engagement'].sum()
    avg_mentions_per_day = filtered_social.groupby('date')['mention_volume'].sum().mean()
    top_platform = filtered_social.groupby('platform')['mention_volume'].sum().idxmax()
    
    with col1:
        st.metric("Total de Menciones", f"{total_mentions:,}")
    with col2:
        st.metric("Total de Engagement", f"{total_engagement:,}")
    with col3:
        st.metric("Menciones Promedio/Dia", f"{avg_mentions_per_day:.0f}")
    with col4:
        st.metric("Plataforma Mas Activa", top_platform)
    
    col1, col2 = st.columns(2)
    
    with col1:
        platform_data = social_metrics['platform_metrics']
        fig_platform = px.bar(
            platform_data,
            x='platform',
            y='total_mentions',
            title='Menciones por Plataforma',
            color='platform'
        )
        st.plotly_chart(fig_platform, use_container_width=True)
    
    with col2:
        hashtag_data = social_metrics['hashtag_metrics'].nlargest(10, 'mention_volume')
        fig_hashtags = px.bar(
            hashtag_data,
            x='mention_volume',
            y='hashtag',
            orientation='h',
            title='Top 10 Hashtags por Menciones',
            color='mention_volume'
        )
        st.plotly_chart(fig_hashtags, use_container_width=True)
    
    st.subheader("Tendencias Temporales en Redes Sociales")
    
    temporal_data = social_metrics['temporal_metrics']
    if len(temporal_data) > 1:
        fig_temporal = go.Figure()
        fig_temporal.add_trace(go.Scatter(
            x=temporal_data['date'],
            y=temporal_data['mention_volume'],
            name='Menciones',
            line=dict(color='#FF6B9D')
        ))
        fig_temporal.add_trace(go.Scatter(
            x=temporal_data['date'],
            y=temporal_data['engagement'],
            name='Engagement',
            line=dict(color='#36A2EB')
        ))
        fig_temporal.update_layout(
            title="Evolucion de Menciones y Engagement",
            xaxis_title="Fecha",
            yaxis_title="Volumen"
        )
        st.plotly_chart(fig_temporal, use_container_width=True)

def main():
    st.title("Sephora Reviews Analytics Dashboard")
    st.markdown("Analisis de reseñas de productos y correlacion con redes sociales")
    
    with st.spinner("Cargando datos..."):
        df = load_data()
    
    if df is None:
        st.error("No se pudieron cargar los datos. Verifica que el pipeline se haya ejecutado correctamente.")
        st.info("Ejecuta primero: python src/orchestrator.py")
        return
    
    social_df = load_social_data()
    
    st.sidebar.title("Filtros")
    
    with st.sidebar.expander("Info del Dataset"):
        st.write(f"Total reviews: {len(df):,}")
        st.write(f"Columnas: {list(df.columns)}")
        if 'brand_name' in df.columns:
            st.write(f"Marcas: {df['brand_name'].nunique()}")
        if 'primary_category' in df.columns:
            st.write(f"Categorias unicas: {df['primary_category'].unique().tolist()}")
    
    brand_list = ['Todas']
    if 'brand_name' in df.columns:
        unique_brands = df['brand_name'].dropna().unique()
        brand_list.extend(sorted(unique_brands.tolist()))
        selected_brand = st.sidebar.selectbox("Marca", brand_list)
        st.sidebar.info(f"Marcas disponibles: {len(unique_brands)}")
    else:
        selected_brand = 'Todas'
        st.sidebar.warning("No hay datos de marcas disponibles")
    
    category_list = ['Todas']
    category_columns = detect_categories(df)
    
    if category_columns:
        best_category_col = max(category_columns, key=lambda x: x[1])[0]
        st.sidebar.success(f"Categorias de: {best_category_col}")
        
        all_categories = sorted(df[best_category_col].dropna().unique().tolist())
        category_list.extend(all_categories)
        selected_category = st.sidebar.selectbox("Categoria", category_list)
        
        st.sidebar.info(f"{len(all_categories)} categorias encontradas")
        if len(all_categories) <= 8:
            for cat in all_categories:
                count = len(df[df[best_category_col] == cat])
                st.sidebar.write(f"- {cat}: {count} reviews")
    else:
        selected_category = 'Todas'
        st.sidebar.error("No se encontraron categorias para filtrar")
        st.sidebar.write("Columnas disponibles:", [col for col in df.columns if df[col].dtype == 'object'][:5])
    
    date_range = None
    if 'review_date' in df.columns:
        try:
            df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
            df = df.dropna(subset=['review_date'])
            
            min_date = df['review_date'].min()
            max_date = df['review_date'].max()
            
            st.sidebar.write("Rango de fechas disponible:")
            st.sidebar.write(f"- Inicio: {min_date.date()}")
            st.sidebar.write(f"- Fin: {max_date.date()}")
            st.sidebar.write(f"- Dias totales: {(max_date - min_date).days}")
            
            date_range = st.sidebar.date_input(
                "Seleccionar rango de fechas",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
        except Exception as e:
            st.sidebar.error(f"Error con fechas: {e}")
            date_range = None
    else:
        st.sidebar.warning("No hay datos de fecha disponibles")
    
    st.sidebar.subheader("Filtros de Redes Sociales")
    
    platform_list = ['Todas']
    if social_df is not None:
        unique_platforms = social_df['platform'].unique()
        platform_list.extend(sorted(unique_platforms.tolist()))
        selected_platforms = st.sidebar.multiselect(
            "Plataformas",
            platform_list,
            default=['Todas']
        )
    else:
        selected_platforms = ['Todas']
    
    hashtag_list = ['Todas']
    if social_df is not None:
        unique_hashtags = social_df['hashtag'].unique()
        hashtag_list.extend(sorted(unique_hashtags.tolist()))
        selected_hashtags = st.sidebar.multiselect(
            "Hashtags",
            hashtag_list,
            default=['Todas']
        )
    else:
        selected_hashtags = ['Todas']
    
    filtered_df = df.copy()
    initial_count = len(filtered_df)
    filter_messages = []
    
    if selected_brand != 'Todas':
        before = len(filtered_df)
        filtered_df = filtered_df[filtered_df['brand_name'] == selected_brand]
        after = len(filtered_df)
        if before != after:
            filter_messages.append(f"Marca: {selected_brand} ({after} reviews)")
    
    if selected_category != 'Todas' and category_columns:
        before = len(filtered_df)
        best_category_col = max(category_columns, key=lambda x: x[1])[0]
        filtered_df = filtered_df[filtered_df[best_category_col] == selected_category]
        after = len(filtered_df)
        if before != after:
            filter_messages.append(f"Categoria: {selected_category} ({after} reviews)")
    
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        before = len(filtered_df)
        filtered_df = filtered_df[
            (filtered_df['review_date'] >= start_date) & 
            (filtered_df['review_date'] <= end_date)
        ]
        after = len(filtered_df)
        if before != after:
            filter_messages.append(f"Fechas: {start_date.date()} a {end_date.date()} ({after} reviews)")
    
    filtered_social = filter_social_data(
        social_df, 
        date_range, 
        selected_hashtags, 
        selected_platforms
    )
    
    final_count = len(filtered_df)
    if filter_messages:
        st.sidebar.success("Filtros aplicados:")
        for msg in filter_messages:
            st.sidebar.write(f"- {msg}")
        
        reduction_pct = ((initial_count - final_count) / initial_count) * 100
        st.sidebar.info(f"Resumen: {final_count:,} de {initial_count:,} reviews ({reduction_pct:.1f}% reduccion)")
    
    if filtered_social is not None:
        social_count = len(filtered_social)
        st.sidebar.info(f"Datos sociales filtrados: {social_count:,} registros")
    
    with st.spinner("Calculando metricas..."):
        temporal_metrics = calculate_temporal_metrics(filtered_df)
        brand_summary = calculate_brand_summary(filtered_df)
        product_metrics = calculate_product_metrics(filtered_df)
        social_metrics = calculate_social_metrics(filtered_social)
    
    if len(filtered_df) == 0:
        st.warning("No hay datos que coincidan con los filtros seleccionados")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_reviews = len(filtered_df)
        st.metric("Total Reseñas", f"{total_reviews:,}")
    
    with col2:
        avg_rating = filtered_df['rating'].mean()
        st.metric("Rating Promedio", f"{avg_rating:.2f}")
    
    with col3:
        avg_sentiment = filtered_df['sentiment_score'].mean()
        st.metric("Sentimiento Promedio", f"{avg_sentiment:.3f}")
    
    with col4:
        if 'product_id' in filtered_df.columns:
            unique_products = filtered_df['product_id'].nunique()
        else:
            unique_products = "N/A"
        st.metric("Productos Unicos", unique_products)
    
    create_alerts_section(filtered_df)
    
    st.header("Distribucion de Ratings y Sentimiento")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_rating = px.histogram(filtered_df, x='rating', nbins=5, 
                                 title='Distribucion de Ratings',
                                 color_discrete_sequence=['#FF6B9D'])
        st.plotly_chart(fig_rating, use_container_width=True)
    
    with col2:
        fig_sentiment = px.histogram(filtered_df, x='sentiment_score', 
                                    title='Distribucion de Sentimiento',
                                    color_discrete_sequence=['#36A2EB'])
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    enhanced_temporal_analysis(temporal_metrics)
    
    simple_forecasting(temporal_metrics)
    
    if brand_summary is not None and len(brand_summary) > 0:
        st.header("Analisis por Marca")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_brand_rating = px.bar(
                brand_summary.nlargest(10, 'avg_rating'),
                x='brand_name', y='avg_rating',
                title='Top 10 Marcas por Rating Promedio',
                color='avg_rating',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_brand_rating, use_container_width=True)
        
        with col2:
            fig_brand_volume = px.bar(
                brand_summary.nlargest(10, 'review_count'),
                x='brand_name', y='review_count',
                title='Top 10 Marcas por Numero de Reviews',
                color='review_count',
                color_continuous_scale='Plasma'
            )
            st.plotly_chart(fig_brand_volume, use_container_width=True)
    
    enhanced_customer_segmentation(filtered_df)
    
    if product_metrics is not None and len(product_metrics) > 0:
        st.header("Analisis de Productos")
        
        top_products = product_metrics.nlargest(10, 'avg_rating')
        
        fig_products = px.scatter(
            top_products,
            x='review_count',
            y='avg_rating',
            size='review_count',
            color='performance_category',
            hover_data=['product_name'],
            title='Top 10 Productos por Rating vs Volumen de Reviews',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_products, use_container_width=True)
    
    st.header("Analisis de Sentimiento por Categoria")
    
    category_cols = ['primary_category', 'secondary_category', 'brand_name']
    available_categories = [col for col in category_cols if col in filtered_df.columns]
    
    if available_categories:
        category_col = available_categories[0]
        
        sentiment_by_category = filtered_df.groupby(category_col).agg({
            'sentiment_score': 'mean',
            'rating': 'mean',
            'review_text': 'count'
        }).rename(columns={'review_text': 'review_count'}).reset_index()
        
        sentiment_by_category = sentiment_by_category.sort_values('sentiment_score', ascending=False)
        
        fig_sentiment_category = px.bar(
            sentiment_by_category.head(15),
            x=category_col,
            y='sentiment_score',
            title=f'Sentimiento Promedio por {category_col.replace("_", " ").title()}',
            color='sentiment_score',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_sentiment_category, use_container_width=True)
    
    statistical_analysis_section(filtered_df)
    
    display_social_media_analysis(filtered_social, social_metrics)
    
    enhanced_social_correlation(filtered_df, filtered_social)
    
    data_quality_metrics(filtered_df)
    
    st.header("Insights y Recomendaciones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Insights Clave")
        
        sentiment_rating_corr = filtered_df['sentiment_score'].corr(filtered_df['rating'])
        st.write(f"- Correlacion sentimiento-rating: {sentiment_rating_corr:.3f}")
        
        sentiment_dist = filtered_df['sentiment_label'].value_counts(normalize=True)
        st.write(f"- Reseñas positivas: {sentiment_dist.get('positive', 0)*100:.1f}%")
        st.write(f"- Reseñas negativas: {sentiment_dist.get('negative', 0)*100:.1f}%")
        st.write(f"- Reseñas neutrales: {sentiment_dist.get('neutral', 0)*100:.1f}%")
        
        rating_dist = filtered_df['rating'].value_counts().sort_index()
        st.write(f"- Rating promedio: {filtered_df['rating'].mean():.2f}/5")
        st.write(f"- Distribucion ratings: {dict(rating_dist)}")
    
    with col2:
        st.subheader("Recomendaciones Accionables")
        
        sentiment_rating_corr = filtered_df['sentiment_score'].corr(filtered_df['rating'])
        
        if sentiment_rating_corr < 0.5:
            st.write("- Mejorar analisis de sentimiento para capturar mejor la relacion con ratings")
        
        sentiment_dist = filtered_df['sentiment_label'].value_counts(normalize=True)
        if sentiment_dist.get('negative', 0) > 0.3:
            st.write("- Atender reseñas negativas - Mas del 30% requieren atencion")
        elif sentiment_dist.get('positive', 0) > 0.7:
            st.write("- Excelente percepcion - Capitalizar la satisfaccion del cliente")
        
        if filtered_df['rating'].mean() >= 4.0:
            st.write("- Productos de alta calidad - Destacar en estrategias de marketing")
        
        st.write("- Monitorear correlacion redes sociales para identificar tendencias")
        st.write("- Segmentar por clusters para personalizar experiencias")
        st.write("- Analizar productos top para replicar factores de exito")

if __name__ == "__main__":
    main()