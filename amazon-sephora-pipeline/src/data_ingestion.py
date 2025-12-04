import os
import pandas as pd
import yaml
import logging
from glob import glob
from datetime import datetime

# Cargar configuracion
with open("config/pipeline_config.yaml") as f:
    config = yaml.safe_load(f)

# Configurar directorios
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Configurar logging
logging.basicConfig(
    filename=config['logging']['log_file'],
    level=getattr(logging, config['logging']['level']),
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def validate_review_files():
    """Valida que existan archivos de reviews antes de procesar"""
    review_files = glob("data/raw/reviews_*.csv")
    if not review_files:
        raise FileNotFoundError("No se encontraron archivos de reviews en data/raw/")
    
    logging.info(f"Encontrados {len(review_files)} archivos de reviews")
    return review_files

def load_and_merge_reviews():
    """
    Une automaticamente todos los archivos de reseñas divididos
    y realiza validaciones basicas
    """
    review_files = validate_review_files()
    
    dfs = []
    for file_path in review_files:
        try:
            logging.info(f"Procesando archivo: {file_path}")
            df = pd.read_csv(file_path)
            
            # Validaciones basicas del archivo
            if df.empty:
                logging.warning(f"Archivo vacio: {file_path}")
                continue
                
            required_cols = ['review_text', 'rating', 'product_id']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logging.warning(f"Archivo {file_path} falta columnas: {missing_cols}")
                continue
                
            dfs.append(df)
            logging.info(f"Archivo {file_path} cargado: {df.shape[0]} filas")
            
        except Exception as e:
            logging.error(f"Error procesando {file_path}: {str(e)}")
            continue
    
    if not dfs:
        raise ValueError("No se pudieron cargar archivos validos")
    
    # Combinar todos los DataFrames
    merged_df = pd.concat(dfs, ignore_index=True, sort=False)
    logging.info(f"Total reseñas combinadas: {merged_df.shape[0]} filas, {merged_df.shape[1]} columnas")
    
    return merged_df

def load_product_info():
    """
    Carga y valida la informacion de productos
    """
    product_path = "data/raw/product_info.csv"
    
    if not os.path.exists(product_path):
        logging.warning("Archivo product_info.csv no encontrado")
        return None
    
    try:
        products_df = pd.read_csv(product_path)
        logging.info(f"Informacion de productos cargada: {products_df.shape[0]} productos")
        
        # Validar columnas requeridas
        required_product_cols = ['product_id', 'product_name', 'brand_name']
        missing_product_cols = [col for col in required_product_cols if col not in products_df.columns]
        
        if missing_product_cols:
            logging.warning(f"Faltan columnas en product_info: {missing_product_cols}")
            return None
            
        return products_df
        
    except Exception as e:
        logging.error(f"Error cargando product_info: {str(e)}")
        return None

def merge_datasets(reviews_df, products_df):
    """
    Fusiona reseñas con informacion de productos
    """
    if products_df is not None and 'product_id' in reviews_df.columns and 'product_id' in products_df.columns:
        # Hacer merge
        merged_df = reviews_df.merge(
            products_df, 
            on='product_id', 
            how='left',
            suffixes=('', '_product')
        )
        
        # Estadisticas del merge
        matched_count = merged_df['product_name'].notna().sum()
        match_percentage = (matched_count / len(merged_df)) * 100
        
        logging.info(f"Merge completado: {match_percentage:.1f}% de reseñas con informacion de producto")
        
        return merged_df
    else:
        logging.warning("No se pudo realizar merge con informacion de productos")
        return reviews_df

def save_combined_dataset(df):
    """
    Guarda el dataset combinado
    """
    output_path = config['dataset']['raw_path']
    
    try:
        df.to_csv(output_path, index=False)
        logging.info(f"Dataset combinado guardado en: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error guardando dataset combinado: {str(e)}")
        return False

def generate_ingestion_report(df, products_df):
    """
    Genera reporte de ingesta
    """
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_reviews': len(df),
        'total_products': df['product_id'].nunique() if 'product_id' in df.columns else 0,
        'date_range': {
            'start': df['submission_time'].min() if 'submission_time' in df.columns else 'N/A',
            'end': df['submission_time'].max() if 'submission_time' in df.columns else 'N/A'
        } if 'submission_time' in df.columns else {},
        'rating_stats': {
            'mean': df['rating'].mean() if 'rating' in df.columns else 'N/A',
            'min': df['rating'].min() if 'rating' in df.columns else 'N/A',
            'max': df['rating'].max() if 'rating' in df.columns else 'N/A'
        }
    }
    
    if products_df is not None:
        report['product_info_available'] = True
        report['unique_brands'] = products_df['brand_name'].nunique() if 'brand_name' in products_df.columns else 0
    else:
        report['product_info_available'] = False
    
    logging.info(f"Reporte de ingesta: {report}")
    return report

if __name__ == "__main__":
    try:
        logging.info("=== INICIANDO PROCESO DE INGESTA ===")
        
        # Cargar y combinar reviews
        reviews_df = load_and_merge_reviews()
        
        # Cargar informacion de productos
        products_df = load_product_info()
        
        # Fusionar datasets
        combined_df = merge_datasets(reviews_df, products_df)
        
        # Guardar dataset combinado
        if save_combined_dataset(combined_df):
            # Generar reporte
            report = generate_ingestion_report(combined_df, products_df)
            
            print("Ingesta completada exitosamente!")
            print(f"Total reseñas procesadas: {report['total_reviews']}")
            print(f"Productos unicos: {report['total_products']}")
            
            if report.get('product_info_available'):
                print(f"Marcas unicas: {report['unique_brands']}")
                
        else:
            print("Error en la ingesta - Verifica los logs")
            exit(1)
            
    except Exception as e:
        logging.error(f"Error critico en ingesta: {str(e)}")
        print(f"Error critico: {str(e)}")
        exit(1)