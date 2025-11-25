import pytest
import pandas as pd
import os
import sys

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from data_ingestion import load_and_merge_reviews, load_product_info
from data_transformation import clean_data, advanced_sentiment_analysis

class TestPipeline:
    def test_data_ingestion(self):
        """Test basico de ingesta de datos"""
        # Este test require datos de prueba
        pass
    
    def test_data_cleaning(self):
        """Test de limpieza de datos"""
        # Crear datos de prueba
        test_data = pd.DataFrame({
            'review_text': ['Great product!', 'Not good', 'Average product', '', '   '],
            'rating': [5, 1, 3, None, 2],
            'product_id': [1, 2, 3, 4, 5]
        })
        
        cleaned = clean_data(test_data)
        
        # Verificar que se eliminen filas vacias
        assert len(cleaned) == 3  # 2 filas deben ser eliminadas
        
        # Verificar que los ratings esten en rango
        assert cleaned['rating'].between(1, 5).all()
    
    def test_sentiment_analysis(self):
        """Test de analisis de sentimiento"""
        test_data = pd.DataFrame({
            'cleaned_text': ['I love this product!', 'This is terrible', 'It is okay'],
            'rating': [5, 1, 3]
        })
        
        result = advanced_sentiment_analysis(test_data)
        
        # Verificar que se calculen las columnas de sentimiento
        assert 'sentiment_score' in result.columns
        assert 'sentiment_label' in result.columns
        
        # Verificar que los scores esten en rango
        assert result['sentiment_score'].between(-1, 1).all()

if __name__ == '__main__':
    pytest.main([__file__])