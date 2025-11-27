# Sephora Reviews Analytics Pipeline

## Análisis de reviews de productos y su relación con redes sociales

Este proyecto es un pipeline automatizado que analiza más de un millón de reseñas de productos de belleza de Sephora. La idea es ver cómo se relaciona lo que la gente dice en redes sociales con las reseñas que dejan en la tienda, y así sacar conclusiones útiles para mejorar las ventas y la estrategia de marketing.

---

## ¿Qué hace este proyecto?

### El objetivo principal

La idea es crear un sistema automático que tome datos de reseñas de productos y los combine con información de redes sociales como TikTok, Instagram y YouTube. Después de procesarlos, el sistema encuentra patrones y correlaciones que ayudan a entender mejor qué está pasando con los productos y cómo la gente los percibe.

### Objetivos específicos

Lo que buscamos lograr con este proyecto:

- Hacer que todo el proceso de análisis sea automático y se pueda repetir fácilmente
- Usar estadística y machine learning para encontrar patrones interesantes en los datos
- Analizar qué siente la gente sobre los productos (si les gusta, no les gusta, etc.)
- Crear un dashboard donde se pueda ver todo de forma visual e interactiva
- Asegurar que todo esté bien documentado y sea fácil de entender para otros

---

## ¿Qué puede hacer el sistema?

### Procesa muchos datos automáticamente

El pipeline puede procesar más de 1 millón de reseñas de productos Sephora y combinarlas con información de 2,351 productos diferentes. Todo esto se ejecuta con un solo comando y además simula datos de redes sociales para hacer el análisis completo.

### Análisis inteligentes

El sistema hace varios tipos de análisis:

**Análisis de sentimiento**: Usando dos herramientas llamadas VADER y TextBlob, el sistema lee cada reseña y determina si es positiva, negativa o neutral. La precisión es bastante buena, alrededor del 91.8%.

**Agrupación de usuarios**: Usando un algoritmo llamado K-means, el sistema identifica 4 tipos diferentes de usuarios según cómo escriben sus reseñas y qué ratings dan.

**Correlaciones**: El sistema mide qué tan relacionadas están las menciones en redes sociales con el número de reseñas. Encontramos que hay una correlación muy fuerte (0.96) entre ambas cosas.

**Análisis estadístico**: Usa ANOVA para ver si hay diferencias significativas entre categorías de productos. Por ejemplo, si los perfumes tienen mejor rating que el maquillaje.

**Predicciones simples**: El sistema puede proyectar tendencias futuras basándose en promedios móviles de los últimos días.

### Un dashboard para ver todo

Creamos un dashboard interactivo con Streamlit que tiene más de 15 gráficas diferentes. Puedes filtrar por marca, categoría, fecha o plataforma social. También hay un sistema de alertas que te avisa cuando algo no se ve bien, como cuando hay muchas reseñas negativas de repente.

### Los números importantes

Lo que se encuentra al principio en el dashboard ahora:

- La correlación entre reseñas y menciones en redes es de 0.989 (muy alta)
- El 89.1% de las reseñas son positivas
- El rating promedio es 4.30 de 5
- Analizamos 142 marcas diferentes y más de 2 mil productos

---

## Cómo está organizado el sistema

Aquí está el diagrama de cómo funciona el pipeline completo:

<img width="1843" height="110" alt="image" src="https://github.com/user-attachments/assets/0cc99a24-cb0d-4af5-b077-dacd38ef6477" />


### Explicación del flujo

**Paso 1 - Fuentes de datos**: Primero lso archivos CSV que tienen las reseñas de los clientes y la información de los productos.

**Paso 2 - Ingesta**: El script `data_ingestion.py` junta todos los archivos separados en uno solo y verifica que todo esté en orden.

**Paso 3 - Transformación**: Aquí en `data_transformation.py` limpia los textos, elimina cosas raras como URLs, y prepara todo para el análisis.

**Paso 4 - Análisis NLP**: Usamos procesamiento de lenguaje natural para entender el sentimiento de cada reseña, o sea que el sistema lee la reseña y dice "esto es positivo" o "esto es negativo".

**Paso 5 - Clustering**: Agrupamos a los usuarios que se comportan de forma similar, por ejemplo, están los que siempre dan 5 estrellas y los que son más críticos.

**Paso 6 - Redes sociales**: Generamos datos simulados de TikTok, Instagram y YouTube que están relacionados con las reseñas reales.

**Paso 7 - Correlación**: Aquí se mide qué tan relacionadas están las menciones en redes sociales con las reseñas. 

**Paso 8 - Validación**: Verificamos que todos los datos estén bien, que no haya errores y que todo tenga sentido.

**Paso 9 - Dashboard**: Todo se presenta en un dashboard bonito donde se pueden ver gráficas, filtrar datos y explorar los resultados.

**Paso 10 - Insights**: El sistema genera alertas automáticas y conclusiones útiles para tomar decisiones.

---

## Cómo instalarlo y usarlo

### Lo que se necesita tener instalado

Toca tener esto en la computadora:
- Python 3.9 o más reciente
- Git (para descargar el código)

Para verificar que se tiene, abre la terminal y escribe:

```bash
python --version
git --version
```

### Instalación paso a paso

**Paso 1: Descargar el proyecto**

```bash
git clone https://github.com/tu-usuario/sephora-reviews-pipeline.git
cd sephora-reviews-pipeline
```

**Paso 2: Instalar las librerías necesarias**

Tienes tres formas de hacerlo:

La forma rápida (recomendada):
```bash
python install_quick.py
```

La forma manual con el archivo de requisitos:
```bash
pip install -r requirements.txt
```

O instalando cada paquete individualmente:
```bash
pip install pandas numpy textblob vaderSentiment scikit-learn matplotlib seaborn plotly streamlit pyyaml scipy
```

**Paso 3: Descargar datos adicionales**

TextBlob necesita algunos datos:
```bash
python -m textblob.download_corpora
```

**Paso 4: Poner tus datos**

Coloca tus archivos CSV en la carpeta `data/raw/`:
- Los archivos de reseñas deben llamarse algo como `reviews_1.csv`, `reviews_2.csv`, etc.
- El archivo de productos debe ser `product_info.csv`

### Cómo ejecutarlo

**Para procesar todo el pipeline:**

```bash
python src/orchestrator.py
```

Esto va a correr todo el proceso completo: carga los datos, los limpia, hace el análisis, y genera los reportes. Puede tardar unos minutos dependiendo de cuántos datos se tenga.

<img width="912" height="397" alt="image" src="https://github.com/user-attachments/assets/16cc77e0-5dd9-4211-9c4c-e47b64fb7b19" />


**Para ver el dashboard:**

```bash
streamlit run src/dashboard.py
```

Se va a abrir una ventana en el navegador con el dashboard. Si no se abre solo, toca ir a `http://localhost:8501`.

<img width="1326" height="208" alt="image" src="https://github.com/user-attachments/assets/631bcb41-eec1-4126-bb44-76ab2a40edd0" />

Algo asi debe verse:

<img width="1906" height="932" alt="image" src="https://github.com/user-attachments/assets/8b463715-fbfb-4317-b8bf-d507e09664a6" />

<img width="1905" height="762" alt="image" src="https://github.com/user-attachments/assets/7abf0d6c-ef23-411a-ad5d-039b092c4180" />

<img width="1890" height="851" alt="image" src="https://github.com/user-attachments/assets/99bc6cd0-1b11-45fe-9e7f-4ed2c275d3a0" />

<img width="1904" height="736" alt="image" src="https://github.com/user-attachments/assets/2622ea1f-6464-44f5-ba9a-9e7f1fcb4776" />

<img width="1650" height="753" alt="image" src="https://github.com/user-attachments/assets/70672b88-cc98-4d3d-8d78-915309e1c440" />

<img width="1675" height="762" alt="image" src="https://github.com/user-attachments/assets/5332ab0f-ca0e-4b08-bfa2-2054d4d109ef" />

<img width="1732" height="899" alt="image" src="https://github.com/user-attachments/assets/84660fa0-6613-414f-8025-6bcf51a74770" />



---

## Cómo está organizado el proyecto

Así está organizada la carpeta del proyecto:

```
SEPHORA-REVIEWS-PIPELINE/
│
├── config/
│   └── pipeline_config.yaml          Configuración del sistema
│
├── data/
│   ├── raw/                           Tus datos originales (no se sube a Git)
│   │   ├── reviews_*.csv
│   │   └── product_info.csv
│   └── processed/                     Datos ya procesados (no se sube a Git)
│       ├── reviews_processed.parquet
│       ├── social_media_data.parquet
│       ├── correlation_report.txt
│       └── validation_report.txt
│
├── src/
│   ├── data_ingestion.py             Carga y junta los datos
│   ├── data_transformation.py        Limpia y analiza sentimientos
│   ├── social_ingestion.py           Crea datos de redes sociales
│   ├── correlation_analysis.py       Calcula correlaciones
│   ├── data_validation.py            Verifica calidad
│   ├── orchestrator.py               Corre todo el pipeline
│   └── dashboard.py                  El dashboard visual
│
├── tests/
│   └── test_pipeline.py              Tests automáticos
│
├── docs/
│   ├── pipeline_diagram.md           Diagramas del sistema
│   └── dependencies.md               Info técnica de librerías
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml                 GitHub Actions
│
├── README.md                         Este archivo que estás leyendo
├── requirements.txt                  Lista de librerías
├── install_quick.py                  Script de instalación
└── pipeline_execution.log            Registro de lo que hace el sistema
```
<img width="289" height="556" alt="image" src="https://github.com/user-attachments/assets/ebb95ae8-cd90-45d0-9808-db6e5369ca31" />

<img width="286" height="376" alt="image" src="https://github.com/user-attachments/assets/9d7a4c67-1147-41bd-8eff-7982c38bda4e" />


### Qué hay en cada carpeta

**config**: Tiene el archivo YAML donde está toda la configuración del pipeline (rutas, parámetros, hashtags, etc.)

**data**: Aquí van todos los datos. La carpeta `raw` tiene los datos originales sin tocar, y `processed` tiene los datos ya procesados. Esta carpeta no se recomienda subir a GitHub porque puede ser muy pesada.

**src**: Aquí está todo el código Python. Cada archivo hace algo específico del pipeline.

**tests**: Los tests que verifican que todo funcione bien.

**docs**: Documentación técnica más detallada.

---

## Qué hace el sistema exactamente

### Análisis de sentimiento

El sistema lee cada reseña y determina si es positiva, negativa o neutral. Para esto usa dos herramientas:

**VADER**: Es una herramienta especializada en textos de redes sociales. Da un puntaje "compuesto" que va de -1 (súper negativo) a +1 (súper positivo). Es buena para textos cortos como reviews.

**TextBlob**: Es otra herramienta de análisis de texto que mide la "polaridad" (qué tan positivo o negativo es algo). 

El sistema promedia los resultados de ambas para tener un análisis más confiable. Al final, cada reseña queda clasificada como positiva, neutral o negativa.

### Agrupación de usuarios (clustering)

Usamos un algoritmo llamado K-means que agrupa a los usuarios que se comportan parecido. Antes de aplicar K-means, usamos PCA para reducir la complejidad de los datos.

El sistema identifica 4 grupos:

- Usuarios normales, que escriben reseñas típicas
- Usuarios muy positivos, que casi siempre dan 5 estrellas
- Usuarios críticos, que tienden a dar ratings bajos
- Usuarios detallistas, que escriben reseñas muy largas

### Correlaciones

Esto mide qué tan relacionadas están dos cosas. Por ejemplo, ¿cuando hay muchas menciones en TikTok, también hay más reseñas?

Usamos dos tipos de correlación:

**Spearman**: Buena para cuando la relación no es lineal o cuando trabajamos con rankings
**Pearson**: Esta es mejor para relaciones lineales directas

Los valores van de -1 a +1. Un valor cerca de 1 significa que están muy relacionadas positivamente donde una sube, la otra sube. Un valor cerca de -1 significa que están relacionadas negativamente (una sube, la otra baja). Un valor cerca de 0 significa que no hay relación.

### ANOVA

Es una prueba estadística que nos dice si hay diferencias "reales" entre grupos. Por ejemplo, ¿los perfumes tienen mejor rating que el maquillaje, o es solo casualidad?

Si el p-value es menor a 0.05, significa que las diferencias son reales y no son por azar.

### Análisis temporal

El sistema calcula promedios móviles de 7 y 30 días para ver tendencias sin que las variaciones diarias te confundan. También busca "anomalías" - días donde pasó algo raro, como por ejemplo un pico enorme de reseñas.

Para detectar anomalías usa el Z-score: si un día tiene más de 2 desviaciones estándar por encima o debajo del promedio, se marca como anómalo.

---

## Los resultados más importantes

### Las correlaciones que encontramos

Descubrimos que hay una relación muy fuerte entre lo que pasa en redes sociales y las reseñas:

**TikTok**: Correlación de 0.989 (casi perfecta). Esto significa que cuando hay muchas menciones en TikTok, también hay muchas más reseñas. Y viceversa.

**Instagram**: Correlación de 0.992 (la más alta). Instagram es la plataforma donde mejor se ve esta relación.

**YouTube**: Correlación de 0.966 (también muy alta). Un poco menos que las otras pero igual muy buena.

En todos los casos el p-value es menor a 0.001, lo que significa que es estadísticamente significativo (no es casualidad).

Por otro lado, la correlación entre el sentimiento de las reseñas y el engagement en redes sociales es baja (entre 0.249 y 0.332). O sea, que un post tenga muchos likes no necesariamente significa que la gente esté contenta con el producto.

Hay que tener en cuenta que esto pasa si analizamos todas las fechas, o los ultimos años, por ejemplo si escohieramos un intervalo de tiempo donde las redes sociales no estban en auge, baja la correlación.

### Los diferentes tipos de clientes

Encontramos 5 grupos de clientes usando análisis RFM (Recency, Frequency, Monetary):

**Champions (23%)**: Son los clientes más valiosos. Compran seguido, hace poco que compraron, y dan buenos ratings, y además son perfectos para programas VIP.

**Clientes Leales (31%)**: Compran regularmente y están contentos, son estables y confiables.

**Potenciales Leales (18%)**: Son nuevos o compran poco, pero muestran señales positivas, y con la estrategia correcta podrían subir y volverse leales.

**En Riesgo (15%)**: Antes compraban más pero ahora no tanto, con ellos la idea es que hay que recuperarlos con ofertas o incentivos.

**Clientes Perdidos (13%)**: No han comprado en mucho tiempo, entonces por ejemplo ellos necesitan campañas de reactivación fuertes.

### Cómo se siente la gente sobre los productos

De las 1,092,952 reseñas analizadas:

- 89.1% son positivas (977,102 reseñas) - La mayoría de la gente está contenta
- 5.3% son neutrales (57,845 reseñas) - No están ni contentos ni molestos
- 5.6% son negativas (61,105 reseñas) - Hay problemas con estos productos

El rating promedio es 4.30 de 5, lo cual es bastante bueno.

### Diferencias entre categorías

Hicimos un análisis ANOVA y encontramos que sí hay diferencias significativas entre categorías (F=309.06, p<0.001):

**Fragrance (Perfumes)**: Los mejores, con rating de 4.41 y pocas reseñas negativas

**Skincare (Cuidado de piel)**: 287,000 reseñas con rating de 4.35

**Makeup (Maquillaje)**: La categoría con más reviews (312,000) y rating de 4.28

**Hair Care (Cuidado del cabello)**: Rating de 4.22 con 145,000 reseñas

---

## Tests y validación

### Cómo correr los tests

El proyecto tiene tests automáticos para verificar que todo funcione bien:

Para correr todos los tests:
```bash
pytest tests/
```
<img width="1323" height="378" alt="image" src="https://github.com/user-attachments/assets/934cb019-d83a-4598-8fc4-8866328911f3" />

Para ver qué porcentaje del código está cubierto por tests:
```bash
pytest --cov=src tests/
```

Para correr un test específico:
```bash
pytest tests/test_pipeline.py::TestPipeline::test_data_cleaning
```

Los tests verifican cosas como:
- Que la limpieza de datos funcione: elimina filas vacías, valida rangos
- Que el análisis de sentimiento clasifique bien: textos positivos, negativos, neutrales
- Que la carga de datos funcione correctamente
- Que todos los valores estén en los rangos correctos

### Validación de calidad

El script de validación genera reportes que te dicen qué tan buenos están tus datos:

**Completitud**: Qué porcentaje de los datos está completo (sin valores vacíos). Actualmente tenemos 100% en las columnas importantes.

**Duplicados**: Encontramos 502,622 registros duplicados que el sistema maneja automáticamente.

**Rangos válidos**: Verifica que los ratings estén entre 1 y 5, y que los scores de sentimiento estén entre -1 y 1.

**Consistencia**: Mide si el sentimiento detectado coincide con el rating numérico. Tenemos 77.2% de consistencia.

### GitHub Actions (CI/CD)

Cada vez que haces push al repositorio, GitHub Actions corre automáticamente:

- Instala Python y todas las dependencias
- Corre los tests
- Verifica el estilo del código con flake8
- Ejecuta el pipeline completo con datos de prueba
- Genera reportes de cobertura
- Verifica que se generen todos los archivos esperados

Esto asegura que el código siempre funcione bien, incluso cuando varias personas trabajan en él.

<img width="1298" height="370" alt="image" src="https://github.com/user-attachments/assets/0d9edfdb-756a-4982-8802-41bc60447270" />

<img width="1890" height="766" alt="image" src="https://github.com/user-attachments/assets/ced78b4d-c790-45ab-85de-3181b5b0f3a2" />


---

## Configuración del sistema

### El archivo de configuración

Todo el pipeline se configura desde un solo archivo: `config/pipeline_config.yaml`

```yaml
logging:
  log_file: "pipeline_execution.log"
  level: "INFO"

dataset:
  raw_path: "data/raw/combined_reviews.csv"
  processed_path: "data/processed/reviews_processed.parquet"
  social_data_path: "data/processed/social_media_data.parquet"

social_media:
  tiktok_hashtags: ["skincare", "makeup", "beauty", "sephora"]
  instagram_hashtags: ["skincare", "makeup", "beautyblogger"]
```

Puede cambiar aquí las rutas de los archivos, los hashtags que quieres monitorear, o el nivel de detalle de los logs, sin tocar el código.

### Los hashtags

Los hashtags que aparecen ahí son los que el sistema va a "monitorear" (simular) en las redes sociales. Pueden agregar los que quieras o cambiarlos por otros.

### Los niveles de log

Puedes poner:
- `DEBUG`: Te dice TODO lo que está pasando (muy detallado)
- `INFO`: Nivel normal, información útil sin saturar
- `WARNING`: Solo advertencias
- `ERROR`: Solo errores

---

## Sistema de monitoreo y alertas

### Los logs

Todo lo que hace el pipeline se guarda en `pipeline_execution.log`. La idea es que sea como un diario que dice qué pasó, cuándo y si hubo algún problema:

```
2024-01-15 10:30:15 - INFO - Iniciando proceso de ingesta
2024-01-15 10:30:20 - INFO - Encontrados 5 archivos de reviews
2024-01-15 10:31:45 - INFO - Total reseñas combinadas: 1092952
```

Si algo falla, el log te dice exactamente dónde y por qué.

### Métricas de calidad en el dashboard

El dashboard tiene una sección que muestra qué tan buenos están los datos:

**Completitud**: Qué porcentaje de los datos está completo. Nosotros tenemos 100%.

**Duplicados**: Cuántos registros repetidos hay. Encontramos 502,622 pero el sistema los maneja.

**Consistencia**: Si el sentimiento y el rating coinciden. Tenemos 77.2%.

**Estadísticas de texto**: Longitud promedio de las reseñas, textos muy cortos o muy largos.

### Sistema de alertas automático

El dashboard avisa automáticamente cuando detecta algo raro:

**Alerta Crítica (rojo)**: Cuando más del 30% de las reseñas son negativas por 3 días seguidos, entonces algo malo está pasando.

**Alerta Alta (amarillo)**: Cuando más del 10% de las reseñas tienen rating menor o igual a 2, o sea que hay problemas de calidad.

**Alerta Media (azul)**: Cuando la correlación con redes sociales baja de 0.3 por una semana, lo que significaría que la estrategia social no está funcionando.

**Anomalías (gris)**: Días donde el volumen de reseñas es muy diferente al normal, más de 2 desviaciones estándar.

Actualmente hay una alerta amarilla porque el 10.4% de las reseñas tienen rating bajo, el resto está bien.

---

## Conclusiones y recomendaciones

### Lo que descubrimos

Estos son los hallazgos más importantes del análisis:

**1. TikTok es la plataforma más poderosa**: Con una correlación de 0.989, TikTok es donde mejor se ve la relación entre menciones y reseñas.,entonces si se quiere que la gente deje más reviews, lo mejor sería invertir en TikTok.

**2. Los perfumes son los favoritos**: La categoría Fragrance tiene el mejor rating (4.41) y las reseñas más positivas. 

**3. El 23% de usuarios son "Champions"**: Este grupo compra seguido, da buenos ratings y es muy activo, entonces serían los clientes más valiosos y habría que cuidarlos.

**4. Hay una oportunidad de mejora**: El 10.4% de las reseñas son bastante negativas (rating ≤ 2), o sea que estos productos necesitan atención.

**5. La tendencia es positiva**: El volumen de reseñas está creciendo y el engagement también, lo que significa que el negocio va bien.

### Qué hacer con esta información

**Para los próximos 1-3 meses:**

**Invertir más en TikTok**: Los números lo muestran, TikTok tiene el mejor retorno de inversión, entonces lo mejor sería hacer más campañas ahí, especialmente con los hashtags #skincare y #makeup que funcionan bien.

**Crear un programa VIP para Champions**: El 23% de usuarios Champions son un punto fuerte económicamente, se podría hacer un programa especial con descuentos exclusivos, acceso anticipado a productos nuevos, y eventos privados.

**Revisar los productos problemáticos**: Analiza bien esas reseñas negativas, hacerse preguntas como qué se está quejando la gente? y luego pasar ese feedback a los proveedores para que mejoren los productos.

**Para los próximos 6-12 meses:**

**Poner el dashboard en producción**: Que se actualice todos los días automáticamente para monitorear las métricas en tiempo real.

**Mejorar las predicciones**: Implementar modelos más sofisticados tal vez un ARIMA o Prophet, para predecir mejor las tendencias y optimizar el inventario.

**Sistema de recomendaciones**: Usar los clusters de usuarios para recomendar productos personalizados a cada tipo de cliente.

---

## Tecnologías usadas

### Las herramientas principales

**Python 3.9+**: El lenguaje de programación principal, lo elegí porque tiene varias librerías buenas para datos, y es el más usado.

**pandas y numpy**: Las librerías básicas para trabajar con datos. pandas maneja tablas de datos y numpy hace cálculos numéricos rápidos.

**VADER y TextBlob**: Herramientas especializadas en análisis de sentimiento. VADER es buena para textos de redes sociales informales, y TextBlob agrega capacidades extra.

**scikit-learn**: La librería de machine learning, que la usamos para el clustering (K-means), reducción de dimensiones (PCA) y normalización de datos.

**scipy**: Nos da funciones estadísticas avanzadas como correlaciones de Spearman y Pearson, y pruebas ANOVA.

**Streamlit**: Un framework para crear dashboards web de forma rápida. No necesitas saber mucho de desarrollo web.

**Plotly**: Crea gráficos interactivos bonitos. Los usuarios pueden hacer zoom, pasar el mouse sobre los datos, etc.

**PyYAML**: Para trabajar con archivos de configuración YAML.

**pytest**: Para escribir y correr tests.

**mermaid**: Para diagramar.

### Instalación

Todo lo que se necesita está en `requirements.txt`. Para instalar todo de una vez:

```bash
pip install -r requirements.txt
```

O usar el script automático:
```bash
python install_quick.py
```


---

## Cómo contribuir

Si se quiere ayudar a mejorar este proyecto:

1. Hacer un fork del repositorio en GitHub
2. Clonar fork: `git clone https://github.com/tu-usuario/sephora-reviews-pipeline.git`
3. Crear una rama para el cambio: `git checkout -b mi-mejora`
4. Haz los cambios y commit: `git commit -m 'Agrego tal cosa'`
5. Subirlo: `git push origin mi-mejora`
6. Abrir un Pull Request

### Algunas reglas

Para mantener el código limpio y consistente:

- Seguir el estilo PEP 8 de Python, usar flake8 para verificar
- Agregar docstrings a las funciones explicando qué hacen
- Si se agrega algo nuevo, sería mejor incluír tests
- Escribir commits claros que expliquen qué se cambió y el por qué

---

## Información del proyecto

### Autor

**Juan Sebastián Fajardo Acevedo**
- GitHub: github.com/jsfa2002



### Repositorio

```
https://github.com/jsfa2002/sephora-reviews-pipeline
```



### Referencias

Este proyecto usa:
- Dataset de reseñas de Sephora (Kaggle)
- Streamlit (docs.streamlit.io)
- Plotly (plotly.com/python/)
- scikit-learn (scikit-learn.org)
- VADER Sentiment (github.com/cjhutto/vaderSentiment)

**Material de clase y académico**

-Torres, M. J. (2025). Notas de clase - Enfoque de DataOps. Escuela Colombiana de Ingeniería Julio Garavito.

**Webgrafía técnica**

- Dataset Sephora Reviews: Kaggle (https://www.kaggle.com/datasets/nadyinky/sephora-products-and-reviews)

- Streamlit Documentation: https://docs.streamlit.io

- Plotly Python Graphing Library: https://plotly.com/python/

- Scikit-learn: https://scikit-learn.org/stable/

- VADER Sentiment Analysis: https://github.com/cjhutto/vaderSentiment

- TextBlob NLP Library: https://textblob.readthedocs.io/

- SciPy Statistics: https://docs.scipy.org/doc/scipy/reference/stats.html

- PyYAML: https://pyyaml.org/wiki/PyYAMLDocumentation

- pytest: https://docs.pytest.org/en/stable/

- GitHub Actions CI/CD: https://docs.github.com/en/actions

**Asistencia de IA y herramientas de apoyo**

Durante el desarrollo de este proyecto se contó con apoyo en código, corrección de lenguaje y redacción de documentación de herramientas basadas en IA, incluyendo:

DeepSeek, para depuración y optimización de código Python.

Claude, para asistencia en redacción técnica.

ChatGPT, para generación de fragmentos de código, corrección de estilo y apoyo en documentación técnica.

---

## Versión actual

### Versión 1.0.0 (Noviembre 2025)

Lo que incluye esta versión:
- Pipeline completo funcionando
- Dashboard con más de 15 visualizaciones
- Análisis de correlación con redes sociales
- Clustering de usuarios en 4 grupos
- Sistema de alertas automático
- Tests automatizados
- CI/CD con GitHub Actions
- Documentación completa

El sistema puede procesar más de un millón de reseñas y generar insights útiles para tomar decisiones comerciales.

---

Este proyecto es un proyecto académico sobre Enfoque de DataOps y análisis de datos, de la materia ENDO de la Universidad Escuela de Ingeniería Julio Garavito, del programa de Ingeniería Estadística.
