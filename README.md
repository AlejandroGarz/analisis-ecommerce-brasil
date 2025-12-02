# Análisis E-Commerce Brasileño - Parte 1: Comprensión del Negocio y Datos

Este proyecto contiene un análisis completo de la base de datos "Brazilian E-Commerce Public Dataset by Olist" de Kaggle, enfocado en la comprensión del negocio y la comprensión de los datos.

## Dataset

**Nombre:** Brazilian E-Commerce Public Dataset by Olist  
**Origen:** Kaggle (olistbr/brazilian-ecommerce)  
**Período:** 2016-2018  
**Volumen:** ~100,000 órdenes de compra

## Contenido del Análisis

### 1. Comprensión del Negocio
- Definición del problema de negocio
- Relevancia del problema
- Objetivos del análisis

### 2. Comprensión de los Datos
- Presentación del dataset
- Estructura de datos y relaciones
- Análisis de calidad de datos
- Estadísticas descriptivas extensas
- Análisis exploratorio inicial
- Visualizaciones descriptivas (33+ gráficas)

## Configuración

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar Kaggle API

Para descargar el dataset, necesitas configurar la API de Kaggle:

1. Ve a tu perfil de Kaggle: https://www.kaggle.com/account
2. En la sección "API", haz clic en "Create New Token"
3. Esto descargará un archivo `kaggle.json`
4. Coloca este archivo en una de las siguientes ubicaciones:
   - **Windows:** `C:\Users\<username>\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`

**Nota:** Asegúrate de que el archivo tenga los permisos correctos:
- **Linux/Mac:** `chmod 600 ~/.kaggle/kaggle.json`

### 3. Ejecutar el Notebook

Abre `analisis_ecommerce_brazil.ipynb` en Jupyter Notebook o JupyterLab y ejecuta todas las celdas.

## Estructura del Proyecto

```
.
├── analisis_ecommerce_brazil.ipynb  # Notebook principal con el análisis
├── requirements.txt                  # Dependencias del proyecto
├── README.md                         # Este archivo
└── .gitignore                        # Archivos a ignorar en git
```

## Problemas de Negocio Analizados

1. ¿Qué factores afectan la satisfacción del cliente (review score)?
2. ¿Cómo reducir los tiempos de entrega?
3. ¿Qué categorías de productos generan más ventas?
4. ¿Qué vendedores tienen mejor desempeño?
5. Predicción de cancelaciones de órdenes

## Autor

Análisis realizado como parte de un proyecto de análisis de datos.
