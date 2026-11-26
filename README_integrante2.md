
# Integrante 2 - Preprocesamiento y PCA

## Proyecto

Clasificación del uso de gas en garrafa
en los hogares de Bolivia.

Fuente: Encuesta de Hogares 2023 - INE Bolivia.

## Datos

- Registros: 12815
- Columnas originales: 37
- Variables predictoras: 12
- Variable objetivo: usa_garrafa
- Hogares que usan garrafa: 6605
- Hogares que no usan garrafa como combustible principal: 6210

## Preprocesamiento

- Revisión de valores faltantes.
- Verificación de registros duplicados.
- Corrección de etiquetas de s06a_05.
- Codificación One-Hot de variables categóricas.
- Estandarización de variables numéricas.
- Generación de 62 características.

## Resultados de PCA

| Componentes | Varianza conservada |
|-------------|---------------------|
| 12          | 79.36%              |
| 10          | 74.82%              |
| 9           | 72.03%              |
| 5           | 57.25%              |
| 3           | 45.13%              |

## Orden de ejecución

1. notebooks/02_preprocesamiento.ipynb
2. notebooks/03_pca.ipynb

## Archivos principales

- src/preprocessing.py
- data/processed/datos_limpios.csv
- results/figures/
- results/tables/

## Uso del módulo

Desde la carpeta raíz del proyecto:

    from src.preprocessing import cargar_datos
    from src.preprocessing import crear_pipeline
    from sklearn.tree import DecisionTreeClassifier

    X, Y = cargar_datos(
        'data/processed/datos_limpios.csv'
    )

    pipeline = crear_pipeline(
        n_componentes=12,
        modelo=DecisionTreeClassifier(random_state=42)
    )

El pipeline debe ajustarse únicamente con los
datos de entrenamiento de cada partición.

Para trabajar sin PCA, utilizar:

    n_componentes=None

Para los experimentos con PCA, utilizar:

12, 10, 9, 5 o 3 componentes.

## Consideraciones metodológicas

- Partición inicial 80/20 con semilla 42.
- División estratificada de los datos.
- Estandarización únicamente de variables numéricas.
- Variables categóricas codificadas mediante One-Hot.
- PCA ajustado exclusivamente con entrenamiento.
- Experimentos realizados sin ponderación muestral.
- Corrección de etiquetas de s06a_05 respaldada
  por la comparación de frecuencias ponderadas oficiales.
- No se eliminaron hogares con características iguales
  porque sus folios originales eran únicos.

## Limitaciones

Una mayor varianza conservada no garantiza
un mejor rendimiento en clasificación.

Los datos corresponden a una encuesta transversal
del año 2023.

Los datos originales deben obtenerse del INE.
No redistribuir los CSV sin verificar sus permisos.
