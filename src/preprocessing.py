
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


# Variables predictoras originales

VARIABLES_PREDICTORAS = [
    'depto',
    'area',
    'totper',
    's06a_01',
    's06a_02',
    's06a_03',
    's06a_04',
    's06a_05',
    's06a_06',
    's06a_07',
    's06a_12',
    's06a_16'
]

VARIABLE_OBJETIVO = 'usa_garrafa'

VARIABLES_NUMERICAS = [
    'totper',
    's06a_16'
]

VARIABLES_CATEGORICAS = [
    variable
    for variable in VARIABLES_PREDICTORAS
    if variable not in VARIABLES_NUMERICAS
]


# Función para cargar los datos limpios

def cargar_datos(ruta_csv):

    datos = pd.read_csv(ruta_csv)

    X = datos[VARIABLES_PREDICTORAS].copy()

    Y = datos[VARIABLE_OBJETIVO].copy()

    if X.isnull().any().any() or Y.isnull().any():
        raise ValueError("Existen valores faltantes")

    if not Y.isin([0, 1]).all():
        raise ValueError("Variable objetivo no válida")

    return X, Y


# Función para crear el preprocesador

def crear_preprocesador():

    return ColumnTransformer(
        transformers=[
            (
                'numericas',
                StandardScaler(),
                VARIABLES_NUMERICAS
            ),
            (
                'categoricas',
                OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False
                ),
                VARIABLES_CATEGORICAS
            )
        ],
        remainder='drop'
    )


# Función para crear un pipeline

def crear_pipeline(n_componentes=None, modelo=None):

    pasos = [
        ('preprocesamiento', crear_preprocesador())
    ]

    if n_componentes is not None:

        pasos.append(
            ('pca', PCA(
                n_components=n_componentes,
                svd_solver='full'
            ))
        )

    if modelo is not None:

        pasos.append(
            ('modelo', modelo)
        )

    return Pipeline(pasos)
