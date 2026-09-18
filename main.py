import pandas as pd

# 1. Cargar el dataset
print("--- 1. Carga de Datos (Titanic) ---")
df = pd.read_csv('train.csv')
print(f"Total de filas: {len(df)}")
print(df.head())

# 2. Exploración y Limpieza Corregida
print("\n--- 2. Preprocesamiento de Datos ---")
# Usar asignación directa para evitar advertencias de pandas
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Eliminar columna Cabin por exceso de nulos
if 'Cabin' in df.columns:
    df = df.drop(columns=['Cabin'])

print("Valores nulos despues del preprocesamiento:")
print(df.isnull().sum())

# 3. Resumen Final
print("\n--- 3. Tasa de Supervivencia por Genero ---")
print(df.groupby('Sex')['Survived'].mean())
