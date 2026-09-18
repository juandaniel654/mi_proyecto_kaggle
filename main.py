import pandas as pd

print("--- 1. Carga de Datos (Titanic) ---")
df = pd.read_csv('train.csv')
print(f"Total de filas: {len(df)}")
print(df.head())

print("\n--- 2. Preprocesamiento de Datos ---")
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

print("Valores nulos despues del preprocesamiento:")
print(df.isnull().sum())

print("\n--- 3. Tasa de Supervivencia por Genero ---")
print(df.groupby('Sex')['Survived'].mean())
