# %%
import pandas as pd

df = pd.read_parquet('data/dados_clones.parquet')

df.rename(columns={'p2o_master_id': 'ID'}, inplace=True)
df['General Jedi encarregado'].unique()
df

#%%
df.columns

# %%

#Resposta
target = "Status "
y = df[target]

# %%

#Características
features = ["Massa(em kilos)",	"Estatura(cm)",	"Distância Ombro a ombro",	"Tamanho do crânio"	,"Tamanho dos pés",	"Tempo de existência(em meses)"]
X = df[features]

#Ajustando o modelo. Convertendo variáveis categóricas em numéricas
X = X.replace({
    "Tipo 1": 1, "Tipo 2": 2, "Tipo 3": 3, "Tipo 4": 4, "Tipo 5": 5,

})


# %%

from sklearn import tree

model = tree.DecisionTreeClassifier(random_state=42)
model.fit(X=X, y=y)

# %%


import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(model, 
               feature_names=features, 
               class_names=model.classes_,
               max_depth=3,
                 filled=True)

# %%
