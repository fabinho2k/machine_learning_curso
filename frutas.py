# %%
import pandas as pd

df = pd.read_excel("data/dados_frutas.xlsx")
df
# %%

from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42)

# %%

#Variável resposta
y = df['Fruta']

#Variável características
caracteristicas = ['Arredondada', 'Suculenta','Vermelha', 'Doce']
X = df[caracteristicas]

# %%
#Treinando o modelo
arvore.fit(X, y)

# %%

arvore.predict([[0, 0, 0, 0]])

# %%
import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(arvore, 
               feature_names=caracteristicas, 
               class_names=arvore.classes_,
                 filled=True)
# %%
