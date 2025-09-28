# %%
import pandas as pd

df = pd.read_excel("data/dados_cerveja.xlsx")
df.head()

# %%
#Características
features = ["temperatura", "copo", "espuma", "cor"]
X = df[features]

#Resposta
target = 'classe'
y = df[target]

#Ajustando o modelo. Convertendo variáveis categóricas em numéricas
X = X.replace({
    "mud": 1, "pint": 2,
    "sim": 1, "não": 0,
    "clara": 0, "escura": 1,
})
X
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
                 filled=True)

# %%