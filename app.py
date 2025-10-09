import streamlit as st
import pandas as pd

model = pd.read_pickle("model_feliz.pkl")

st.markdown("## Descubra a felicidade")

redes_options = ['LinkedIn', 'Twitch', 'YouTube', 'Instagraam', 'Amigos', 'Twitter / X', 'Outra rede social']
redes = st.selectbox("Como conheceu o Téo Me Why?", options=redes_options)

cursos_options = ['0', '1', '2', '3', 'Mais que 3']
cursos = st.selectbox("Quantos cursos acompanhou do Téo Me Why?", options=cursos_options)
 


col1, col2, col3 = st.columns(3)

with col1:
    video_game = st.radio("Curte Video Game?", ["Sim", "Não"])
    futebol = st.radio("Curte Futebol?", ["Sim", "Não"])
    idade = st.number_input("Sua idade", 18, 100)
    
    areas_formacao = ['Biológicas', 'Exatas', 'Humanas']
    tempo_option = ['De 0 a 6 meses', 'De 1 ano a 2 anos',
                'De 6 meses a 1 ano', 'Mais de 4 anos',
                'Não atuo', 'de 2 anos a 4 anos']

    tempo = st.selectbox("Tempo que atua na área de dados", options=tempo_option)



with col2:
    livros = st.radio("Curte Livros?", ["Sim", "Não"])
    tabuleiro = st.radio("Curte jogos de tabuleiro?", ["Sim", "Não"])

    ufs_options = ['AM', 'BA', 'CE', 'DF', 'ES', 'GO',
                   'MA', 'MG', 'MT', 'PA', 'PB', 'PE',
                   'PR', 'RJ', 'RN', 'RS', 'SC', 'SP']
    ufs = st.selectbox("Estado que mora atualmente", options=ufs_options)

    niveis_options = [
    'Iniciante', "Júnior", 'Pleno', 'Sênior',
    'Especialista', 'Cooerdenação',
    'Gerência', 'Diretoria', 'C-Level']

    nivel_cadeira = st.selectbox("Posição da cadeira (senioridade)", options=niveis_options)


with col3:
    f1 = st.radio("Curte jogos de Fórmula 1?", ["Sim", "Não"])
    MMA = st.radio("Curte jogos de MMA?", ["Sim", "Não"])
    formacao = st.selectbox('Área de Formação', options=areas_formacao)


data = {
    'Como conheceu o Téo Me Why?': redes,
    'Quantos cursos acompanhou do Téo Me Why?': cursos,
    'Curte games?': video_game,
    'Curte futebol?': futebol, 
    'Curte livros?': livros, 
    'Curte jogos de tabuleiro?': tabuleiro,
    'Curte jogos de fórmula 1?': f1, 
    'Curte jogos de MMA?': MMA, 
    'Idade': idade,
    'Estado que mora atualmente': ufs, 
    'Área de Formação': formacao,
    'Tempo que atua na área de dados': tempo, 
    'Posição da cadeira (senioridade)': nivel_cadeira,
}


df = pd.DataFrame([data]).replace({"Sim": 1, "Não": 0})

dummy_vars =[
    'Como conheceu o Téo Me Why?',
    'Quantos cursos acompanhou do Téo Me Why?',
    'Estado que mora atualmente',
    'Área de Formação',
    'Tempo que atua na área de dados',
    'Posição da cadeira (senioridade)'
]

df = pd.get_dummies(df[dummy_vars]).astype(int)

df_template = pd.DataFrame(columns=['Como conheceu o Téo Me Why?_Amigos',
       'Como conheceu o Téo Me Why?_Instagram',
       'Como conheceu o Téo Me Why?_LinkedIn',
       'Como conheceu o Téo Me Why?_Outra rede social',
       'Como conheceu o Téo Me Why?_Twitch',
       'Como conheceu o Téo Me Why?_Twitter / X',
       'Como conheceu o Téo Me Why?_YouTube',
       'Quantos cursos acompanhou do Téo Me Why?_0',
       'Quantos cursos acompanhou do Téo Me Why?_1',
       'Quantos cursos acompanhou do Téo Me Why?_2',
       'Quantos cursos acompanhou do Téo Me Why?_3',
       'Quantos cursos acompanhou do Téo Me Why?_Mais que 3',
       'Estado que mora atualmente_AM', 
       'Estado que mora atualmente_BA',
       'Estado que mora atualmente_CE', 
       'Estado que mora atualmente_DF',
       'Estado que mora atualmente_ES', 
       'Estado que mora atualmente_GO',
       'Estado que mora atualmente_MA', 
       'Estado que mora atualmente_MG',
       'Estado que mora atualmente_MT', 
       'Estado que mora atualmente_PA',
       'Estado que mora atualmente_PB', 
       'Estado que mora atualmente_PE',
       'Estado que mora atualmente_PR', 
       'Estado que mora atualmente_RJ',
       'Estado que mora atualmente_RN', 
       'Estado que mora atualmente_RS',
       'Estado que mora atualmente_SC', 
       'Estado que mora atualmente_SP',
       'Área de Formação_Biológicas', 
       'Área de Formação_Exatas',
       'Área de Formação_Humanas',
       'Tempo que atua na área de dados_De 0 a 6 meses',
       'Tempo que atua na área de dados_De 1 ano a 2 anos',
       'Tempo que atua na área de dados_De 6 meses a 1 ano',
       'Tempo que atua na área de dados_Mais de 4 anos',
       'Tempo que atua na área de dados_Não atuo',
       'Tempo que atua na área de dados_de 2 anos a 4 anos',
       'Posição da cadeira (senioridade)_C-Level',
       'Posição da cadeira (senioridade)_Coordenação',
       'Posição da cadeira (senioridade)_Diretoria',
       'Posição da cadeira (senioridade)_Especialista',
       'Posição da cadeira (senioridade)_Gerência',
       'Posição da cadeira (senioridade)_Iniciante',
       'Posição da cadeira (senioridade)_Júnior',
       'Posição da cadeira (senioridade)_Pleno',
       'Posição da cadeira (senioridade)_Sênior', 'Curte games?',
       'Curte futebol?', 'Curte livros?', 'Curte jogos de tabuleiro?',
       'Curte jogos de fórmula 1?', 'Curte jogos de MMA?', 'Idade',])

df = pd.concat([df_template, df], axis=0).fillna(0)
df = df[df_template.columns]

proba = model["model"].predict_proba(df[model['features']])[:,1][0]

if proba > 0.7:
    st.success(f"Você é uma pessoa feliz! Probabilidade: {100 * proba:.0f}%")
elif proba > 0.4:
    st.warning(f"Você é uma pessoa meio feliz! Probabilidade: {100 * proba:.0f}%")
else :
    st.error(f"Você é uma pessoa nada feliz! Probabilidade: {100 * proba:.0f}%")
