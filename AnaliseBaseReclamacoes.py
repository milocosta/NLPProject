# %% Importação das bibliotecas
import nltk
import pandas as pd
import os
import numpy as np
import string
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
# %% Importacao da Base de Dados
df = pd.read_csv('BaseReclamacoes.csv',encoding='windows-1252',sep =';')
df.head()
# %% Quantitativo de linhas e colunas do Dataframe
df.shape
# %% Informativo do dataframe
df.info()
# %% Verificaação de Missing
df.isnull().sum()
# %% Determinação dos Dataframes
df_empresa = pd.DataFrame(df['empresa'])
df_estado   = pd.DataFrame(df['estado'])
df_problema = pd.DataFrame(df['problema'])
# %% Tratamento de Missing dos novos dataframes
print(df_empresa.isnull().sum())
print(df_estado.isnull().sum())
print(df_problema.isnull().sum())
# %%
df_empresa.dropna(inplace=True)
df_estado.dropna(inplace=True)
df_problema.dropna(inplace=True)
# %% Nova verificação de null
print(df_empresa.isnull().sum())
print(df_estado.isnull().sum())
print(df_problema.isnull().sum())
# %% Para funcionamento do Wordcloud
dicionario = {' ':'_','/':'',',':'','\(':'','\)':'',"-":'','\.':'',r"^\t":''}
df_empresa['NomeEmpresaSemEspaco'] = df_empresa.replace({'empresa': dicionario}, regex=True)
# %% Agrupando e Ordenando as empresas com maior quantidade de chamados
print(df_empresa.groupby('NomeEmpresaSemEspaco').size().sort_values(ascending=False))
# %% Criação de uma string única com as empresas
s_empresas = " ".join([text for text in df_empresa['NomeEmpresaSemEspaco']])
# %% Criação do Wordcloud
wordcloud_empresas = WordCloud(background_color='black', collocations=False, colormap='GnBu', width=3000, height=2000,max_font_size=1000, max_words=40).generate(s_empresas)
plt.figure(figsize=(40,30), facecolor='k', edgecolor='k')
plt.imshow(wordcloud_empresas, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('WordcloudEmpresas.png')
plt.show()
# %% Agrupando e Ordenando os Estados com maior quantidade de chamados
print(df_estado.groupby('estado').size().sort_values(ascending=False))
# %% Criação de uma string única com os estados
s_estados = " ".join([text for text in df_estado['estado']])
# %% Criação do Wordcloud dos Estados
wordcloud_estados = WordCloud(background_color='black', collocations=False, colormap='GnBu', width=3000, height=2000,max_font_size=1000, max_words=17).generate(s_estados)
plt.figure(figsize=(40,30), facecolor='k', edgecolor='k')
plt.imshow(wordcloud_estados, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('WordcloudEstados.png')
plt.show()
# %% # %% Para funcionamento do Wordcloud dos Problemas
df_problema['ProblemaSemEspaco'] = df_problema.replace({'problema': dicionario}, regex=True)
# %% Agrupando e Ordenando os problemas pela quantidade
print(df_problema.groupby('problema').size().sort_values(ascending=False))
# %% Criação de uma string única com os problemas
s_problemas = " ".join([text for text in df_problema['ProblemaSemEspaco']])
# %% Criação do Wordcloud dos Problemas
wordcloud_problemas = WordCloud(background_color='black', collocations=False, colormap='GnBu', width=3000, height=2000,max_font_size=1000, max_words=40).generate(s_problemas)
plt.figure(figsize=(40,30), facecolor='k', edgecolor='k')
plt.imshow(wordcloud_problemas, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('WordcloudProblemas.png')
plt.show()