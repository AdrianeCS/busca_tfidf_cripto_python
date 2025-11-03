# 📚 Sistema TF-IDF

## 1\. Visão Geral

Este documento descreve o script Python `tfidf.py`, que implementa o algoritmo **TF-IDF** (Frequência de Termo-Inverso da Frequência de Documento) para quantificar a importância de palavras em um conjunto de documentos de texto (notícias sobre criptomoedas).

O sistema processa um arquivo CSV, limpa o texto utilizando *stop words* em português e, finalmente, calcula a matriz TF-IDF.

## 2\. Pré-requisitos

Para executar o script com sucesso, os seguintes pacotes Python devem estar instalados.

| Pacote | Versão Mínima | Comando de Instalação |
| :--- | :--- | :--- |
| `pandas` | 1.0+ | `pip install pandas` |
| `scikit-learn` | 1.0+ | `pip install scikit-learn` |
| `nltk` | 3.0+ | `pip install nltk` |

### 🚨 Configuração do NLTK

A biblioteca `nltk` requer o download dos dados de *stop words* em português. Isso deve ser feito uma vez antes da execução do script:

1.  Abra o terminal/PowerShell.
2.  Execute o interpretador Python: `python`
3.  Execute os comandos de download:
    ```python
    >>> import nltk
    >>> nltk.download('stopwords')
    >>> exit()
    ```

## 3\. Estrutura do Arquivo

### 📂 Arquivos Necessários

O sistema requer que o arquivo de dados de entrada (`cripto_noticias.csv`) esteja no **mesmo diretório** do script `tfidf.py`.

  * `tfidf.py`: O script principal que contém a lógica do TF-IDF.
  * `cripto_noticias.csv`: O arquivo de dados de entrada. Deve conter uma coluna com o texto a ser analisado (provavelmente chamada **`'texto'`**).

## 4\. Uso (Execução)

Para executar o sistema, navegue até o diretório do projeto no terminal e execute o script usando o interpretador Python:

```bash
cd C:\Users\jp8pe\OneDrive\Documentos\Busca_Python
py tfidf.py
```

## 5\. Componentes Principais do Código

O script `tfidf.py` deve incluir as seguintes etapas lógicas:

### 5.1. Importação de Bibliotecas

```python
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
```

### 5.2. Carregamento e Preparação dos Dados

Esta seção carrega o arquivo CSV em um `DataFrame` do Pandas. É crucial que o nome do arquivo esteja correto e que ele esteja no diretório correto.

```python
# Carrega o arquivo CSV
df = pd.read_csv("cripto_noticias.csv")

# Assume-se que a coluna de texto seja 'texto'
documentos = df['texto'] 
```

### 5.3. Configuração do Stop Words

Para garantir a remoção de palavras comuns em português (como 'de', 'a', 'o'), a lista de *stop words* do NLTK é carregada e passada para o vetorizador.

```python
# Carrega a lista de stop words em português
portuguese_stopwords = stopwords.words('portuguese')
```

### 5.4. Cálculo do TF-IDF

O objeto `TfidfVectorizer` é instanciado e treinado (`fit_transform`) com o conjunto de documentos, resultando na matriz TF-IDF.

```python
# Instancia o TfidfVectorizer com as stop words em português
vectorizer = TfidfVectorizer(stop_words=portuguese_stopwords)

# Calcula a matriz TF-IDF
# Esta linha (ou similar) é a linha 10 no seu traceback
tfidf_matrix = vectorizer.fit_transform(documentos)
```

## 6\. Saída do Sistema

Após a execução, a variável **`tfidf_matrix`** conterá a representação esparsa dos seus documentos, onde:

  * **Linhas:** Correspondem a cada documento (notícia).
  * **Colunas:** Correspondem a cada termo único (palavra) encontrado no conjunto de documentos.
  * **Valores:** São as pontuações TF-IDF, que indicam a relevância de cada termo para cada documento.

A matriz `tfidf_matrix` e o `vectorizer` podem ser usados posteriormente para:

  * Visualização de dados (Nuvem de Palavras, etc.)
  * Busca Semântica (Cálculo de similaridade entre documentos).
  * Clusterização ou Classificação de textos.

Ficou faltando alguma parte específica do seu código que você gostaria de incluir ou detalhar na documentação?
