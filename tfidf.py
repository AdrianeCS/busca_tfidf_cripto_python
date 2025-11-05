import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import nltk

try:
    nltk.download('stopwords')
    nltk.download('punkt')
except Exception as e:
    print("Erro ao baixar dados do NLTK:", e)


# 1. Carregar dataset
df = pd.read_csv("cripto_noticias(1).csv")

# 2. Criar matriz TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['texto'])

# 3. Função de busca
def buscar(query, top_n=5):
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    indices = cosine_similarities.argsort()[-top_n:][::-1]
    resultados = df.iloc[indices][['id', 'texto']]
    resultados['similaridade'] = cosine_similarities[indices]
    return resultados

# 4. Loop interativo
if __name__ == "__main__":
    print("🔎 Sistema de busca de notícias sobre Criptomoedas (TF-IDF)\n")
    while True:
        query = input("Digite sua busca (ou 'sair' para encerrar): ").strip()
        if query.lower() == "sair":
            print("Encerrando o programa. Até mais! 🚀")
            break
        resultados = buscar(query)
        print("\n🔍 Resultados mais semelhantes:\n")
        print(resultados.to_string(index=False))
        print("\n" + "-"*60 + "\n")
