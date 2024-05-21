import nltk
nltk.download('punkt')
nltk.download('stopwords')
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
except ImportError:
    st.error("Missing optional dependency 'Sastrawi'. Use pip to install Sastrawi.")

# Load data
def load_data(file):
    if file is not None:
        try:
            filename = file.name.lower()
            if filename.endswith('.csv'):
                st.write("Uploaded file:", file.name)  # Menampilkan nama file yang diunggah
                return pd.read_csv(file)
            elif filename.endswith('.xlsx'):
                st.write("Uploaded file:", file.name)  # Menampilkan nama file yang diunggah
                return pd.read_excel(file)
            else:
                st.warning("Unsupported file format. Please upload a CSV or Excel file.")
                return None
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            return None
    else:
        return None

# Preprocessing
def preprocessing(df):
    st.subheader("Preprocessing:")
    st.text("Menghapus karakter spesial")

    def removeSpecialText(text):
        text = text.replace('\\t',"").replace('\\n',"").replace('\\u',"").replace('\\',"")
        text = text.encode('ascii', 'replace').decode('ascii')
        return text.replace("http://"," ").replace("https://", " ")

    df['title'] = df['title'].astype(str).apply(removeSpecialText)
    st.dataframe(df['title'])

    st.text("Menghapus tanda baca")
    def removePunctuation(text):
        text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text)
        return text

    df['title'] = df['title'].apply(removePunctuation)
    st.dataframe(df['title'])

    st.text("Menghapus angka pada teks")
    def removeNumbers(text):
        return re.sub(r"\d+", "", text)

    df['title'] = df['title'].apply(removeNumbers)
    st.dataframe(df['title'])

    st.text("Mengubah semua huruf pada teks menjadi huruf kecil")
    def casefolding(comment):
        return comment.lower()

    df['title'] = df['title'].apply(casefolding)
    st.dataframe(df['title'])

    st.text("Tokenisasi")
    def tokenize(text):
        tokens = word_tokenize(text)
        return tokens

    df['title_tokens'] = df['title'].apply(tokenize)
    st.dataframe(df[['title', 'title_tokens']])

    st.text("Menghapus stopwords")
    additional_stopwords = ["yg", "dg", "rt", "dgn", "ny", "d", 'klo',
                           'kalo', 'amp', 'biar', 'bikin', 'bilang',
                           'gak', 'ga', 'krn', 'nya', 'nih', 'sih',
                           'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                           'jd', 'jgn', 'sdh', 'aja', 'n', 't',
                           'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                           '&amp', 'yah', 'bisnis', 'pandemi', 'indonesia']

    stopword = set(stopwords.words('indonesian'))
    stopword.update(additional_stopwords)

    def removeStopwords(tokens):
        return [word for word in tokens if word.lower() not in stopword]

    df['title_stop'] = df['title_tokens'].apply(removeStopwords)
    df['title_stop'] = df['title_stop'].apply(lambda tokens: '[' + ', '.join([f"'{token}'" for token in tokens]) + ']')
    st.dataframe(df[['title_tokens', 'title_stop']])

    st.text("Stemming")
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    def stemming(tokens):
        return [stemmer.stem(token) for token in tokens]

    df['title_stemmer'] = df['title_stop'].apply(lambda x: stemming(eval(x)))
    df['title_stemmer'] = df['title_stemmer'].apply(lambda tokens: '[' + ', '.join([f"'{token}'" for token in tokens]) + ']')
    st.dataframe(df[['title_stop', 'title_stemmer']])

    return df['title_stemmer']

# Ekstraksi Fitur TF-IDF
def feature_extraction(data):
    vectorizer = TfidfVectorizer(lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(data)
    return tfidf_matrix, vectorizer.get_feature_names_out()

# Fungsi Ekstraksi Fitur
def ekstraksiFitur(df):
    st.subheader("Ekstraksi Fitur (TF-IDF):")
    tfidf_matrix, feature_names = feature_extraction(df['title'])
    st.text(tfidf_matrix.shape)
    st.dataframe(pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names))
    return tfidf_matrix  # Return the tfidf_matrix for later use

# Fungsi LDA
def lda(data, n_topics, n_words):
    vectorizer = TfidfVectorizer(lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names_out()
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(tfidf_matrix)

    topic_words = []
    # Print top words for each topic with their values
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-n_words - 1:-1]
        top_words = [(feature_names[i], topic[i]) for i in top_words_idx]
        topic_words.append(top_words)

    # Format hasil sesuai keinginan
    topic_result = []
    for i, topic in enumerate(topic_words):
        topic_string = f"Topic {i}: "
        for word, value in topic:
            topic_string += f"{word}: {value:.4f}, "
        topic_result.append(topic_string[:-2])  # Menghapus koma terakhir
    return topic_result

# Fungsi untuk menghitung Dunn Index
def dunn_index(X, labels):
    min_inter_cluster_distance = np.inf
    max_intra_cluster_diameter = -np.inf
    for i in np.unique(labels):
        cluster_points = X[labels == i]
        max_intra_cluster_diameter = max(max_intra_cluster_diameter, np.max(distance.pdist(cluster_points)))
        for j in np.unique(labels):
            if i != j:
                other_cluster_points = X[labels == j]
                min_inter_cluster_distance = min(min_inter_cluster_distance, np.min(distance.cdist(cluster_points, other_cluster_points)))
    dunn = min_inter_cluster_distance / max_intra_cluster_diameter
    return dunn

# Fungsi Clustering 
def clustering(tfidf_matrix):
    st.subheader('Clustering (K-Means)')
    num_of_clusters = st.number_input('Masukkan jumlah cluster: ', min_value=1, max_value=100, value=2, step=1)

    show_cluster_result = st.button('Tampilkan hasil clustering')
    if show_cluster_result:
        # Convert sparse matrix to dense array
        dense_tfidf_matrix = tfidf_matrix.toarray()

        # Menjalankan algoritma k-means
        kmeans = KMeans(n_clusters=num_of_clusters, n_init=10, random_state=0)
        kmeans.fit(dense_tfidf_matrix)

        # Menampilkan hasil clustering
        abstract = []
        cluster = []
        for i, doc in enumerate(st.session_state.df['title']):
            abstract.append(doc)
            cluster.append(kmeans.labels_[i])

        data = {'title': abstract, 'Cluster': cluster}
        df_cluster = pd.DataFrame(data)
        st.dataframe(df_cluster)

# Streamlit App
def main():
    st.title("Topic Modeling")

    selected = st.sidebar.selectbox('Mulai dari Langkah Mana?', ['Pilih Langkah Anda', 'Load Data', 'Preprocessing', 'Feature Extraction', 'LDA', 'Clustering'])

    if selected == 'Pilih Langkah Anda':
        st.image('topic_modeling.png', use_column_width=True)
        st.write("""
        ## Selamat datang di Aplikasi Topic Modeling
        
        Aplikasi ini memungkinkan Anda melakukan topic modeling pada teks yang Anda unggah. Silakan ikuti langkah-langkah di bawah ini:
        
        1. *Load Data*: Unggah file teks Anda (dalam format CSV atau Excel).
        2. *Preprocessing*: Lakukan preprocessing pada teks untuk membersihkan data.
        3. *Feature Extraction*: Ekstrak fitur menggunakan metode TF-IDF.
        4. *LDA*: Terapkan Latent Dirichlet Allocation (LDA) untuk mendapatkan topik-topik dari teks.
        5. *Clustering & Evaluation*: Lakukan clustering (K-Means) dan evaluasi hasil clustering.
        
        Pilih tugas di sidebar untuk memulai!
        """)

    if selected == 'Load Data':
        file = st.file_uploader("Upload file teks", type=['txt', 'csv', 'xlsx'])
        if file:
            df = load_data(file)
            if df is not None:
                st.session_state.df = df
                st.dataframe(df.head())

    elif selected == 'Preprocessing':
        if 'df' not in st.session_state or st.session_state.df is None:
            st.warning("Harap unggah file terlebih dahulu!")
        else:
            processed_data = preprocessing(st.session_state.df)
            st.session_state.processed_data = processed_data
            st.write("Langkah Preprocessing Selesai.")
            st.write("Data yang Diproses:")
            if processed_data is not None:
                st.write(processed_data.head())
            else:
                st.warning("Tidak ada data yang ditampilkan.")

    elif selected == 'Feature Extraction':
        if 'df' not in st.session_state or st.session_state.df is None:
            st.warning("Harap unggah file terlebih dahulu!")
        else:
            st.session_state.tfidf_matrix = ekstraksiFitur(st.session_state.df)
            st.write("Langkah Ekstraksi Fitur Selesai.")

    elif selected == 'LDA':
        if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
            st.warning("Harap lakukan preprocessing terlebih dahulu!")
        else:
            data = st.session_state.processed_data.tolist()  # Assuming 'processed_data' is the column containing text data
            n_topics = st.number_input("Jumlah topik", min_value=2, max_value=10, step=1, value=3)
            n_words = st.number_input("Jumlah kata per topik", min_value=5, max_value=20, step=1, value=10)
            result = lda(data, n_topics, n_words)
            for topic_string in result:
                st.write(topic_string)
            st.write("Langkah LDA Selesai.")

    elif selected == 'Clustering':
        if 'tfidf_matrix' not in st.session_state or st.session_state.tfidf_matrix is None:
            st.warning("Harap lakukan ekstraksi fitur terlebih dahulu!")
        else:
            clustering(st.session_state.tfidf_matrix)

if __name__ == "__main__":
    main()