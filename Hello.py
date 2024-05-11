import nltk
nltk.download('punkt')
nltk.download('stopwords')
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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

    st.text("Tokenisasi dan penghapusan stopwords")
    stopword = set(stopwords.words('indonesian'))

    def stopwordRemoval(text):
        words = word_tokenize(text)
        result = [word for word in words if word.lower() not in stopword]
        return ' '.join(result)

    df['title'] = df['title'].apply(stopwordRemoval)
    st.dataframe(df['title'])

    st.text("Stemming")
    factory = StopWordRemoverFactory()
    stemmer = factory.create_stop_word_remover()
    def stemming(text):
        return stemmer.remove(text)

    df['title'] = df['title'].apply(stemming)
    st.dataframe(df['title'])

    # Mengembalikan hasil preprocessing, tfidf_matrix, dan vectorizer
    return df['title'], None, None  

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

    # Print top words for each topic
    for topic_idx, topic in enumerate(lda.components_):
        st.write(f"Top words for Topic {topic_idx+1}:")
        top_words_idx = topic.argsort()[:-n_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        st.write(top_words)

    return lda

def clustering_and_classification(tfidf_matrix):
    from sklearn.cluster import KMeans

    st.subheader('Clustering (K-Means)')
    num_of_clusters = st.number_input('Masukkan jumlah cluster: ', min_value=1, max_value=100, value=2, step=1)

    show_cluster_result = st.button('Tampilkan hasil clustering')
    if show_cluster_result:
        # Menjalankan algoritma k-means
        kmeans = KMeans(n_clusters=num_of_clusters, n_init=10, random_state=0)
        kmeans.fit(tfidf_matrix)

        # Menampilkan hasil clustering
        abstract = []
        cluster = []
        for i, doc in enumerate(st.session_state.df['title']):
            abstract.append(doc)
            cluster.append(kmeans.labels_[i])

        data = {'title': abstract, 'Cluster': cluster}
        df_cluster = pd.DataFrame(data)
        st.dataframe(df_cluster)

        # Evaluasi menggunakan Silhouette Score
        from sklearn.metrics import silhouette_score
        preds = kmeans.predict(tfidf_matrix)
        silhouette_avg = silhouette_score(tfidf_matrix, preds)
        st.text("Silhouette Score: {:.2f}".format(silhouette_avg))

        # Klasifikasi
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.metrics import classification_report

        # Membagi data menjadi data training dan data testing
        X_train, X_test, y_train, y_test = train_test_split(df_cluster['title'], df_cluster['Cluster'], test_size=0.2, random_state=42)

        # Mengubah teks menjadi vektor fitur menggunakan CountVectorizer
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Membuat dan melatih model Naive Bayes
        nb = MultinomialNB()
        nb.fit(X_train_vec, y_train)

        # Melakukan prediksi pada data testing
        y_pred = nb.predict(X_test_vec)

        # Menampilkan laporan klasifikasi
        st.subheader('Klasifikasi (Naive Bayes)')
        report = classification_report(y_test, y_pred)
        st.text(report)

        # Visualisasi Confusion Matrix
        st.subheader('Confusion Matrix')
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        st.pyplot(plt)


# Streamlit App
def main():
    if 'show_preprocessing' not in st.session_state:
        st.session_state.show_preprocessing = False

    st.title("Topic Modeling")

    selected = st.sidebar.selectbox('Mulai dari Langkah Mana?', ['Pilih Langkah Anda', 'Load Data', 'Preprocessing', 'Feature Extraction', 'LDA', 'Clustering & Classification'], key='select_step')

    if selected == 'Pilih Langkah Anda':
        st.image('topic_modeling.png', use_column_width=True)

        st.write("""
        ## Selamat datang di Aplikasi Topic Modeling
        
        Aplikasi ini memungkinkan Anda melakukan topic modeling pada teks yang Anda unggah. Silakan ikuti langkah-langkah di bawah ini:
        
        1. **Load Data**: Unggah file teks Anda (dalam format CSV atau Excel).
        2. **Preprocessing**: Lakukan preprocessing pada teks untuk membersihkan data.
        3. **Feature Extraction**: Ekstrak fitur menggunakan metode TF-IDF.
        4. **LDA**: Terapkan Latent Dirichlet Allocation (LDA) untuk mendapatkan topik-topik dari teks.
        5. **Clustering & Classification**: Lakukan clustering (K-Means) dan klasifikasi (Naive Bayes) pada data.
        
        Pilih tugas di sidebar untuk memulai!
        """)

    if selected == 'Load Data':
        file = st.file_uploader("Upload file teks", type=['txt', 'csv', 'xlsx'])
        if file:
            df = load_data(file)
            if df is not None:
                st.session_state.df = df
                st.dataframe(df.head())
                if st.session_state.show_preprocessing:
                    preprocessing(df)

    elif selected == 'Preprocessing':
        if 'df' not in st.session_state or st.session_state.df is None:
            st.warning("Harap unggah file terlebih dahulu!")
        else:
            st.session_state.show_preprocessing = True
            processed_data, tfidf_matrix, vectorizer = preprocessing(st.session_state.df)
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
        if 'df' not in st.session_state or st.session_state.df is None:
            st.warning("Harap unggah file terlebih dahulu!")
        else:
            data = st.session_state.df['title'].tolist()  # Assuming 'title' is the column containing text data
            n_topics = st.number_input("Jumlah topik", min_value=2, max_value=10, step=1, value=3)
            n_words = st.number_input("Jumlah kata per topik", min_value=5, max_value=20, step=1, value=10)
            lda_model = lda(data, n_topics, n_words)
            st.write("Langkah LDA Selesai.")

    elif selected == 'Clustering & Classification':
        if 'tfidf_matrix' not in st.session_state or st.session_state.tfidf_matrix is None:
            st.warning("Harap lakukan ekstraksi fitur terlebih dahulu!")
        else:
            clustering_and_classification(st.session_state.tfidf_matrix)

if __name__ == "__main__":
    main()