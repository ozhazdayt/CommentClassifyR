import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from TurkishStemmer import TurkishStemmer
import string
import os
import contextlib
from openpyxl import load_workbook
# Türkçe için metin ön işleme adımları
def preprocess_text_turkish(text):
    text = text.lower()  # Küçük harfe dönüştürme
    text = text.translate(str.maketrans('', '', string.punctuation))  # Noktalama işaretlerini kaldırma
    tokens = word_tokenize(text, language='turkish')  # Tokenleme için Türkçe
    stop_words = set(stopwords.words('turkish'))  # Türkçe stop-word'leri alma
    tokens = [word for word in tokens if word not in stop_words]  # Stop-word'leri kaldırma
    stemmer = TurkishStemmer()
    tokens = [stemmer.stem(word) for word in tokens]  # Use stem() method instead of stemWord()
    return ' '.join(tokens)

# Load the dataset
file_path = 'Dataset.xlsx'
data = pd.read_excel(file_path, sheet_name='Data')

# Preprocess the dataset
data['Clean_Comment_Turkish'] = data['Comment'].apply(preprocess_text_turkish)

# Tokenize the text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['Clean_Comment_Turkish'])
X_seq = tokenizer.texts_to_sequences(data['Clean_Comment_Turkish'])
X_padded = pad_sequences(X_seq, padding='post', maxlen=50)

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['Topic'])
y_categorical = to_categorical(y_encoded)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_categorical, test_size=0.2, random_state=42)

# Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=64, input_length=50),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(y_categorical.shape[1], activation='softmax')  # Use 'sigmoid' for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Use 'binary_crossentropy' for binary

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
# Input ve Output path'lerini belirleme
input_path = 'C:\\Users\\SELEN\\Desktop\\Development\\Data\\Input\\amazon_reviews.xlsx'
output_path = 'C:\\Users\\SELEN\\Desktop\\Development\\Data\\Output\\amazon_reviews_categorized.xlsx'

# Input Excel dosyasını okuma (başlık yoksa)
data = pd.read_excel(input_path, header=None)

# Yorumların bulunduğu sütunun indeksi (örneğin, ilk sütun için 0)
comment_column_index = 0

# Yorumları modelle etiketleme
def label_comment(comment):
    new_comment_clean = preprocess_text_turkish(comment)
    new_comment_seq = tokenizer.texts_to_sequences([new_comment_clean])
    new_comment_padded = pad_sequences(new_comment_seq, padding='post', maxlen=50)
    probabilities = model.predict(new_comment_padded)[0]

    # Olasılığı %35'un üstünde olan kategorileri bulma
    threshold = 0.35
    likely_categories = [label_encoder.classes_[i] for i, prob in enumerate(probabilities) if prob > threshold]

    # Kategorilere ait olasılıkları alarak yüzdeye çevirme
    category_probabilities = [f"{prob * 100:.2f}%" for i, prob in enumerate(probabilities) if label_encoder.classes_[i] in likely_categories]

    return likely_categories, category_probabilities

# Her yorum için etiket ekleme
data['Predicted_Topics'], data['Probabilities'] = zip(*data[comment_column_index].apply(label_comment))

# Excel'e yazdırma işlemi
with open(os.devnull, 'w') as nullfile:
    with contextlib.redirect_stdout(nullfile):
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
        for label in data['Predicted_Topics'].explode().unique():
            if isinstance(label, str):  # Check if label is already a string
                cleaned_label = label.replace("/", "_")  # Replace "/" with "_"
            else:
                cleaned_label = str(label)  # Convert non-string types to strings and then replace
                cleaned_label = cleaned_label.replace("/", "_")  # Replace "/" with "_"

            labeled_data = data[data['Predicted_Topics'].apply(lambda x: label in x)]
            labeled_data.to_excel(writer, sheet_name=cleaned_label, index=False, columns=[comment_column_index])

        # 'All' kısmında Comments ve Predicted Topics kolonları olsun
        all_data = data.explode('Predicted_Topics')[[comment_column_index, 'Predicted_Topics', 'Probabilities']]
        all_data.to_excel(writer, sheet_name='All', index=False, header=['Comments', 'Predicted Topics', 'Probabilities'])

        # Excel dosyasını kaydetme ve kapatma
        writer.save()
