import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  # Import classification_report
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from snowballstemmer import TurkishStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Function for text preprocessing
def preprocess_text_turkish(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text, language='turkish')  # Tokenize in Turkish
    stop_words = set(stopwords.words('turkish'))  # Turkish stopwords
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    stemmer = TurkishStemmer()  # Turkish stemmer
    tokens = [stemmer.stemWord(word) for word in tokens]  # Stemming
    return ' '.join(tokens)

# Load the dataset (adjust the path as needed)
data_path = 'Data.xlsx'  # Replace with your file path
data = pd.read_excel(data_path, sheet_name='Data')

# Preprocess the dataset
data['Clean_Comment_Turkish'] = data['Comment'].apply(preprocess_text_turkish)

# Assuming 'data' is your DataFrame with raw text and labels
X_raw = data['Comment']  # Replace with your column name
y_raw = data['Topic']  # Replace with your column name

# Tokenize the text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_raw)
X_seq = tokenizer.texts_to_sequences(X_raw)
X_padded = pad_sequences(X_seq, padding='post', maxlen=50)

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
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

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Neural Network Accuracy: {accuracy:.2f}")

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Convert one-hot encoded predictions to class labels
y_pred_labels = [label_encoder.inverse_transform([tf.argmax(pred)]) for pred in y_pred]
y_test_labels = [label_encoder.inverse_transform([tf.argmax(true)]) for true in y_test]

# Generate a classification report
classification_rep = classification_report(y_test_labels, y_pred_labels)
print("Classification Report:\n", classification_rep)
