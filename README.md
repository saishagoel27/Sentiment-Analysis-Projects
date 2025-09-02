# Sentiment Analysis Projects 🎭

## Project Overview
This repository contains two comprehensive sentiment analysis projects that demonstrate different approaches to text classification using machine learning and deep learning techniques. These projects serve as excellent learning resources for natural language processing and model comparison.

## Projects Included

### 1. Twitter Sentiment Analysis (Classical ML Approach)
**Dataset**: Sentiment140 - 1.6M Twitter tweets
**Approach**: Traditional machine learning with TF-IDF vectorization

### 2. IMDB Movie Reviews Sentiment Analysis (Deep Learning Approach)
**Dataset**: IMDB Dataset - 50K movie reviews
**Approach**: LSTM neural networks with word embeddings

## Key Learning Differences

| Aspect | Twitter Project | IMDB Project |
|--------|----------------|--------------|
| **Algorithm** | Logistic Regression | LSTM Neural Network |
| **Text Processing** | Stemming + TF-IDF | Tokenization + Embeddings |
| **Data Size** | 1.6M tweets | 50K reviews |
| **Preprocessing** | Manual regex + stemming | Keras preprocessing |
| **Feature Extraction** | TF-IDF vectors | Word embeddings |

## Implementation Workflows

### Twitter Sentiment Analysis
```python
# Text preprocessing with stemming
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    return ' '.join(stemmed_content)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
```

### IMDB Movie Reviews Analysis
```python
# Tokenization and padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data["review"])
X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["review"]), maxlen=200)

# LSTM model architecture
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))
```

## Key Learning Concepts

### Classical ML Approach (Twitter)
- **Text Preprocessing**: Regex patterns, stemming, stopword removal
- **Feature Engineering**: TF-IDF vectorization for numerical representation
- **Traditional ML**: Logistic regression for binary classification
- **Model Persistence**: Pickle for model saving and loading

### Deep Learning Approach (IMDB)
- **Modern Preprocessing**: Tokenization and sequence padding
- **Word Embeddings**: Dense vector representations of words
- **Sequential Models**: LSTM for sequence processing
- **Neural Network Training**: Epochs, batch processing, validation splits

## Technical Implementations

### Data Acquisition
Both projects use Kaggle API for dataset download:
```python
os.environ["KAGGLE_USERNAME"] = kaggle_dictionary["username"]
os.environ["KAGGLE_KEY"] = kaggle_dictionary["key"]
!kaggle datasets download -d [dataset-name]
```

### Model Evaluation
- **Twitter Project**: Training accuracy assessment
- **IMDB Project**: Test loss and accuracy with validation splits

### Prediction Examples
Both models provide interactive prediction capabilities for new text inputs.

## Educational Value

### For Beginners
- **Twitter Project**: Learn fundamental NLP preprocessing and classical ML
- **Data handling**: CSV processing, text cleaning, train-test splits

### For Intermediate Learners
- **IMDB Project**: Understand deep learning for NLP
- **Advanced concepts**: Embeddings, LSTM architecture, neural network training

### Comparative Learning
- **Preprocessing Differences**: Manual vs automated text processing
- **Model Complexity**: Linear vs neural network approaches
- **Performance Trade-offs**: Speed vs accuracy considerations

## Requirements
```
# Common dependencies
pandas
numpy
scikit-learn
nltk

# Twitter project specific
pickle

# IMDB project specific
tensorflow
keras
```

## Dataset Sources
- **Twitter**: Sentiment140 dataset (1.6M tweets)
- **IMDB**: Movie reviews dataset (50K reviews)

## Key Takeaways
- **Classical ML**: Fast, interpretable, good for structured problems
- **Deep Learning**: Better for complex patterns, requires more data and computation
- **Text Processing**: Multiple valid approaches depending on use case
- **Model Selection**: Consider data size, complexity, and performance requirements

---

*These projects demonstrate the evolution from traditional machine learning to deep learning approaches in NLP, providing comprehensive learning opportunities for different skill
