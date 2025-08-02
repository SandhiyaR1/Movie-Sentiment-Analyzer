# 🎬 Movie-Sentiment-Analyzer
IMDb Sentiment Analysis - This project is a simple yet effective Sentiment Analysis App built to classify - IMDb movie reviews as Positive 🙂 or Negative 😠.

## 📚 Dataset

- **Dataset**: IMDB Movie Reviews Dataset (50,000 labeled movie reviews)
- **Source**: Preloaded in Keras via `keras.datasets.imdb`
- **Split**: 25,000 for training and 25,000 for testing

## 🧠 Model Used

- **Model Type**: Long Short-Term Memory (LSTM)
- **Framework**: Keras with TensorFlow backend

## 🏗️ Model Architecture

```python
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=200))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
```
## Explanation:
Embedding Layer: Converts input word indices into dense vector embeddings of size 128.

LSTM Layer: Contains 128 LSTM units, capable of learning temporal dependencies in text data. Includes dropout and recurrent dropout of 20% to prevent overfitting.

Dense Layer: A single neuron with sigmoid activation to output probability (sentiment prediction).
### 📈 Model Performance
Loss Function: Binary Crossentropy

Optimizer: Adam

Metrics: Accuracy

## ✅ Final Results:
Test Accuracy: 87.79%

Test Loss: 0.3744

## Sample Output
<img width="958" height="780" alt="image" src="https://github.com/user-attachments/assets/28a33e8f-573f-420f-a7a0-6d54dc4628c8" />
<img width="972" height="752" alt="image" src="https://github.com/user-attachments/assets/80ea5f2b-fd90-41da-aa2e-fd3db004389d" />

## 🔚 Conclusion
This project demonstrates a simple yet effective sentiment analysis system using an LSTM-based neural network built with Keras. The model was trained on a pre-tokenized IMDb movie reviews dataset to classify text as either positive or negative. With a validation accuracy of ~87.79% and a loss of 0.3744, the model shows good performance on unseen data. The architecture leverages an Embedding layer followed by an LSTM layer and a sigmoid-activated Dense output, allowing it to learn temporal patterns in text. This setup can be further enhanced with techniques like bidirectional LSTM, attention mechanisms, or pretrained word embeddings in future iterations.



