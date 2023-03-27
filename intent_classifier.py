import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn import preprocessing
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from nlp_pipeline import get_nel

tfds.disable_progress_bar()
model = tf.keras.models.load_model("models/model_intent_classify")

def load_data(filename):
    # Load the CSV file into a Pandas DataFrame
    data = pd.read_csv(filename)

    label = preprocessing.LabelEncoder()
    y = label.fit_transform(data['intent'])
    data['label'] = y

    # Separate classes
    class_count_0, class_count_1 = data['label'].value_counts()
    class_0 = data[data['label'] == 0]
    class_1 = data[data['label'] == 1]
    class_0_under = class_0.sample(class_count_1)
    data = pd.concat([class_0_under, class_1], axis=0)

    print("Total number of examples:", data.shape[0])
    print("Class counts:", data['label'].value_counts())

    return data

def preprocess_data(data, vocab_size=1000, max_length=200):
    df = data.copy()

    # Split the DataFrame into training and testing sets
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    # Create a TensorFlow Dataset from the training and testing sets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_df['parent'].values, train_df['label'].values))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_df['parent'].values, test_df['label'].values))

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    # Shuffle and batch the training dataset
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Batch the testing dataset
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Create a TextVectorization layer to encode text
    encoder = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_sequence_length=max_length
    )
    encoder.adapt(train_dataset.map(lambda text, label: text))

    return train_dataset, test_dataset, encoder

def build_model(encoder, vocab_size, embedding_dim=64, lstm_units=64):
    # Build the model architecture
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])
    model.summary()
    return model

def train_model(model, train_dataset, test_dataset, epochs=5):
    # Compile the model
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        validation_steps=30
    )
    model.save('/models/model_intent_classifier')
    return history, model
    
def predict_text(model, encoder, text):
    # Make prediction on input text
    prediction = model.predict(np.array([text]))
    proba = tf.sigmoid(prediction).numpy()[0][0]
    label = 'chitchat' if proba < 0.5 else 'reddit'
    return {'intent': label, 'confidence': round(proba, 3)}

def plot_history(history):
    # Plot the training and validation accuracy and loss over time
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

def get_model():
    global model
    if model == None:
        model = tf.keras.models.load_model("models/model_intent_classify")
    return model
   
def classify(prompt, topics):
    model = get_model()
    predictions = model.predict(np.array([prompt]))
    print(predictions[0][0])
    topic = False
    if topics != 0:
        topic = True
    if predictions[0][0] < 0 and topic==0:
        return "chitchat"
    else:
        return "topic"