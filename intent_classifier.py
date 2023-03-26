import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf


model = tf.keras.models.load_model("models/model_intent_classify")

sample_text = ('Can you tell me more about city of new york?')

def classify(prompt):
    predictions = model.predict(np.array([prompt]))
    print(predictions[0][0])
    if predictions[0][0] < 0:
        return "chitchat"
    else:
        return "topic"