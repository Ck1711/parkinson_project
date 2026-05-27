import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

# Create and save a Sequential model
model = Sequential()
model.add(Dense(128, input_dim=10, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.save('test_model.h5')

# Load the model
loaded_model = load_model('test_model.h5')

try:
    print("Trying .input:", loaded_model.input)
except Exception as e:
    print("Error on .input:", type(e).__name__, e)

try:
    print("Trying .inputs:", loaded_model.inputs)
except Exception as e:
    print("Error on .inputs:", type(e).__name__, e)

try:
    extractor = tf.keras.models.Sequential(loaded_model.layers[:-1])
    out = extractor(loaded_model.inputs[0])
    print("Sequential slicing success. Output shape:", out.shape)
except Exception as e:
    print("Sequential slicing failed:", e)

# What about Model(inputs, outputs)?
try:
    extractor2 = tf.keras.models.Model(inputs=loaded_model.inputs, outputs=loaded_model.layers[-2].output)
    print("Model slicing success")
except Exception as e:
    print("Model slicing failed:", e)
