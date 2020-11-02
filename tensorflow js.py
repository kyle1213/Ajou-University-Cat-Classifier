import tensorflowjs as tfjs
import tensorflow as tf
import tensorflow_hub as hub

model = tf.keras.models.load_model('resnet.h5', custom_objects={'KerasLayer':hub.KerasLayer})

model.summary()

tfjs.converters.save_keras_model(model, 'tfjs')
