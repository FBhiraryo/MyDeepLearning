import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    print("GPU is available. TensorFlow will use it.")
    print(tf.config.list_physical_devices('GPU'))
else:
    print("No GPU devices found.")