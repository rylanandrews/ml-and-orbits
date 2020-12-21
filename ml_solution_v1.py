import tensorflow as tf

savedModelPath = "C:\\Users\\Rylan\\Documents\\Schoolwork\\12th Grade\\Science Research 2021\\Programs\\threeBodyModelV1.pb"

loadedModel = tf.saved_model.load(savedModelPath)