import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
os.chdir("/home/monijesu/Documents/Project/output/trafficsignnet.model")
pb_saved_model = "/home/monijesu/Documents/Project/output/trafficsignnet.model/saved_model.pb"

_graph = tf.Graph()
with _graph.as_default():
    _sess = tf.Session(graph=_graph)
    model = tf.saved_model.loader.load(_sess, ["serve"], pb_saved_model)

with tf.gfile.GFile("/home/monijesu/Documents/Project/frozen/frozen.pb", "wb") as f:
    f.write(model.SerializeToString())