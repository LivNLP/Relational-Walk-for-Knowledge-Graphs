import config
import models
import tensorflow as tf
import numpy as np
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
# (1) Set import files and OpenKE will automatically load models via tf.Saver().
con = config.Config()
# -------------------------------
KG="FB15K237"
embedding_file="KGEs_d100.npy"
print "KG:",KG
print "embeddings:",embedding_file
# -------------------------------
con.set_in_path("./benchmarks/%s/"%KG)
con.set_test_link_prediction(True)
con.set_test_triple_classification(True)
con.set_work_threads(8)
con.set_dimension(100)
# con.set_import_files("./res/model.vec.tf")
con.init()
con.set_model(models.RelWalk)
content = np.load("./Pre-trained-Embeddings/%s/%s"%(KG,embedding_file)).item()
for key in content:
	print key, content[key].shape
con.set_parameters(content)
con.test()

con.show_link_prediction(2,1)
con.show_triple_classification(2,1,3)
print "KG:",KG
print "embeddings:",embedding_file
