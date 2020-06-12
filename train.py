import config
import models
import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
#Input training files from benchmarks/FB15K/ folder.
con = config.Config()
# ----Hyperparameters----
MaxIter=1000	# total epochs
lr=0.01	# set learning rate
margin=1.0
lambda1=10.0	# regularisation coefficient of R1 orthogonality constraint
lambda2=10.0	# regularisation coefficient of R2 orthogonality constraint
dim=100
neg_rate=1	#number of negative examples (corrupt head or tail) per positive one
opt_method="SGD"
LossFunc="Marginal" # or Marginal/LogSoftmax
bern=0
neval=1000 #subset size to speed the evaluation on valid set (only for link prediction)
export_steps=20 #Evaluate the performance after n steps
if bern==1:
	negSampMethod="bern"
else:
	negSampMethod="unif"
print "Hyperparameters:"
print "%s_lr%.3f_d=%d_epochs=%d_l1=%d_l2=%d_m=%d_NegRate=%d_%s_%s"%(opt_method,lr,dim,MaxIter,lambda1,lambda2,margin,neg_rate,negSampMethod,LossFunc)
# ----------------------
#True: Input test files from the same folder.
KG="WN11"
con.set_KG_name(KG)
con.set_in_path("./benchmarks/%s/"%KG)
con.set_test_link_prediction(False)
con.set_test_triple_classification(False)
con.set_work_threads(8)
con.set_train_times(MaxIter) 
con.set_nbatches(100)
con.set_alpha(lr) 
con.set_margin(margin)
con.set_lambda1(lambda1) 
con.set_lambda2(lambda2) 
con.set_bern(bern)
con.set_dimension(dim)
con.set_ent_neg_rate(neg_rate) 
con.set_rel_neg_rate(0) # set of corrupting relations to generate negative examples. 
con.set_opt_method(opt_method)
con.set_loss_function(LossFunc)

#Models will be exported via tf.Saver() automatically.
con.set_export_files("./res/model.vec.tf", export_steps,neval)
#Model parameters will be exported to json files automatically.
con.set_out_files("./res/embedding.vec.json")
#Initialize experimental settings.
con.init()
#Set the knowledge embedding model
con.set_model(models.RelWalk)
#Train the model.
con.run()
#To test models after training needs "set_test_flag(True)".
# con.test()
# con.show_link_prediction(2,1)
# con.show_triple_classification(2,1,3)

#Get the embeddings (numpy.array)
embeddings = con.get_parameters("numpy")
np.save("./embeddings/RelWalk_Embeddings/%s/Last_embeddings_%s_lr%.3f_d=%d_epochs=%d_l1=%d_l2=%d_m=%d_NegRate=%d_%s_%s.npy"%(KG,opt_method,lr,dim,MaxIter,lambda1,lambda2,margin,neg_rate,negSampMethod,LossFunc),embeddings)

