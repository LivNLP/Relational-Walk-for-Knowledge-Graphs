#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes
from ctypes.util import find_library
import json
import random

class Config(object):
	r'''
	use ctypes to call C functions from python and set essential parameters.
	'''
	def __init__(self):
		base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../release/Base.so'))
		print ctypes.cdll.LoadLibrary(find_library('c'))
		self.lib = ctypes.cdll.LoadLibrary(base_file)
		self.lib.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
		self.lib.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.testHead.argtypes = [ctypes.c_void_p]
		self.lib.testTail.argtypes = [ctypes.c_void_p]
		self.lib.getTestBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getValidBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getBestThreshold.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.test_triple_classification.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.test_flag = False
		self.in_path = None
		self.out_path = None
		self.bern = 0
		self.hidden_size = 100
		self.ent_size = self.hidden_size
		self.rel_size = self.hidden_size
		self.train_times = 0
		self.margin = 1.0
		self.lambda1=1.0
		self.lambda2=1.0
		self.LossFunc='Marginal'
		self.nbatches = 100
		self.negative_ent = 1
		self.negative_rel = 0
		self.workThreads = 1
		self.alpha = 0.001
		self.lmbda = 0.000
		self.log_on = 1
		self.exportName = None
		self.importName = None
		self.export_steps = 0
		self.opt_method = "SGD"
		self.optimizer = None
		self.test_link_prediction = False
		self.test_triple_classification = False

	def init_link_prediction(self):
		self.lib.importTestFiles()
		self.lib.importTypeFiles()
		self.test_h = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
		self.test_t = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
		self.test_r = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
		self.test_h_addr = self.test_h.__array_interface__['data'][0]
		self.test_t_addr = self.test_t.__array_interface__['data'][0]
		self.test_r_addr = self.test_r.__array_interface__['data'][0]

	def init_triple_classification(self):
		self.lib.importTestFiles()
		self.lib.importTypeFiles()

		self.test_pos_h = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_pos_t = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_pos_r = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_neg_h = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_neg_t = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_neg_r = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_pos_h_addr = self.test_pos_h.__array_interface__['data'][0]
		self.test_pos_t_addr = self.test_pos_t.__array_interface__['data'][0]
		self.test_pos_r_addr = self.test_pos_r.__array_interface__['data'][0]
		self.test_neg_h_addr = self.test_neg_h.__array_interface__['data'][0]
		self.test_neg_t_addr = self.test_neg_t.__array_interface__['data'][0]
		self.test_neg_r_addr = self.test_neg_r.__array_interface__['data'][0]

		self.valid_pos_h = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_pos_t = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_pos_r = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_neg_h = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_neg_t = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_neg_r = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_pos_h_addr = self.valid_pos_h.__array_interface__['data'][0]
		self.valid_pos_t_addr = self.valid_pos_t.__array_interface__['data'][0]
		self.valid_pos_r_addr = self.valid_pos_r.__array_interface__['data'][0]
		self.valid_neg_h_addr = self.valid_neg_h.__array_interface__['data'][0]
		self.valid_neg_t_addr = self.valid_neg_t.__array_interface__['data'][0]
		self.valid_neg_r_addr = self.valid_neg_r.__array_interface__['data'][0]	
		self.relThresh = np.zeros(self.lib.getRelationTotal(), dtype = np.float32)
		self.relThresh_addr = self.relThresh.__array_interface__['data'][0]

	def init(self):
		self.trainModel = None
		if self.in_path != None:
			self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
			self.lib.setBern(self.bern)
			self.lib.setWorkThreads(self.workThreads)
			self.lib.randReset()
			self.lib.importTrainFiles()
			self.relTotal = self.lib.getRelationTotal()
			self.entTotal = self.lib.getEntityTotal()
			self.trainTotal = self.lib.getTrainTotal()
			self.testTotal = self.lib.getTestTotal()
			self.validTotal = self.lib.getValidTotal()
			self.batch_size = int(self.lib.getTrainTotal() / self.nbatches)
			self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)
			self.batch_h = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_t = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_r = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_y = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.float32)
			self.batch_h_addr = self.batch_h.__array_interface__['data'][0]
			self.batch_t_addr = self.batch_t.__array_interface__['data'][0]
			self.batch_r_addr = self.batch_r.__array_interface__['data'][0]
			self.batch_y_addr = self.batch_y.__array_interface__['data'][0]
		if self.test_link_prediction:
			self.init_link_prediction()
		if self.test_triple_classification:
			self.init_triple_classification()

	def get_ent_total(self):
		return self.entTotal

	def get_rel_total(self):
		return self.relTotal

	def set_lmbda(self, lmbda):
		self.lmbda = lmbda

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

	def set_opt_method(self, method):
		self.opt_method = method

	def set_loss_function(self,LossFunc):
		self.LossFunc=LossFunc

	def set_test_link_prediction(self, flag):
		self.test_link_prediction = flag

	def set_test_triple_classification(self, flag):
		self.test_triple_classification = flag

	def set_log_on(self, flag):
		self.log_on = flag

	def set_alpha(self, alpha):
		self.alpha = alpha
		print "Learning_rate=",self.alpha
	
	def set_KG_name(self,KG):
		self.KG=KG

	def set_in_path(self, path):
		self.in_path = path

	def set_out_files(self, path):
		self.out_path = path

	def set_bern(self, bern):
		self.bern = bern
		if self.bern==1:
			self.negSampMethod="bern"
		else:
			self.negSampMethod="unif"

	def set_dimension(self, dim):
		self.hidden_size = dim
		self.ent_size = dim
		self.rel_size = dim

	def set_ent_dimension(self, dim):
		self.ent_size = dim

	def set_rel_dimension(self, dim):
		self.rel_size = dim

	def set_train_times(self, times):
		self.train_times = times
		print "Number of epochs=",self.train_times

	def set_nbatches(self, nbatches):
		self.nbatches = nbatches

	def set_margin(self, margin):
		self.margin = margin
		print "margin=",self.margin

	def set_lambda1(self,lambda1):
		self.lambda1=lambda1
		print "lambda1=",self.lambda1

	def set_lambda2(self,lambda2):
		self.lambda2=lambda2
		print "lambda2=",self.lambda2

	def set_work_threads(self, threads):
		self.workThreads = threads

	def set_ent_neg_rate(self, rate):
		self.negative_ent = rate

	def set_rel_neg_rate(self, rate):
		self.negative_rel = rate

	def set_import_files(self, path):
		self.importName = path

	def set_export_files(self, path, steps,neval):
		self.exportName = path
		self.export_steps = steps
		self.neval=neval

	def set_export_steps(self, steps):
		self.export_steps = steps

	def sampling(self):
		self.lib.sampling(self.batch_h_addr, self.batch_t_addr, self.batch_r_addr, self.batch_y_addr, self.batch_size, self.negative_ent, self.negative_rel)

	def save_tensorflow(self):
		with self.graph.as_default():
			with self.sess.as_default():
				self.saver.save(self.sess, self.exportName)

	def restore_tensorflow(self):
		with self.graph.as_default():
			with self.sess.as_default():
				self.saver.restore(self.sess, self.importName)


	def export_variables(self, path = None):
		with self.graph.as_default():
			with self.sess.as_default():
				if path == None:
					self.saver.save(self.sess, self.exportName)
				else:
					self.saver.save(self.sess, path)

	def import_variables(self, path = None):
		with self.graph.as_default():
			with self.sess.as_default():
				if path == None:
					self.saver.restore(self.sess, self.importName)
				else:
					self.saver.restore(self.sess, path)

	def get_parameter_lists(self):
		return self.trainModel.parameter_lists

	def get_parameters_by_name(self, var_name):
		with self.graph.as_default():
			with self.sess.as_default():
				if var_name in self.trainModel.parameter_lists:
					return self.sess.run(self.trainModel.parameter_lists[var_name])
				else:
					return None

	def get_parameters(self, mode = "numpy"):
		res = {}
		lists = self.get_parameter_lists()
		for var_name in lists:
			if mode == "numpy":
				res[var_name] = self.get_parameters_by_name(var_name)
			else:
				res[var_name] = self.get_parameters_by_name(var_name).tolist()
		return res

	def save_parameters(self, path = None):
		if path == None:
			path = self.out_path
		f = open(path, "w")
		f.write(json.dumps(self.get_parameters("list")))
		f.close()

	def set_parameters_by_name(self, var_name, tensor):
		with self.graph.as_default():
			with self.sess.as_default():
				if var_name in self.trainModel.parameter_lists:
					self.trainModel.parameter_lists[var_name].assign(tensor).eval()

	def set_parameters(self, lists):
		for i in lists:
			self.set_parameters_by_name(i, lists[i])

	def set_model(self, model):
		self.model = model
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.sess = tf.Session()
			with self.sess.as_default():
				initializer = tf.contrib.layers.xavier_initializer(uniform = True)
				with tf.variable_scope("model", reuse=None, initializer = initializer):
					self.trainModel = self.model(config = self)
					if self.optimizer != None:
						pass
					elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
						self.optimizer = tf.train.AdagradOptimizer(learning_rate = self.alpha, initial_accumulator_value=1e-20)
					elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
						self.optimizer = tf.train.AdadeltaOptimizer(self.alpha)
					elif self.opt_method == "Adam" or self.opt_method == "adam":
						self.optimizer = tf.train.AdamOptimizer(self.alpha)
					else:
						self.optimizer = tf.train.GradientDescentOptimizer(self.alpha)
					grads_and_vars = self.optimizer.compute_gradients(self.trainModel.loss)
					self.train_op = self.optimizer.apply_gradients(grads_and_vars)
				self.saver = tf.train.Saver()
				self.sess.run(tf.global_variables_initializer())

	def train_step(self, batch_h, batch_t, batch_r, batch_y):
		feed_dict = {
			self.trainModel.batch_h: batch_h,
			self.trainModel.batch_t: batch_t,
			self.trainModel.batch_r: batch_r,
			self.trainModel.batch_y: batch_y
		}
		_, loss = self.sess.run([self.train_op, self.trainModel.loss], feed_dict)
		return loss

	def test_step(self, test_h, test_t, test_r):
		feed_dict = {
			self.trainModel.predict_h: test_h,
			self.trainModel.predict_t: test_t,
			self.trainModel.predict_r: test_r,
		}
		predict = self.sess.run(self.trainModel.predict, feed_dict)
		return predict

	def run(self):
		triples=[]
		with open(self.in_path+"valid2id.txt", "r") as f:
			for i,line in enumerate(f):
				if i!=0:
					h,t,r=line.strip().split()
					triples.append((h,t,r))
		valid_triples=random.sample(triples,self.neval)
		patience_cnt = 0
		winner_MeanRank=1e05
		winner_hits10=winner_MRR=winner_acc=-1
		best_epoch=0
		with self.graph.as_default():
			with self.sess.as_default():
				if self.importName != None:
					self.restore_tensorflow()
				# self.trainModel.Normalize_Relations() #normalize relation embeddings
				# self.trainModel.Normalize_Entities() #normalize entity embeddings
				for times in range(self.train_times):
					res = 0.0
					for batch in range(self.nbatches):
						self.sampling()
						res += self.train_step(self.batch_h, self.batch_t, self.batch_r, self.batch_y)
						# self.trainModel.Normalize_Entities() #normalize entity embeddings
					if self.log_on:
						print(times)
						print(res)
					# if self.exportName != None and (self.export_steps!=0 and times % self.export_steps == 0):
					# 	self.save_tensorflow()
					if self.export_steps!=0 and times%self.export_steps==0:
						if self.KG=='FB15K' or self.KG=='WN18' or self.KG=="WN18RR" or self.KG=="FB15K237":
							MeanRank,MRR,hit10=self.Link_Prediction_on_validset(valid_triples)
							print "epoch:%d MeanRank:%f,MRR:%f, hits_10:%f"%(times,MeanRank,MRR,hit10)
							if MeanRank<winner_MeanRank or hit10>winner_hits10 or MRR>winner_MRR:
								if MeanRank<winner_MeanRank:
									winner_MeanRank=MeanRank
								if hit10>winner_hits10:
									winner_hits10=hit10
								if MRR>winner_MRR:
									winner_MRR=MRR
								print "Embedding changed to the best!"
								Best_embeddings = self.get_parameters("numpy")
								best_epoch=times
								patience_cnt=0
							else:
								patience_cnt+=1
							if patience_cnt>3:
								print "early stopping ... epoch number %d"%times
								print "Winner MeanRank:%f, winner hits@10:%f at epoch:%d"%(winner_MeanRank,winner_hits10,best_epoch)
								np.save("./embeddings/RelWalk_Embeddings/%s/Best_embeddings_%s_lr%.3f_d=%d_epochs=%d_l1=%d_l2=%d_m=%d_NegRate=%d_%s_%s.npy"\
									%(self.KG,self.opt_method,self.alpha,self.ent_size,self.train_times,self.lambda1,self.lambda2,self.margin,self.negative_ent,self.negSampMethod,self.LossFunc),\
									Best_embeddings)
								break
						elif self.KG=='WN11' or self.KG=='FB13':
							acc=self.Triple_Classification()
							print "epoch:%d triple classification acc:%f"%(times,acc)
							if acc>winner_acc:
								winner_acc=acc
								print "Embedding changed to the best!"
								Best_embeddings = self.get_parameters("numpy")
								best_epoch=times
								patience_cnt=0
							else:
								patience_cnt+=1
							if patience_cnt>3:
								print "early stopping ... epoch number %d"%times
								print "Winner Accuracy:%f at epoch:%d"%(winner_acc,best_epoch)
								np.save("./embeddings/RelWalk_Embeddings/%s/Best_embeddings_%s_lr%.3f_d=%d_epochs=%d_l1=%d_l2=%d_m=%d_NegRate=%d_%s_%s.npy"\
									%(self.KG,self.opt_method,self.alpha,self.ent_size,self.train_times,self.lambda1,self.lambda2,self.margin,self.negative_ent,self.negSampMethod,self.LossFunc),\
									Best_embeddings)
								break
				if self.exportName != None:
					self.save_tensorflow()
				if self.out_path != None:
					self.save_parameters(self.out_path)

	def test(self):
		with self.graph.as_default():
			with self.sess.as_default():
				if self.importName != None:
					self.restore_tensorflow()
				if self.test_link_prediction:
					total = self.lib.getTestTotal()
					for times in range(total):
						self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
						res = self.test_step(self.test_h, self.test_t, self.test_r)
						self.lib.testHead(res.__array_interface__['data'][0])

						self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
						res = self.test_step(self.test_h, self.test_t, self.test_r)
						self.lib.testTail(res.__array_interface__['data'][0])
						if self.log_on:
							print(times)
					self.lib.test_link_prediction()
				if self.test_triple_classification:
					self.lib.getValidBatch(self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr, self.valid_neg_h_addr, self.valid_neg_t_addr, self.valid_neg_r_addr)
					res_pos = self.test_step(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
					res_neg = self.test_step(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)
					self.lib.getBestThreshold(self.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])

					self.lib.getTestBatch(self.test_pos_h_addr, self.test_pos_t_addr, self.test_pos_r_addr, self.test_neg_h_addr, self.test_neg_t_addr, self.test_neg_r_addr)

					res_pos = self.test_step(self.test_pos_h, self.test_pos_t, self.test_pos_r)
					res_neg = self.test_step(self.test_neg_h, self.test_neg_t, self.test_neg_r)
					self.lib.test_triple_classification(self.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])
					
	def show_link_prediction(self, h, r):
		self.init_link_prediction()
		if self.importName != None:
			self.restore_tensorflow()
		test_h = np.array([h] * self.entTotal)
		test_r = np.array([r] * self.entTotal)
		test_t = np.array(range(self.entTotal))
		res = self.test_step(test_h, test_t, test_r).reshape(-1).argsort()[:10]
		print(res)
		return res
	def show_triple_classification(self, h, t, r):
		self.init_triple_classification()
		if self.importName != None:
			self.restore_tensorflow()
		self.lib.getValidBatch(self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr, self.valid_neg_h_addr, self.valid_neg_t_addr, self.valid_neg_r_addr)
		res_pos = self.test_step(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
		res_neg = self.test_step(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)
		self.lib.getBestThreshold(self.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])
		res = self.test_step(np.array([h]), np.array([t]), np.array([r]))
		if res >= self.relThresh[r]:
			print("triple (%d,%d,%d) is correct" % (h, t, r))
		else: 
			print("triple (%d,%d,%d) is wrong" % (h, t, r))
#--------------------------------------------------------
# I added the following part
#---------------------------------------------------------
	def Link_Prediction_on_validset(self,triples):
		print "Evaluating the embeddings on valid set..."
		raw_MeanRank=raw_MRR=raw_hit10=0.0
		k=1
		for (h,t,r) in triples:
			if k%500==0:
				print "Finish %d, completed %.2f percent"%(k,(float(k)/len(triples)*100))
			# Corrupt Heads (raw and filt)
			corrupted_head_triples=[(h_,t,r) for h_ in range(self.lib.getEntityTotal()) if h_!=h]
			raw_neg_head_triples=[(h,t,r)]+corrupted_head_triples #raw setting
			h_idx,t_idx,r_idx=shred_triples(raw_neg_head_triples)
			scores=self.test_step(h_idx, t_idx, r_idx)
			scores=scores.reshape(1,-1).flatten()
			ranks=(-scores).argsort()
			raw_neg_head_rank=np.where(ranks==0)[0][0]+1.0
			# Corrupt Tails (raw and filt)
			corrupted_tail_triples=[(h,t_,r) for t_ in range(self.lib.getEntityTotal()) if t_!=t]
			raw_neg_tail_triples=[(h,t,r)]+corrupted_tail_triples #raw setting
			h_idx,t_idx,r_idx=shred_triples(raw_neg_tail_triples)
			scores=self.test_step(h_idx, t_idx, r_idx)
			scores=scores.reshape(1,-1).flatten()
			ranks=(-scores).argsort()
			raw_neg_tail_rank=np.where(ranks==0)[0][0]+1.0
			raw_MeanRank += raw_neg_head_rank + raw_neg_tail_rank
			raw_MRR += ((1.0 / raw_neg_head_rank) + (1.0 / raw_neg_tail_rank))
			raw_hit10 += float(raw_neg_head_rank <= 10) + float(raw_neg_tail_rank <= 10)
			k+=1
		raw_MeanRank   /= (2.0 * len(triples))
		raw_MRR   /= (2.0 * len(triples))
		raw_hit10   /= (2.0 * len(triples))
		return raw_MeanRank,raw_MRR,raw_hit10*100.0
#---------------------------------------------------------
	def Triple_Classification(self):
		# Get Best Threshold for classification for each relation using validation set
		valid_dic=self.GetTripleSet('valid')
		interval=0.01
		BestThreshold={}
		BestAccuracy={}
		for rel in valid_dic:
			BestThreshold[rel]=0.0
			BestAccuracy[rel]=-1
		for rel in valid_dic:
			# Get prediction scores of triples in relation rel
			h_idx,t_idx,r_idx=shred_triples(valid_dic[rel])
			scores=self.test_step(h_idx, t_idx, r_idx)
			scores=scores.reshape(1,-1).flatten()
			min_=min(scores)
			max_=max(scores)
			current_threshold=min_
			while(current_threshold<max_):
				# compute accuracy in the current_threshold
				acc=ComputeAccuracy(valid_dic[rel],scores,current_threshold)
				if acc>BestAccuracy[rel]:
					BestAccuracy[rel]=acc
					BestThreshold[rel]=current_threshold
				current_threshold+=interval
		
		# Triple classification accuracy
		test_dic=self.GetTripleSet('test')
		total=corrects=0
		for rel in test_dic:
			# Get prediction scores of triples in relation rel
			h_idx,t_idx,r_idx=shred_triples(test_dic[rel])
			scores=self.test_step(h_idx, t_idx, r_idx)
			scores=scores.reshape(1,-1).flatten()
			for i,(h,t,r,lable) in enumerate(test_dic[rel]):
				total+=1
				if scores[i]>=BestThreshold[rel] and lable==1:
					corrects+=1
				if scores[i]<BestThreshold[rel] and lable==-1:
					corrects+=1
		acc=corrects/float(total)
		return acc
	#---------------------------------------------------------	
	def GetTripleSet(self,flag):
		d={}
		with open("./benchmarks/%s/%s_neg.txt"%(self.KG,flag),'r') as valid:
			for line in valid:
				line=line.strip().split()
				h,t,r,lable=int(line[0]),int(line[1]),int(line[2]),int(line[3])
				d.setdefault(r,[])
				d[r].append((h,t,r,lable))
		return d
#---------------------------------------------------------
def ComputeAccuracy(list_,scores,current_threshold):
	total=corrects=0.0
	for i,(h,t,r,lable) in enumerate(list_):
		total+=1
		if scores[i]>=current_threshold and lable==1:
			corrects+=1
		if scores[i]<current_threshold and lable==-1:
			corrects+=1
	acc=corrects/float(total)
	return acc
#---------------------------------------------------------
def shred_triples(triples):
	h_idx = [triples[i][0] for i in range(len(triples))]
	t_idx = [triples[i][1] for i in range(len(triples))]
	r_idx = [triples[i][2] for i in range(len(triples))]
	return h_idx, t_idx, r_idx
#----------------------------------------------------------

