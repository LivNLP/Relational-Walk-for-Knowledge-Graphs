#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model

class RelWalk(Model):

	def _transfer(self, transfer_matrix, embeddings):
		return tf.matmul(transfer_matrix, embeddings)

	def _calc(self, h, t):
		return tf.square(h + t)

	def embedding_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#Defining required parameters of the model, including embeddings of entities and relations, and mapping matrices
		self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [config.entTotal, config.ent_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.R1 = tf.get_variable(name = "R1", shape = [config.relTotal, config.ent_size * config.rel_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.R2 = tf.get_variable(name = "R2", shape = [config.relTotal, config.ent_size * config.rel_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		self.parameter_lists = {"ent_embeddings":self.ent_embeddings, \
								"R1":self.R1, \
								"R2":self.R2}
		self.I=tf.eye(config.rel_size,config.rel_size,[config.relTotal],dtype=tf.float32) #Identity tensor


	def loss_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#To get positive triples and negative triples for training
		#The shapes of pos_h, pos_t, pos_r are (batch_size, 1)
		#The shapes of neg_h, neg_t, neg_r are (batch_size, negative_ent + negative_rel)
		pos_h, pos_t, pos_r = self.get_positive_instance(in_batch = True)
		neg_h, neg_t, neg_r = self.get_negative_instance(in_batch = True)
		#Embedding entities and relations of triples, e.g. pos_h_e, pos_t_e and pos_r_e are embeddings for positive triples
		pos_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, pos_h), [-1, config.ent_size, 1])
		pos_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, pos_t), [-1, config.ent_size, 1])
		neg_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, neg_h), [-1, config.ent_size, 1])
		neg_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, neg_t), [-1, config.ent_size, 1])
		#Getting the required mapping matrices
		pos_R1_matrix = tf.reshape(tf.nn.embedding_lookup(self.R1, pos_r), [-1, config.rel_size, config.ent_size])
		pos_R2_matrix = tf.reshape(tf.nn.embedding_lookup(self.R2, pos_r), [-1, config.rel_size, config.ent_size])
		neg_R1_matrix = tf.reshape(tf.nn.embedding_lookup(self.R1, neg_r), [-1, config.rel_size, config.ent_size])
		neg_R2_matrix = tf.reshape(tf.nn.embedding_lookup(self.R2, neg_r), [-1, config.rel_size, config.ent_size])
		#Calculating score functions for all positive triples and negative triples
		p_h = tf.reshape(self._transfer(pos_R1_matrix, pos_h_e), [-1, config.rel_size])
		p_t = tf.reshape(self._transfer(pos_R2_matrix, pos_t_e), [-1, config.rel_size])

		n_h = tf.reshape(self._transfer(neg_R1_matrix, neg_h_e), [-1, config.rel_size])
		n_t = tf.reshape(self._transfer(neg_R2_matrix, neg_t_e), [-1, config.rel_size])

		#The shape of _p_score is (batch_size, 1, hidden_size)
		#The shape of _n_score is (batch_size, negative_ent + negative_rel, hidden_size)
		_p_score = self._calc(p_h, p_t)
		_p_score = tf.reshape(_p_score, [-1, 1, config.rel_size])
		_n_score = self._calc(n_h, n_t)
		_n_score = tf.reshape(_n_score, [-1, config.negative_ent + config.negative_rel, config.rel_size])

		regulariser1 = tf.nn.l2_loss(tf.matmul(tf.transpose(tf.reshape(self.R1,[-1, config.rel_size, config.ent_size]),perm=[0, 2, 1]),tf.reshape(self.R1,[-1, config.rel_size, config.ent_size]))-self.I)
		regulariser2 = tf.nn.l2_loss(tf.matmul(tf.transpose(tf.reshape(self.R2,[-1, config.rel_size, config.ent_size]),perm=[0, 2, 1]),tf.reshape(self.R2,[-1, config.rel_size, config.ent_size]))-self.I)
		regulariser=(config.lambda1*regulariser1)+(config.lambda2*regulariser2)

		if config.LossFunc=="LogSoftmax":
			#The shape of p_score is (batch_size, 1)
			#The shape of n_score is (batch_size, 1)
			p_score =  tf.reduce_sum(tf.reduce_mean(_p_score, 1, keepdims = False), 1, keepdims = True)
			n_score =  tf.reduce_sum(_n_score, 2, keepdims = False)
			scores= tf.concat([p_score,n_score],1)
			prob=tf.nn.log_softmax(scores)
			pos_prob=tf.gather(prob,[0],axis=1) #The first column is the probabilities of the positive triples
			self.loss = tf.reduce_sum(-pos_prob) + regulariser
		elif config.LossFunc=='Marginal': # default is Marginal loss
			#The shape of p_score is (batch_size, 1)
			#The shape of n_score is (batch_size, 1)
			p_score =  tf.reduce_sum(tf.reduce_mean(_p_score, 1, keepdims = False), 1, keepdims = True)
			n_score =  tf.reduce_sum(tf.reduce_mean(_n_score, 1, keepdims = False), 1, keepdims = True)
			self.loss = tf.reduce_sum(tf.maximum(-p_score + n_score + config.margin, 0)) + regulariser
		
	def predict_def(self):
		config = self.get_config()
		predict_h, predict_t, predict_r = self.get_predict_instance()
		predict_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, predict_h), [-1, config.ent_size, 1])
		predict_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, predict_t), [-1, config.ent_size, 1])
		predict_R1 = tf.reshape(tf.nn.embedding_lookup(self.R1, predict_r), [-1, config.rel_size, config.ent_size])
		predict_R2 = tf.reshape(tf.nn.embedding_lookup(self.R2, predict_r), [-1, config.rel_size, config.ent_size])
		h_e = tf.reshape(self._transfer(predict_R1, predict_h_e), [-1, config.rel_size])
		t_e = tf.reshape(self._transfer(predict_R2, predict_t_e), [-1, config.rel_size])
		self.predict = tf.reduce_sum(self._calc(h_e, t_e), 1, keepdims = True)
