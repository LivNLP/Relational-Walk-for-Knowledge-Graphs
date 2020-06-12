import numpy as np


def Read_Embeddings():
	embeddings=np.load("KGEs_d100.npy",allow_pickle=True).item()
	Entity_Embeddings=embeddings['ent_embeddings']
	# dimensionality of entity embeddings
	d=Entity_Embeddings.shape[1]
	R1_Embeddings=embeddings['R1'].reshape(-1, d, d)
	R2_Embeddings=embeddings['R2'].reshape(-1, d, d)
	print "Entity embeddings of shape (num_of_entities,dim):", Entity_Embeddings.shape
	print "Relation R1 embeddings of shape (num_of_rel,dim,dim):", R1_Embeddings.shape
	print "Relation R2 embeddings of shape (num_of_rel,dim,dim):", R2_Embeddings.shape


if __name__=="__main__":
	Read_Embeddings()