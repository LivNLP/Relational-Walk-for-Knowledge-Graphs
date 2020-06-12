valid_list=[]
with open("valid2id.txt",'r') as valid:
	for i,line in enumerate(valid):
		if i!=0:
			h,t,r=line.strip().split()
			valid_list.append((h,t,r))
print len(valid_list)
import random

R=random.sample(valid_list,1000)
rel_set=set()
for (h,t,r) in R:
	rel_set.add(r)
print len(rel_set)
