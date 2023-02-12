import random
import os
import csv
class_names = os.listdir('/home/jansi/plantclef/PlantCLEF2022/train/images')
r = csv.reader(open('/home/jansi/plantclef/PlantCLEF2022/test/test/PlantCLEF2022_test_metadata.csv'), delimiter=';')
res=[]
obid = set()
j=0
for row in r:
	if j==0: 
		j=1
		continue
	if row[0] not in obid: obid.add(row[0])
for row in obid:
	random.shuffle(class_names)
	if j==0:
		j=1
		continue
	for i in range(30):
		res.append([row,class_names[i],1.25e-05,i+1])
	print(j,"of",len(obid))
	j+=1
import csv
f = open('res.csv','w')
w = csv.writer(f, delimiter=';')
w.writerows(res)
