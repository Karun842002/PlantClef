import os
def check(path):
	data = open(path,'rb').read(10)
	if data[:3] == b'\xff\xd8\xff': return True
	return False

train_path = '/home/jansi/plantclef/PlantCLEF2022/train/images'
val_path = '/home/jansi/plantclef/PlantCLEF2022/val/images' 
logfile = open('imageog.txt','w')
i = 0
for sub in os.listdir(train_path):
	ct = 0
	for img in os.listdir(os.path.join(train_path,sub)):
		if not check(os.path.join(train_path,sub,img)): ct+=1
	print("TRAIN:",sub,ct)
	if(ct>0): logfile.write("TRAIN:"+str(sub)+'-'+str(ct)+'\n')
	i+=1
	print(i,"of","80000")
i=0
for sub in os.listdir(val_path):
	vct = 0
	for img in os.listdir(os.path.join(val_path,sub)):
		if not check(os.path.join(val_path,sub,img)): vct+=1
	print("VAL:",sub, vct)
	if(vct>0): logfile.write("VAL:"+str(sub)+'-'+str(vct)+'\n')
	i+=1
	print(i,"of","80000")
