import os

path = '/home/jansi/plantclef/PlantCLEF2022'
f = open('imageog.txt').readlines()

def check(path):
        data = open(path,'rb').read(10)
        if data[:3] == b'\xff\xd8\xff': return True
        return False
i = 0
for x in f:
	x = x.replace('-',':')
	x = x.split(':')
	print(x)
	pathn = os.path.join(path,x[0].lower(),'images',x[1])
	for file in os.listdir(pathn):
		if not check(os.path.join(pathn,file)):
			os.remove(os.path.join(pathn,file))
	i+=1
	print(i,"of",len(f))
