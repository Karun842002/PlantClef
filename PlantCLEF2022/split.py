import json
from collections import defaultdict
clas = defaultdict(lambda: [])
order =  defaultdict(lambda: [])
family = defaultdict(lambda: [])
genus = defaultdict(lambda: [])
split = json.load(open('split.json'))
for k,v in split.items():
	clas[v["class"]] += [k]
	order[v["order"]] += [k]
	family[v["family"]] += [k]
	genus[v["genus"]] += [k]

json.dump(clas,open('class.json','w'))

json.dump(order,open('order.json','w'))

json.dump(family,open('family.json','w'))

json.dump(genus,open('genus.json','w'))
