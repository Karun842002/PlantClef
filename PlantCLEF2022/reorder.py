import json
import os
import pathlib
import shutil
meta = json.load(open('metadata.json'))
for clas,cv in meta.items():
    for order,ov in cv["orders"].items():
        for family, fv in ov["families"].items():
            for genus, gv in fv["genus"].items():
                pathlib.Path(os.path.join('/home/jansi/plantclef/PlantCLEF2022/train',clas,order,family,genus)).mkdir(parents=True, exist_ok = True)

split = json.load(open('split.json'))
i = 1
for species, sv in split.items():
    source = os.path.join('/home/jansi/plantclef/PlantCLEF2022/train/images', species)
    dest = os.path.join('/home/jansi/plantclef/PlantCLEF2022/train',sv["class"],sv["order"],sv["family"],sv["genus"],species)
    try:
        shutil.copytree(source, dest)
    except Exception:
        pass
    print(i," of 80000")
    i+=1
