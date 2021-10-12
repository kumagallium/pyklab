import pymatgen.core as mg
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

class Features:

    def info(self):
        print("featurename is " + self.featurename)
    
    def getVEC(self,elementstr):
        elobj = mg.Element(elementstr)
        VEC = 0
        if elobj.is_lanthanoid:
            VEC = 3
        elif elobj.is_actinoid:
            if elobj.name == "Np":
                VEC = 5
            elif elobj.name == "Pu":
                VEC = 4
            elif elobj.name == "Es":
                VEC = 2
            else:
                VEC = 3
        else:
            group = elobj.group
            if group > 12:
                VEC = group - 10
            else:
                VEC = group

            if group == 17:
                VEC = 0

        return VEC
    
    def getcompdict(self,composition):
        return dict(mg.Composition(composition).fractional_composition.get_el_amt_dict())

    def ave(self,composition,property):
        composition = self.getcompdict(composition)
        if property == "group":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += mg.Element(el).group*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                tmp = mg.Element(list(composition.keys())[0]).group
                return tmp
        elif property == "row":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += mg.Element(el).row*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                tmp = mg.Element(list(composition.keys())[0]).row
                return tmp
        elif property == "VEC":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += self.getVEC(el)*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                tmp = self.getVEC(list(composition.keys())[0])
                return tmp
        elif property == "Molar volume":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += float(mg.Element(el).data[property].split("cm<sup>3</sup>")[0])*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                tmp = float(mg.Element(list(composition.keys())[0]).data[property].split("cm<sup>3</sup>")[0])
                return tmp
        elif property == "Bulk modulus":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += float(mg.Element(el).data[property].split("(liquid)GPa")[0].split("GPa")[0])*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                tmp = float(mg.Element(list(composition.keys())[0]).data[property].split("(liquid)GPa")[0].split("GPa")[0])
                return tmp
        elif property == "Melting point":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += float(mg.Element(el).data[property].split("K")[0])*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                tmp = float(mg.Element(list(composition.keys())[0]).data[property].split("K")[0])
                return tmp
        elif property == "Boiling point":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += float(mg.Element(el).data[property].split("K")[0])*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                tmp = float(mg.Element(list(composition.keys())[0]).data[property].split("K")[0])
                return tmp
        else:
            if len(composition) > 1:
                try:
                    tmp = 0
                    for el, frac in composition.items():
                        tmp += mg.Element(el).data[property]*frac / sum(composition.values())
                    return tmp
                except:
                    print("error")
                    pass
            elif len(composition) == 1:
                tmp = mg.Element(list(composition.keys())[0]).data[property]
                return tmp

    def var(self,composition,property):
        composition = self.getcompdict(composition=composition)
        if property == "group":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += ((mg.Element(el).group-self.ave(composition,property))**2)*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                return 0
        elif property == "row":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += ((mg.Element(el).row-self.ave(composition,property))**2)*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                return 0
        elif property == "VEC":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += ((self.getVEC(el)-self.ave(composition,property))**2)*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                return 0
        elif property == "Molar volume":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp +=( (float(mg.Element(el).data[property].split("cm<sup>3</sup>")[0])-self.ave(composition,property))**2)*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                return 0
        elif property == "Bulk modulus":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += ((float(mg.Element(el).data[property].split("(liquid)GPa")[0].split("GPa")[0])-self.ave(composition,property))**2)*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                return 0
        elif property == "Melting point":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += ((float(mg.Element(el).data[property].split("K")[0])-self.ave(composition,property))**2)*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                return 0
        elif property == "Boiling point":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += ((float(mg.Element(el).data[property].split("K")[0])-self.ave(composition,property))**2)*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                return 0
        else:
            if len(composition) > 1:
                try:
                    tmp = 0
                    for el, frac in composition.items():
                        tmp += ((mg.Element(el).data[property]-self.ave(composition,property))**2)*frac / sum(composition.values())
                    return tmp
                except:
                    pass
            elif len(composition) == 1:
                return 0

    def main_max1min1diff(self,composition,property):
        composition = self.getcompdict(composition=composition)
        if property == "group":
            if len(composition) > 1:
                maxval = 0
                minval = 1000000
                for el in np.array(list(composition.keys()))[np.array(list(composition.values()))>=0.1]:
                    val = float(mg.Element(el).group)
                    if val >= maxval:
                        maxval = val
                    if val <= minval:
                        minval = val
                return np.abs(maxval - minval)
            elif len(composition) == 1:
                return 0
        elif property == "row":
            if len(composition) > 1:
                maxval = 0
                minval = 1000000
                for el in np.array(list(composition.keys()))[np.array(list(composition.values()))>=0.1]:
                    val = float(mg.Element(el).row)
                    if val >= maxval:
                        maxval = val
                    if val <= minval:
                        minval = val
                return np.abs(maxval - minval)
            elif len(composition) == 1:
                return 0
        elif property == "VEC":
            if len(composition) > 1:
                maxval = 0
                minval = 1000000
                for el in np.array(list(composition.keys()))[np.array(list(composition.values()))>=0.1]:
                    val = float(self.getVEC(el))
                    if val >= maxval:
                        maxval = val
                    if val <= minval:
                        minval = val
                return np.abs(maxval - minval)
            elif len(composition) == 1:
                return 0
            
        elif property == "Molar volume":
            if len(composition) > 1:
                maxval = 0
                minval = 1000000
                for el in np.array(list(composition.keys()))[np.array(list(composition.values()))>=0.1]:
                    val = float(mg.Element(el).data[property].split("cm<sup>3</sup>")[0])
                    if val >= maxval:
                        maxval = val
                    if val <= minval:
                        minval = val
                return np.abs(maxval - minval)
            elif len(composition) == 1:
                return 0
        elif property == "Boiling point":
            if len(composition) > 1:
                maxval = 0
                minval = 1000000
                for el in np.array(list(composition.keys()))[np.array(list(composition.values()))>=0.1]:
                    val = float(mg.Element(el).data[property].split("K")[0])
                    if val >= maxval:
                        maxval = val
                    if val <= minval:
                        minval = val
                return np.abs(maxval - minval)
            elif len(composition) == 1:
                return 0
            
        elif property == "Bulk modulus":
            if len(composition) > 1:
                maxval = 0
                minval = 1000000
                for el in np.array(list(composition.keys()))[np.array(list(composition.values()))>=0.1]:
                    val = float(mg.Element(el).data[property].split("(liquid)GPa")[0].split("GPa")[0])
                    if val >= maxval:
                        maxval = val
                    if val <= minval:
                        minval = val
                return np.abs(maxval - minval)
            elif len(composition) == 1:
                return 0
            
        elif property == "Melting point":
            if len(composition) > 1:
                maxval = 0
                minval = 1000000
                for el in np.array(list(composition.keys()))[np.array(list(composition.values()))>=0.1]:
                    val = float(mg.Element(el).data[property].split("K")[0])
                    if val >= maxval:
                        maxval = val
                    if val <= minval:
                        minval = val
                return np.abs(maxval - minval)
            elif len(composition) == 1:
                return 0
            
        else:
            if len(composition) > 1:
                try:
                    maxval = 0
                    minval = 1000000
                    for el in np.array(list(composition.keys()))[np.array(list(composition.values()))>=0.1]:
                        val = float(mg.Element(el).data[property])
                        if val >= maxval:
                            maxval = val
                        if val <= minval:
                            minval = val
                    return np.abs(maxval - minval)
                except:
                    pass
            elif len(composition) == 1:
                return 0

    def getCompProps(self,composition,proplist=["Atomic no","group","row","Mendeleev no","Atomic mass","Atomic radius","X","VEC"]):
        try:
            compdict = self.getcompdict(composition=composition)
            
            comp_length = len(compdict)
            compbase_length = len(np.array(list(compdict.keys()))[np.array(list(compdict.values()))>=0.1])
            response = {"comp_length":comp_length,"compbase_length":compbase_length}
            
            for prop in proplist:
                response.update({"ave:"+prop:self.ave(compdict,prop),"var:"+prop:self.var(compdict,prop),"main_max1min1diff:"+prop:self.main_max1min1diff(compdict,prop)})
            
            return response
        except:
            response = {"comp_length":np.nan,"compbase_length":np.nan}
            for prop in proplist:
                response.update({"ave:"+prop:np.nan,"var:"+prop:np.nan,"main_max1min1diff:"+prop:np.nan})#,"harm:"+prop:np.nan})
            return response

    def getCompPropsFeature(self,complist):
        features = []
        for comp in tqdm(complist):
            tmp = {"composition":comp}
            tmp.update(self.getCompProps(comp))
            features.append(tmp)
        df_feature = pd.DataFrame(features)
        return df_feature

