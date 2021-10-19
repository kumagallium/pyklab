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

    def ave(self,composition,description):
        composition = self.getcompdict(composition)
        if description == "group":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += mg.Element(el).group*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                tmp = mg.Element(list(composition.keys())[0]).group
                return tmp
        elif description == "row":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += mg.Element(el).row*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                tmp = mg.Element(list(composition.keys())[0]).row
                return tmp
        elif description == "VEC":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += self.getVEC(el)*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                tmp = self.getVEC(list(composition.keys())[0])
                return tmp
        elif description == "Molar volume":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += float(mg.Element(el).data[description].split("cm<sup>3</sup>")[0])*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                tmp = float(mg.Element(list(composition.keys())[0]).data[description].split("cm<sup>3</sup>")[0])
                return tmp
        elif description == "Bulk modulus":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += float(mg.Element(el).data[description].split("(liquid)GPa")[0].split("GPa")[0])*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                tmp = float(mg.Element(list(composition.keys())[0]).data[description].split("(liquid)GPa")[0].split("GPa")[0])
                return tmp
        elif description == "Melting point":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += float(mg.Element(el).data[description].split("K")[0])*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                tmp = float(mg.Element(list(composition.keys())[0]).data[description].split("K")[0])
                return tmp
        elif description == "Boiling point":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += float(mg.Element(el).data[description].split("K")[0])*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                tmp = float(mg.Element(list(composition.keys())[0]).data[description].split("K")[0])
                return tmp
        else:
            if len(composition) > 1:
                try:
                    tmp = 0
                    for el, frac in composition.items():
                        tmp += mg.Element(el).data[description]*frac / sum(composition.values())
                    return tmp
                except:
                    pass
            elif len(composition) == 1:
                tmp = mg.Element(list(composition.keys())[0]).data[description]
                return tmp

    def var(self,composition,description):
        composition = self.getcompdict(composition=composition)
        if description == "group":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += ((mg.Element(el).group-self.ave(composition,description))**2)*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                return 0
        elif description == "row":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += ((mg.Element(el).row-self.ave(composition,description))**2)*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                return 0
        elif description == "VEC":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += ((self.getVEC(el)-self.ave(composition,description))**2)*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                return 0
        elif description == "Molar volume":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp +=( (float(mg.Element(el).data[description].split("cm<sup>3</sup>")[0])-self.ave(composition,description))**2)*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                return 0
        elif description == "Bulk modulus":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += ((float(mg.Element(el).data[description].split("(liquid)GPa")[0].split("GPa")[0])-self.ave(composition,description))**2)*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                return 0
        elif description == "Melting point":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += ((float(mg.Element(el).data[description].split("K")[0])-self.ave(composition,description))**2)*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                return 0
        elif description == "Boiling point":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += ((float(mg.Element(el).data[description].split("K")[0])-self.ave(composition,description))**2)*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                return 0
        else:
            if len(composition) > 1:
                try:
                    tmp = 0
                    for el, frac in composition.items():
                        tmp += ((mg.Element(el).data[description]-self.ave(composition,description))**2)*frac / sum(composition.values())
                    return tmp
                except:
                    pass
            elif len(composition) == 1:
                return 0

    def main_max1min1diff(self,composition,description):
        composition = self.getcompdict(composition=composition)
        if description == "group":
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
        elif description == "row":
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
        elif description == "VEC":
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
            
        elif description == "Molar volume":
            if len(composition) > 1:
                maxval = 0
                minval = 1000000
                for el in np.array(list(composition.keys()))[np.array(list(composition.values()))>=0.1]:
                    val = float(mg.Element(el).data[description].split("cm<sup>3</sup>")[0])
                    if val >= maxval:
                        maxval = val
                    if val <= minval:
                        minval = val
                return np.abs(maxval - minval)
            elif len(composition) == 1:
                return 0
        elif description == "Boiling point":
            if len(composition) > 1:
                maxval = 0
                minval = 1000000
                for el in np.array(list(composition.keys()))[np.array(list(composition.values()))>=0.1]:
                    val = float(mg.Element(el).data[description].split("K")[0])
                    if val >= maxval:
                        maxval = val
                    if val <= minval:
                        minval = val
                return np.abs(maxval - minval)
            elif len(composition) == 1:
                return 0
            
        elif description == "Bulk modulus":
            if len(composition) > 1:
                maxval = 0
                minval = 1000000
                for el in np.array(list(composition.keys()))[np.array(list(composition.values()))>=0.1]:
                    val = float(mg.Element(el).data[description].split("(liquid)GPa")[0].split("GPa")[0])
                    if val >= maxval:
                        maxval = val
                    if val <= minval:
                        minval = val
                return np.abs(maxval - minval)
            elif len(composition) == 1:
                return 0
            
        elif description == "Melting point":
            if len(composition) > 1:
                maxval = 0
                minval = 1000000
                for el in np.array(list(composition.keys()))[np.array(list(composition.values()))>=0.1]:
                    val = float(mg.Element(el).data[description].split("K")[0])
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
                        val = float(mg.Element(el).data[description])
                        if val >= maxval:
                            maxval = val
                        if val <= minval:
                            minval = val
                    return np.abs(maxval - minval)
                except:
                    pass
            elif len(composition) == 1:
                return 0

    def getCompDesc(self,composition,desclist=["comp_length","compbase_length","Atomic no","group","row","Mendeleev no","Atomic mass","Atomic radius","X","VEC"]):
        try:
            compdict = self.getcompdict(composition=composition)
            
            comp_length = len(compdict)
            compbase_length = len(np.array(list(compdict.keys()))[np.array(list(compdict.values()))>=0.1])
            response = {}
            desc_tmp = desclist.copy()
            if "comp_length" in desc_tmp:
                response.update({"comp_length":comp_length})
                desc_tmp.remove("comp_length")

            if "compbase_length" in desc_tmp:
                response.update({"compbase_length":compbase_length})
                desc_tmp.remove("compbase_length")
            
            for desc in desc_tmp:
                response.update({"ave:"+desc:self.ave(compdict,desc),"var:"+desc:self.var(compdict,desc),"main_max1min1diff:"+desc:self.main_max1min1diff(compdict,desc)})
            
            return response
        except:
            response = {}
            desc_tmp = desclist.copy()
            if "comp_length" in desc_tmp:
                response.update({"comp_length":np.nan})
                desc_tmp.remove("comp_length")

            if "compbase_length" in desc_tmp:
                response.update({"compbase_length":np.nan})
                desc_tmp.remove("compbase_length")
                
            for desc in desc_tmp:
                response.update({"ave:"+desc:np.nan,"var:"+desc:np.nan,"main_max1min1diff:"+desc:np.nan})#,"harm:"+feat:np.nan})
            return response

    def getCompDescFeatures(self,complist,desclist=["comp_length","compbase_length","Atomic no","group","row","Mendeleev no","Atomic mass","Atomic radius","X","VEC"]):
        features = []
        for comp in tqdm(complist):
            tmp = {"composition":comp}
            tmp.update(self.getCompDesc(comp,desclist))
            features.append(tmp)
        df_feature = pd.DataFrame(features)
        return df_feature

