import pymatgen.core as mg
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from .atom_init import Atom_init
from bokeh.sampledata.periodic_table import elements


class Features:

    def info(self):
        print("featurename is " + self.featurename)

    def get_VEC(self, elementstr):
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

    def get_comp_dict(self, composition):
        try:
            return dict(mg.Composition(composition).fractional_composition.get_el_amt_dict())
        except:
            return {}

    def get_atom_init(self):
        atom_init = Atom_init.cgcnn_atom_init()
        return atom_init
    
    def get_ave_atom_init(self, composition):
        compdict = self.get_comp_dict(composition)

        atom_init = Atom_init.cgcnn_atom_init()
        el_dict = dict(elements[["symbol","atomic number"]].values)

        try:
            if len(compdict) > 1:
                tmp = 0
                for el, _ in compdict.items():
                    tmp += np.array(atom_init[el_dict[el]])/len(compdict)
                return np.array(tmp)
            elif len(compdict) == 1:
                tmp = atom_init[el_dict[list(compdict.keys())[0]]]
                return np.array(tmp)
        except:
            #Elements with atomic number more than 100
            return np.array([np.nan]*len(atom_init[1]))

    def ave(self, composition, description):
        try:
            composition = self.get_comp_dict(composition)
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
            elif description == "block":
                block_dict = dict(zip(["s","p","d","f"],[0]*4))
                if len(composition) > 1:
                    for el, frac in composition.items():
                        block_dict[mg.Element(el).block] += frac / sum(composition.values())
                    return block_dict
                elif len(composition) == 1:
                    block_dict[mg.Element(list(composition.keys())[0]).block] = 1
                    return block_dict
            elif description == "VEC":
                if len(composition) > 1:
                    tmp = 0
                    for el, frac in composition.items():
                        tmp += self.get_VEC(el)*frac / sum(composition.values())
                    return tmp
                elif len(composition) == 1:
                    tmp = self.get_VEC(list(composition.keys())[0])
                    return tmp
            elif description == "First ionization energies":
                if len(composition) > 1:
                    try:
                        tmp = 0
                        for el, frac in composition.items():
                            tmp += np.log10(mg.Element(el).data["Ionization energies"][0])*frac / sum(composition.values())
                        return tmp
                    except:
                        pass
                elif len(composition) == 1:
                    tmp = np.log10(mg.Element(list(composition.keys())[0]).data["Ionization energies"][0])
                    return tmp

            elif description == "Molar volume":
                if len(composition) > 1:
                    tmp = 0
                    for el, frac in composition.items():
                        tmp += np.log10(float(mg.Element(el).data[description].split("cm<sup>3</sup>")[0]))*frac / sum(composition.values())
                    return tmp
                elif len(composition) == 1:
                    tmp = np.log10(float(mg.Element(list(composition.keys())[0]).data[description].split("cm<sup>3</sup>")[0]))
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
                        return tmp
                elif len(composition) == 1:
                    return mg.Element(list(composition.keys())[0]).data[description]
        except:
            #print(composition, description)
            return np.nan

    def var(self, composition, description):
        composition = self.get_comp_dict(composition=composition)
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
        elif description == "block":
            block_dict = dict(zip(["s","p","d","f"],[0]*4))
            ave_block = self.ave(composition,description)
            if len(composition) > 1:
                for el, frac in composition.items():
                    block_idx = dict(zip(["s","p","d","f"],[0]*4))
                    block_idx[mg.Element(el).block] = 1
                    for k, v in block_idx.items():
                        block_dict[k] += ((v - ave_block[k])**2)*frac / sum(composition.values())
                return block_dict
            elif len(composition) == 1:
                return dict(zip(["s","p","d","f"],[0]*4))
        elif description == "VEC":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp += ((self.get_VEC(el)-self.ave(composition,description))**2)*frac / sum(composition.values())
                return tmp
            elif len(composition) == 1:
                return 0
        elif description == "First ionization energies":
            if len(composition) > 1:
                try:
                    tmp = 0
                    for el, frac in composition.items():
                        tmp += ((np.log10(mg.Element(el).data["Ionization energies"][0])-self.ave(composition, description))**2)*frac / sum(composition.values())
                    return tmp
                except:
                    pass
            elif len(composition) == 1:
                return 0
        elif description == "Molar volume":
            if len(composition) > 1:
                tmp = 0
                for el, frac in composition.items():
                    tmp +=( (np.log10(float(mg.Element(el).data[description].split("cm<sup>3</sup>")[0]))-self.ave(composition,description))**2)*frac / sum(composition.values())
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

    def main_max1min1diff(self, composition, description):
        composition = self.get_comp_dict(composition=composition)
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
        elif description == "block":
            if len(composition) > 1:
                block_max_dict = dict(zip(["s","p","d","f"],[0]*4))
                block_min_dict = dict(zip(["s","p","d","f"],[1]*4))
                for el in np.array(list(composition.keys()))[np.array(list(composition.values()))>=0.1]:
                    block_idx = dict(zip(["s","p","d","f"],[0]*4))
                    block_idx[mg.Element(el).block] = 1
                    for k, v in block_idx.items():
                        if v >= block_max_dict[k]:
                            block_max_dict[k] = v
                        if v <= block_min_dict[k]:
                            block_min_dict[k] = v
                return dict(zip(["s","p","d","f"],np.array(list(block_max_dict.values())) - np.array(list(block_min_dict.values()))))
            elif len(composition) == 1:
                return dict(zip(["s","p","d","f"],[0]*4))
        elif description == "VEC":
            if len(composition) > 1:
                maxval = 0
                minval = 1000000
                for el in np.array(list(composition.keys()))[np.array(list(composition.values()))>=0.1]:
                    val = float(self.get_VEC(el))
                    if val >= maxval:
                        maxval = val
                    if val <= minval:
                        minval = val
                return np.abs(maxval - minval)
            elif len(composition) == 1:
                return 0
            
        elif description == "First ionization energies":
            if len(composition) > 1:
                try:
                    maxval = 0
                    minval = 1000000
                    for el in np.array(list(composition.keys()))[np.array(list(composition.values()))>=0.1]:
                        val = np.log10(float(mg.Element(el).data["Ionization energies"][0]))
                        if val >= maxval:
                            maxval = val
                        if val <= minval:
                            minval = val
                    return np.abs(maxval - minval)
                except:
                    pass
            elif len(composition) == 1:
                return 0
        elif description == "Molar volume":
            if len(composition) > 1:
                maxval = 0
                minval = 1000000
                for el in np.array(list(composition.keys()))[np.array(list(composition.values()))>=0.1]:
                    val = np.log10(float(mg.Element(el).data[description].split("cm<sup>3</sup>")[0]))
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

    def get_comp_desc(self,composition, func=["ave","var","main_max1min1diff"],desclist=["comp_length","compbase_length","Atomic no","group","row","Mendeleev no","Atomic mass","Atomic radius","X","VEC"]):
        try:
            compdict = self.get_comp_dict(composition=composition)

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
                if "ave" in func:
                    if "block" == desc:
                        blocks = self.ave(compdict,desc)
                        for k, v in blocks.items():
                            response.update({"ave:"+k: v})
                    else:
                        ave_tmp = self.ave(compdict,desc)
                        if ave_tmp != "no data":
                            response.update({"ave:"+desc: ave_tmp})
                        else:
                            response.update({"ave:"+desc: 0})
                if "var" in func:
                    if "block" == desc:
                        blocks = self.var(compdict,desc)
                        for k, v in blocks.items():
                            response.update({"var:"+k: v})
                    else:
                        var_tmp = self.var(compdict,desc)
                        if var_tmp != "no data":
                            response.update({ "var:"+desc: var_tmp})
                        else:
                            response.update({"var:"+desc: 0})
                if "main_max1min1diff" in func:
                    if "block" == desc:
                        blocks = self.main_max1min1diff(compdict,desc)
                        for k, v in blocks.items():
                            response.update({"main_max1min1diff:"+k: v})
                    else:
                        diff_tmp = self.main_max1min1diff(compdict,desc)
                        if diff_tmp != "no data":
                            response.update({ "main_max1min1diff:"+desc: diff_tmp})
                        else:
                            response.update({"main_max1min1diff:"+desc: 0})
            return response
        except:
            response = {}
            block_dict = dict(zip(["s","p","d","f"],[np.nan]*4))
            desc_tmp = desclist.copy()
            if "comp_length" in desc_tmp:
                response.update({"comp_length":np.nan})
                desc_tmp.remove("comp_length")

            if "compbase_length" in desc_tmp:
                response.update({"compbase_length":np.nan})
                desc_tmp.remove("compbase_length")

            for desc in desc_tmp:
                if "ave" in func:
                    if "block" == desc:
                        blocks = block_dict
                        for k, v in blocks.items():
                            response.update({"ave:"+k: v})
                    else:
                        response.update({"ave:"+desc: np.nan})
                if "var" in func:
                    if "block" == desc:
                        blocks = block_dict
                        for k, v in blocks.items():
                            response.update({"var:"+k: v})
                    else:
                        response.update({ "var:"+desc: np.nan})
                if "main_max1min1diff" in func:
                    if "block" == desc:
                        blocks = block_dict
                        for k, v in blocks.items():
                            response.update({"main_max1min1diff:"+k: v})
                    else:
                        response.update({"main_max1min1diff:"+desc: np.nan})
            return response

    def get_comp_descfeatures(self, complist, func=["ave","var","main_max1min1diff"], desclist=["comp_length", "compbase_length", "Atomic no", "group", "row", "Mendeleev no", "Atomic mass", "Atomic radius", "X", "VEC"]):
        features = []
        for comp in complist:
            tmp = {"composition": comp}
            tmp.update(self.get_comp_desc(comp, func, desclist))
            features.append(tmp)
        df_feature = pd.DataFrame(features)
        return df_feature

    def get_tetrahedron_index(self, i,ijks):
        nn_faces = np.array([], dtype=np.float32)
        for j, ijk_2 in enumerate(ijks):
            for idx in range(4):
                ijk_tmp = list(ijks[i].copy())
                ijk_tmp.pop(idx)
                if (len(set(ijk_tmp)&set(ijk_2))==3) and (i!=j):
                    #print(ijk_tmp,ijk_2)
                    nn_faces = np.append(nn_faces,j)
                    break
        if len(nn_faces) < 4:
            nn_faces = np.append(nn_faces,[i]*(4-len(nn_faces)))
        return nn_faces

    def get_delaunay_feature(self, pts, ijks, atom_species):
        mesh_tmp = pts[ijks[0][0]].astype(np.float32)
        mesh_tmp = np.append(mesh_tmp,pts[ijks[0][1]].astype(np.float32), axis=0)
        mesh_tmp = np.append(mesh_tmp,pts[ijks[0][2]].astype(np.float32), axis=0)
        mesh_tmp = np.append(mesh_tmp,pts[ijks[0][3]].astype(np.float32), axis=0)
        
        mesh_tmp = np.append(mesh_tmp,self.get_tetrahedron_index(0,ijks).astype(np.float32), axis=0)

        comp = atom_species[ijks[0][0]]+atom_species[ijks[0][1]]+atom_species[ijks[0][2]]
        #comp = mg.Composition(comp_tmp).fractional_composition.formula
        
        feature = self.get_ave_atom_init(comp).astype(np.float32)
        mesh_tmp = np.append(mesh_tmp,feature, axis=0)

        mesh_data = mesh_tmp.reshape(1, len(mesh_tmp))

        for idx, ijk in enumerate(ijks[1:]):
            mesh_tmp = pts[ijk[0]].astype(np.float32)
            mesh_tmp = np.append(mesh_tmp,pts[ijk[1]].astype(np.float32), axis=0)
            mesh_tmp = np.append(mesh_tmp,pts[ijk[2]].astype(np.float32), axis=0)
            mesh_tmp = np.append(mesh_tmp,pts[ijk[3]].astype(np.float32), axis=0)
            
            mesh_tmp = np.append(mesh_tmp,self.get_tetrahedron_index(1+idx,ijks).astype(np.float32), axis=0)

            comp = atom_species[ijk[0]]+atom_species[ijk[1]]+atom_species[ijk[2]]
            
            feature = self.get_ave_atom_init(comp).astype(np.float32)
            mesh_tmp = np.append(mesh_tmp,feature, axis=0)

            #print(mesh_data)
            #print(mesh_tmp)
            mesh_data = np.append(mesh_data,mesh_tmp.reshape(1, len(mesh_tmp)), axis=0)

        return mesh_data

    def create_ctn_datasets(self, delaunay_alldata):
        pts, ijks, _, atom_species, _, _, _ = delaunay_alldata
        mesh_data = self.get_delaunay_feature(pts, ijks, atom_species)
        if np.isnan(mesh_data.flatten()).sum() == 0:
            return mesh_data

