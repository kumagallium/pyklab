import pandas as pd
import numpy as np
from pymatgen import MPRester
import plotly.graph_objects as go
from scipy.spatial import Delaunay
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import itertools
from bokeh.sampledata.periodic_table import elements
import pymatgen as mg

class Structure():
    def setSigFig(self, val):
        try:
            return '{:.3g}'.format(float(val))
        except:
            return val

    def getStructure(self, structure_tmp, is_primitive, scale):
        sa_structure = SpacegroupAnalyzer(structure_tmp)
        print(sa_structure.get_space_group_symbol())
        if is_primitive:
            structure = sa_structure.get_primitive_standard_structure()
        else:
            structure = sa_structure.get_refined_structure()
            structure.make_supercell([scale, scale, scale])
        return structure

    def get_mpdata_from_composition(self, composition):
        pmg = MPRester("UP0x1rTAXR52g7pi")
        properties = ["task_id", "pretty_formula", "spacegroup.symbol", "formation_energy_per_atom", "e_above_hull", "band_gap"]
        mpdata = pmg.query(criteria=composition, properties=properties)
        df_mpdata = pd.DataFrame(mpdata)
        df_mpdata = df_mpdata.rename(columns={"task_id": "mpid"})
        df_mpdata = df_mpdata.applymap(self.setSigFig)
        df_mpdata = df_mpdata.sort_values("e_above_hull")

        return df_mpdata

    def get_delaunay_from_mpid(self, mpid="mp-19717", scale=1, is_primitive=False):
        pmg = MPRester("UP0x1rTAXR52g7pi")
        structure_tmp = pmg.get_structure_by_material_id(mpid)
        structure = self.getStructure(structure_tmp, is_primitive, scale)

        sites_list = structure.as_dict()["sites"]  # 結晶構造中の各原子の情報
        sites_list_len = len(sites_list)  # 要素の数
        atom_cartesian = []  # 各サイトのデカルト座標
        atom_species = []  # サイトを占有する原子の種類、割合

        for j in range(sites_list_len):  # 結晶構造中の各原子の情報を取得
            atom_cartesian.append(sites_list[j]["xyz"])  # デカルト座標
            atmlabel = sites_list[j]["label"]
            atom_species.append(atmlabel)  # サイトを占有する原子の種類、割合
        tri = Delaunay(atom_cartesian)  # ドロネー分割を行う

        atoms_radius = [mg.Element(el).atomic_radius*10 for el in atom_species]
        atoms_color = [elements[elements["symbol"]==el]["CPK"].values[0] for el in atom_species]
        atom_idx_dict = dict(zip(set(atom_species), range(len(set(atom_species)))))
        atom_idxs = [atom_idx_dict[atmsp] for atmsp in atom_species]

        ijklist = []
        for tet in tri.simplices:
            for comb in itertools.combinations(tet, 3):
                i = comb[0]
                j = comb[1]
                k = comb[2]

                ijklist.append((i, j, k))

        pts = np.array(tri.points)
        x, y, z = pts.T
        ijk = np.array(list(set(ijklist)))
        i, j, k = ijk.T

        fig = go.Figure(data=[go.Mesh3d(x=np.array(x), y=np.array(y), z=np.array(z),
                                color='lightblue',
                                opacity=0.2,
                                        flatshading=True,
                                contour=dict(show=False),
                                i = i,
                                j = j,
                                k = k),
                                go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                            hovertext=atom_species,
                                            marker=dict(
                                                    size=atoms_radius,
                                                    color=atom_idxs,
                                                    colorscale=atoms_color,
                                                    opacity=0.8
                                                )
                                            )]
                    )

        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            autosize=False,
            width=700,
            height=700,
            scene=dict(
                xaxis=dict(showgrid=False, showbackground=True, showticklabels=False, title="x"),
                yaxis=dict(showgrid=False, showbackground=True ,showticklabels=False, title="y"),
                zaxis=dict(showgrid=False, showbackground=True, showticklabels=False, title="z")
            )
        )
        fig.show()
