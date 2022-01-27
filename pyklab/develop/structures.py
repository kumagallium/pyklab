import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from scipy.spatial import Delaunay
import pymatgen.core as mg
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import IsayevNN, MinimumDistanceNN, CrystalNN, CutOffDictNN, EconNN, JmolNN, MinimumOKeeffeNN, MinimumVIRENN, VoronoiNN
import itertools
from itertools import product
from bokeh.sampledata.periodic_table import elements
import networkx as nx
import matplotlib.pyplot as plt

pmg = MPRester("UP0x1rTAXR52g7pi")
elcolor = dict(zip(elements["atomic number"].values, elements["CPK"].values))


class Structure():

    def __init__(self, structure_dirpath="structures/"):
        self.structure_dirpath = structure_dirpath

    def set_sig_fig(self, val):
        try:
            return '{:.3g}'.format(float(val))
        except:
            return val

    def get_structure(self, mpid, is_primitive, scale, structure=""):
        if structure == "":
            structure_tmp = pmg.get_structure_by_material_id(mpid)
        else:
            structure_tmp = structure

        sa_structure = SpacegroupAnalyzer(structure_tmp)
        #print(sa_structure.get_space_group_symbol())
        if is_primitive:
            #structure = structure.get_primitive_structure()
            structure = sa_structure.get_primitive_standard_structure()
            structure.make_supercell([scale, scale, scale])
        else:
            #structure = structure.get_primitive_structure().get_reduced_structure()
            structure = sa_structure.get_refined_structure()
            structure.make_supercell([scale, scale, scale])
        return structure

    def get_mpdata_from_composition(self, composition):
        properties = ["task_id", "pretty_formula", "spacegroup.symbol", "formation_energy_per_atom", "e_above_hull", "band_gap"]
        mpdata = pmg.query(criteria=composition, properties=properties)
        df_mpdata = pd.DataFrame(mpdata)
        df_mpdata = df_mpdata.rename(columns={"task_id": "mpid"})
        df_mpdata = df_mpdata.applymap(self.set_sig_fig)
        df_mpdata = df_mpdata.sort_values("e_above_hull")

        return df_mpdata

    def my_round(self, val, digit=2):
        p = 10 ** digit
        return (val * p * 2 + 1) // 2 / p

    def get_round(self, arr,digit=2):
        res = np.array([self.my_round(val,digit) for val in arr])
        return res

    def get_delaunay(self, mpid="mp-19717", scale=1, is_primitive=False, structure=""):
        structure_tmp = self.get_structure(mpid, is_primitive, scale, structure=structure)

        structure_tmp.make_supercell([3, 3, 3])
        xyz_list = [site["xyz"] for site in structure_tmp.as_dict()["sites"]]  # Information on each site in the crystal structure
        label_list = [site["label"] for site in structure_tmp.as_dict()["sites"]] 
        matrix = structure_tmp.lattice.matrix
        a, b, c = self.get_round(structure_tmp.lattice.abc)

        tri = Delaunay(xyz_list)

        simplices_all = tri.simplices
        points_all = tri.points

        include_idxs = []
        for i, point in enumerate(points_all):
            abc_mat = self.get_round(structure_tmp.lattice.get_vector_along_lattice_directions(point))
            if (abc_mat[0]>=(a/3)) and (abc_mat[1]>=(b/3)) and (abc_mat[2]>=(c/3)) and (abc_mat[0]<=(a*2/3)) and (abc_mat[1]<=(b*2/3)) and (abc_mat[2]<=(c*2/3)):
                include_idxs.append(i)
        
        ijklist = []
        pidxs = []
        for tet in simplices_all:
            if len(set(tet)&set(include_idxs)) > 0:
                for comb in itertools.combinations(tet, 3):
                    comb = np.sort(comb)
                    i = comb[0]
                    j = comb[1]
                    k = comb[2]

                    ijklist.append((i, j, k))  
                    pidxs.extend((i, j, k))
                    pidxs = list(set(pidxs))
        
        atom_idx_dict = dict(zip(set(np.array(label_list)), range(len(set(np.array(label_list))))))
        viz_points = []
        atoms_radius = []
        atoms_color = []
        atom_idxs = []
        atom_species = []
        pidx_dict = {}
        for i, pidx in enumerate(np.sort(pidxs)):
            viz_points.append(points_all[pidx])
            if mg.Element(label_list[pidx]).atomic_radius != None:
                atoms_radius.append(mg.Element(label_list[pidx]).atomic_radius*(10/scale))
            else:
                atoms_radius.append(10/scale)
            atoms_color.append(elements[elements["symbol"]==label_list[pidx]]["CPK"].values[0])
            atom_idxs.append(atom_idx_dict[label_list[pidx]])
            atom_species.append(label_list[pidx])
            pidx_dict[pidx] = i
        
        viz_ijk = []
        for ijk in ijklist:
            ijk_tmp = []
            for tmp in ijk:
                ijk_tmp.append(pidx_dict[tmp])
            viz_ijk.append(tuple(ijk_tmp))

        pts = np.array(viz_points)
        ijk = np.array(list(set(viz_ijk)))

        return {"pts": pts, "ijk": ijk, "matrix":matrix, "atom_species": atom_species, "atoms_radius": atoms_radius, "atoms_color": atoms_color, "atom_idxs": atom_idxs}

    def show_delaunay(self, mpid="mp-19717", scale=1, is_primitive=False, structure=""):
        pts, ijk, matrix, atom_species, atoms_radius, atoms_color, atom_idxs = self.get_delaunay(mpid=mpid, scale=scale, is_primitive=is_primitive, structure=structure).values()

        x, y, z = pts.T
        i, j, k = ijk.T
        print(len(pts))
        print(len(ijk))

        xyz = list(product([1/3,2/3], repeat=3))
        xyz = [np.dot(np.array(xyz_tmp),matrix) for xyz_tmp in xyz]

        xx,yy,zz = np.array([xyz[0],xyz[1],xyz[3],xyz[2],xyz[0]
                    ,xyz[4],xyz[5],xyz[1]
                    ,xyz[3],xyz[7],xyz[5]
                    ,xyz[5],xyz[7],xyz[6],xyz[4]
                    ,xyz[6],xyz[2]]).T
        fig = go.Figure(data=[go.Mesh3d(x=np.array(x), y=np.array(y), z=np.array(z),
                                color='lightblue',
                                opacity=0.2,
                                        flatshading=True,
                                contour=dict(show=False),
                                hoverinfo="text",
                                i = i,
                                j = j,
                                k = k),
                                go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                            #hoverinfo="text",
                                            #hovertext=atom_species,
                                            marker=dict(
                                                    size=atoms_radius,
                                                    color=atoms_color,
                                                    opacity=0.8
                                                )
                                            ),
                        go.Scatter3d(x=xx,
                                    y=yy,
                                    z=zz,
                                    #hoverinfo="text",
                                    mode='lines',
                                    name='',
                                    line=dict(color= 'rgb(70,70,70)', width=2))]
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
        fig.update_scenes(camera_projection=dict(type="orthographic"))
        fig.show()

    def get_delaunay_to_offformat(self, mpid="mp-19717", scale=1, is_primitive=False, structure="", nodes=False):
        pts, ijks, _, atom_species, _, _, _ = self.get_delaunay(mpid=mpid, scale=scale, is_primitive=is_primitive, structure=structure).values()

        offtext = "OFF\n"
        offtext += str(len(pts)) + " " + str(len(ijks)) + " 0\n"
        for pt in pts:
            offtext += " ".join(map(str, pt)) + "\n"
        for ijk in ijks:
            offtext += str(len(ijk)) + " " + " ".join(map(str, ijk)) + "\n"

        if nodes:
            offtext += "\n".join(map(str, atom_species))

        if not os.path.exists(self.structure_dirpath):
            os.mkdir(self.structure_dirpath)
        if not os.path.exists(self.structure_dirpath+"mp_delaunay_offformat/"):
            os.mkdir(self.structure_dirpath+"mp_delaunay_offformat/")
        # Refactor: Add naming process when mpid is missing
        with open(self.structure_dirpath+"mp_delaunay_offformat/"+mpid+".off", mode='w') as f:
            f.write(offtext)

        return offtext


    def get_delaunay_to_objformat(self, mpid="mp-19717", scale=1, is_primitive=False, structure=""):
        pts, ijks, _, atom_species, _, _, _ = self.get_delaunay(mpid=mpid, scale=scale, is_primitive=is_primitive, structure=structure).values()

        objtext ="####\n#\n# OBJ File Generated by Pyklab\n#\n####\n"+ \
                 "# Object "+mpid+".obj\n"+ \
                 "#\n# Vertices: "+str(len(pts))+"\n"+ \
                 "# Faces: "+str(len(ijks))+"\n#\n####\n"
        for pt in pts:
            objtext += "v " + " ".join(map(str, pt)) + "\n"
        objtext += "\n"
        for ijk in ijks:
            ijk = map(lambda x: x+1, ijk)
            objtext += "f " + " ".join(map(str, ijk)) + "\n"
        objtext += "\n# End of File"

        if not os.path.exists(self.structure_dirpath):
            os.mkdir(self.structure_dirpath)
        if not os.path.exists(self.structure_dirpath+"mp_delaunay_objformat/"):
            os.mkdir(self.structure_dirpath+"mp_delaunay_objformat/")
        # Refactor: Add naming process when mpid is missing
        with open(self.structure_dirpath+"mp_delaunay_objformat/"+mpid+".obj", mode='w') as f:
            f.write(objtext)

        return objtext

    def create_crystal_graph(self, structure, graphtype="IsayevNN"):
        #https://pymatgen.org/pymatgen.analysis.local_env.html
        #IsayevNN: https://www.nature.com/articles/ncomms15679.pdf
        if graphtype == "IsayevNN":
            nn = IsayevNN(cutoff=6, allow_pathological=True)
        elif graphtype == "MinimumDistanceNN":
            nn = MinimumDistanceNN(cutoff=5)
        elif graphtype == "CrystalNN":
            nn = CrystalNN()
        elif graphtype == "CutOffDictNN":
            nn = CutOffDictNN()
        elif graphtype == "EconNN":
            nn = EconNN()
        elif graphtype == "JmolNN":
            nn = JmolNN()
        elif graphtype == "MinimumOKeeffeNN":
            nn = MinimumOKeeffeNN()
        elif graphtype == "MinimumVIRENN":
            nn = MinimumVIRENN()
        elif graphtype == "VoronoiNN":
            nn = VoronoiNN()

        originalsites = {}
        originalsites_inv = {}
        for site in structure.sites:
            originalsites[site] = nn._get_original_site(structure, site)
            originalsites_inv[nn._get_original_site(structure, site)] = site

        nodes = {}  # ノードの初期化
        edges = {}  # エッジの初期化
        adj = []  # 隣接行列の初期化
        weights = []  # 重み行列の初期化
        distances = []  # 原子間距離行列の初期化
        # 元の各サイト
        for i, basesite in enumerate(nn.get_all_nn_info(structure)):
            orisite1 = originalsites_inv[i]
            nodes[i] = orisite1.as_dict()["species"][0]["element"]
            sitenum = structure.num_sites   # 元の結晶構造のサイト数
            adj.append([0]*sitenum)  # 隣接行列の初期化
            weights.append([0]*sitenum)  # 重み行列の初期化
            distances.append([0]*sitenum)  # 原子間距離行列の初期化
            # uniquesite = []
            # 各隣接サイト
            for neighbor in basesite:
                # 隣接サイトと同一の元サイトの探索
                # for orisite2  in list(originalsites.keys()):
                for orisite2 in list(originalsites.keys())[i+1:]:
                    # https://pymatgen.org/pymatgen.core.sites.html
                    # 同一サイトであるか判定
                    if neighbor["site"].is_periodic_image(orisite2):
                        adj[i][originalsites[orisite2]] += 1
                        weights[i][originalsites[orisite2]] += neighbor["weight"]
                        distances[i][originalsites[orisite2]] += orisite1.distance(neighbor["site"])
                        edges.setdefault(i, [])
                        edges[i].append(originalsites[orisite2])
                        break

        return nodes, edges, adj, weights, distances

    def view_graph(self, graph, node2atom):
        g_nodes = [mg.Composition(node).elements[0].symbol for node in graph.nodes]
        pos = nx.spring_layout(graph)  # ,k=10)
        if len(graph.edges) > 0:
            edge_labels = {}
            u, v, d = np.array(list(graph.edges(data=True))).T
            sites = list(zip(u, v))
            for st in sites:
                edge_labels.setdefault(st, 0)
                edge_labels[st] += 1
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=5)
        else:
            print("No edges")

        nx.draw_networkx(graph, pos, font_size=5, width=0.5, node_color=[elcolor[node2atom[node]] for node in g_nodes],
                            node_size=[mg.Element(node).atomic_radius*100 for node in g_nodes])

    def visualize_crystal_graph(self, nodes, edges, distances):
        G = nx.MultiGraph()
        node2atom = {}
        atomcount = {}
        renamenodes = {}
        for siteidx, el in nodes.items():
            atomcount.setdefault(el, 0)
            atomcount[el] += 1
            renamenodes[siteidx] = el + str(atomcount[el])
            G.add_node(renamenodes[siteidx])
            node2atom[el] = mg.Element(el).number

        for siteidx, edge in edges.items():
            for i, e in enumerate(edge):
                G.add_edge(renamenodes[siteidx], renamenodes[e], length=distances[siteidx][e])

        fig = plt.figure(figsize=(3, 3), dpi=300, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        # Remove axis ticks
        ax.tick_params(labelbottom="off", bottom="off")
        ax.tick_params(labelleft="off", left="off")
        # Remove labels
        ax.set_xticklabels([])
        # Remove axis
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.grid(False)

        self.view_graph(G, node2atom)
        plt.show()
