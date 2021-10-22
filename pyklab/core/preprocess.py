
import pymatgen.core as mg
import numpy as np
import pandas as pd


class Preprocess:
    def get_dense(self, df, columns=[]):
        return df[columns].dropna().reset_index(drop=True)

    def set_reduce_formula(self, composition):
        try:
            response = mg.Composition(composition).fractional_composition.get_integer_formula_and_factor()[0]
            return response
        except:
            return np.nan

    def getyear(self, published):
        if type(published) == str:
            return int(published.split("-")[0])
        else:
            return np.nan 
