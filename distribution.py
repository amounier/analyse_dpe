#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 23:33:02 2026

@author: amounier
"""

import time
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

from administrative import Departement
from download import get_bdnb
from utils import etiquette_colors_dict,etiquette_ep_dict


def get_dpe_consumption(dep_code):
    dpe_data, _ , _ = get_bdnb(dep_code)
    dpe_data = dpe_data[dpe_data.type_dpe=='dpe arrêté 2021 3cl logement'][['conso_5_usages_ep_m2','conso_5_usages_ef_m2']].compute() 
    return dpe_data


def plot_dpe_distribution(path, dep_code, save=True, max_xlim=600):
    """
    graphe de la distribution des DPE, en indiquant les limites entre catégories.

    Parameters
    ----------
    path : str
        chemin de sauvegarde.
    dep_code : str
        code du departement.
    save : boolean, optional
        sauvegarde. The default is True.
    max_xlim : int, optional
        limite du graphe. The default is 600.

    Returns
    -------
    None
    """
    departement = Departement(dep_code)
    
    dpe_data = get_dpe_consumption(dep_code)
    dpe_data = dpe_data.dropna()
    dpe_data = dpe_data.map(int)
    counter_dict = dict(dpe_data.conso_5_usages_ep_m2.value_counts())
    counter_dict_sorted = {k: v for k, v in sorted(counter_dict.items(), key=lambda item: item[0])}
    
    
    fig, ax = plt.subplots(figsize=(5,5), dpi=300,)
    for eti in etiquette_colors_dict.keys():
        inf_ep, sup_ep = etiquette_ep_dict.get(eti)
        color = etiquette_colors_dict.get(eti)
        counter_dict_eti = {k:v for k,v in counter_dict_sorted.items() if k > inf_ep and k <= sup_ep}
        ax.bar(list(counter_dict_eti.keys()), list(counter_dict_eti.values()), width=1., color=color, label=eti)
    
    ax.set_xlim([0,max_xlim])
    ax.set_ylabel(f"Nombre d'observations ({departement.name} - {departement.code})")
    ax.legend()
    ax.set_xlabel("Consommation annuelle en énergie primaire (kWh.m$^{-2}$)")
    ax.set_xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [max_xlim])
    if save:
        save_path = os.path.join(path,'distribution_dpe_{}.png'.format(dep_code))
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    plt.close()
    return 

#%% ===========================================================================
# script principal
# =============================================================================

def main():
    tic = time.time()
    
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')
    output_folder = os.path.join('output',today)
    os.makedirs(output_folder, exist_ok=True)
    
    # test des consommations des dpe
    if True:
        dep = Departement(24)
        dpe_data = get_dpe_consumption(dep.code)
        plot_dpe_distribution(output_folder,dep.code)
        
    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')
    
if __name__ == '__main__':
    main()
