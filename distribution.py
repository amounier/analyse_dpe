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
from scipy.stats import beta
from scipy.stats import norm
from datetime import date

from administrative import Departement  # on importe la classe
from download import get_bdnb  # on importe la fonction
from utils import etiquette_colors_dict,etiquette_ep_dict,etiquette_ep_seuils


def get_dpe_consumption(dep_code, old_build_filter=False):
    # output_folder =   #mettre un nom de fichier de sauvegarde qui prend en compte les filtres 
    # if save_name in output_folder
    dpe_data, _ , _ = get_bdnb(dep_code)
    dpe_data = dpe_data[dpe_data.type_dpe=='dpe arrêté 2021 3cl logement'][['conso_5_usages_ep_m2','conso_5_usages_ef_m2','periode_construction_dpe']].compute() 
    if old_build_filter:
        dpe_data = dpe_data[dpe_data.periode_construction_dpe.isin(['avant 1948','1948-1974'])]
    dpe_data = dpe_data[['conso_5_usages_ep_m2','conso_5_usages_ef_m2']]
    return dpe_data


def plot_dpe_distribution(path, dep_code, fit=True, save=True, calcul_bunching=True, max_xlim=800, itv_bunching=5):
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
    
    dpe_data = get_dpe_consumption(dep_code) # c'est ca qui prend du temps
    dpe_data = dpe_data.dropna()
    dpe_data = dpe_data.map(round)
    nb_dpe = len(dpe_data) # pour pouvoir normaliser
    counter_dict = dict(dpe_data.conso_5_usages_ep_m2.value_counts())
    counter_dict_sorted = {k: v for k, v in sorted(counter_dict.items(), key=lambda item: item[0])}
    counter_df_sorted = pd.DataFrame(list(counter_dict_sorted.items()), columns=["conso_5_usages_ep_m2_arrondie","nb_observations"]) # df trié par conso_ep (j'avais besoin d'un DataFrame pour le rolling)
    
    print(counter_df_sorted)
    
    fig, ax = plt.subplots(figsize=(5,5), dpi=300)

    for eti in etiquette_colors_dict.keys():
        inf_ep, sup_ep = etiquette_ep_dict.get(eti)
        color = etiquette_colors_dict.get(eti)
        counter_dict_eti = {k:v for k,v in counter_dict_sorted.items() if k > inf_ep and k <= sup_ep}
        ax.bar(list(counter_dict_eti.keys()), list(counter_dict_eti.values()), width=1., color=color, label=eti)
        
    
    

    # tracé des données à fit
    # plt.plot(list(counter_dict_sorted.keys()), list(counter_dict_sorted.values()), "k", label='données', linewidth = 0.5)
    
    # calcul de la moyenne/médiane glissante
    window_size = 50
    rolling_dpe = counter_df_sorted['nb_observations'].rolling(window=window_size, min_periods=1, center=True)
    counter_df_sorted["nb_obs_moyenne"] = rolling_dpe.mean()  # ajout colonne moyenne dans le DataFrame
    counter_df_sorted["nb_obs_mediane"] = rolling_dpe.median()  # ajout colonne mediane dans le DataFrame
    
    #pd.options.display.max_columns = None
    #print('counter_df_sorted pimpé \n', counter_df_sorted)
    
    # tracé de la moyenne/médiane glissante
    plt.plot(counter_df_sorted["conso_5_usages_ep_m2_arrondie"], counter_df_sorted["nb_obs_moyenne"], "k", label='moyenne', linewidth = 0.7)
    plt.plot(counter_df_sorted["conso_5_usages_ep_m2_arrondie"], counter_df_sorted["nb_obs_mediane"], "r", label='mediane', linewidth = 0.7)
    
    ax.set_xlim([0,max_xlim])
    ax.set_ylabel(f"Nombre d'observations ({departement.name} - {departement.code})")
    ax.legend()
    ax.set_xlabel("Consommation annuelle en énergie primaire (kWh.m$^{-2}$)'+f' \n \n Taille de l'échantillon : {nb_dpe} DPE")
    ax.set_xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [max_xlim])
    
    # tracé fit de la moyenne par loi beta
    if fit : 
        #plt.figure()
        #plt.plot(counter_df_sorted["conso_5_usages_ep_m2_arrondie"], (counter_df_sorted["nb_observations"].values)/nb_dpe, "r", label='données normalisées', linewidth = 0.5)
        #plt.plot(counter_df_sorted["conso_5_usages_ep_m2_arrondie"], (counter_df_sorted["nb_obs_moyenne"].values)/nb_dpe, label='distribution moyenne', linewidth = 1)
        a, b, loc, scale = beta.fit(dpe_data["conso_5_usages_ep_m2"]) # beta.fit prend comme argument des datas de type array_like
        print(a, b, loc, scale)
        pdf = beta.pdf(counter_df_sorted["conso_5_usages_ep_m2_arrondie"], a, b, loc=loc, scale=scale)
        #plt.plot(counter_df_sorted["conso_5_usages_ep_m2_arrondie"], pdf, "k", label='beta fit', linewidth = 1)
        #plt.xlim([0,max_xlim])
        
        plt.plot(counter_df_sorted["conso_5_usages_ep_m2_arrondie"], pdf*(nb_dpe), "k--", label='beta fit', linewidth = 1)
        plt.legend()
    
    if save:
        save_path = os.path.join(path,'distribution_dpe_{}.png'.format(dep_code))
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    plt.close()
    
    
    return
    
'''
    # calcul "Average Manipulation Density" (Civel et al.)
 def calcul_bunching: # COMPLETer
     bunching = []
     for seuil in etiquette_ep_seuils:
        nb_droite = sum([int(v) for k,v in counter_dict_sorted.items() if k > seuil and k <= seuil+itv_bunching])
        nb_gauche = sum([int(v) for k,v in counter_dict_sorted.items() if k > seuil-itv_bunching and k <= seuil])
        AMP = (nb_gauche - nb_droite) #/ (nb_gauche+nb_droite)  # average manipulation density 
        bunching.append(AMP)
            
        print(f'Bunching pour {departement}, avec un intervalle de +-{itv_bunching} kWh/m2 autour des seuils : ', bunching)
    return 
    '''
   

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
        dep = Departement(91)
        #dpe_data = get_dpe_consumption(dep.code) # cette ligne ne sert a rien car déjà dans plot_dpe_distribution ?
        plot_dpe_distribution(output_folder,dep.code)
        #calcul_bunching
        
    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')
    
if __name__ == '__main__':
    main()
