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
from scipy.optimize import curve_fit
from datetime import date

from administrative import Departement  # on importe la classe
from download import get_bdnb  # on importe la fonction
from utils import etiquette_colors_dict,etiquette_ep_dict,etiquette_ep_seuils


def get_dpe_consumption(dep_code, old_built_filter=False):
    """
    Formate les données BDNB pour ne garder que les conso d'énergie compilées, et les enregistre en .csv

    Parameters
    ----------
    dep_code : str
        code du département.
    old_built_filter : boolean, optional
        filtrage : on ne garde que les logements construits avant 1974. The default is False.

    Returns
    -------
    dpe_data : panda DataFrame
        base des conso d'énergie du département dep_code (filtrée ou non sur les vieux bâtiments)
    """
    # Définition du chemin de sauvegarde
    output_folder = os.path.join('data','BDNB','dpe_conso')
    os.makedirs(output_folder, exist_ok=True)
    existing_files = os.listdir(output_folder)
    
    # Définition du nom du fichier final
    save_name = f'conso_5_usages_millesime_2025-07_dep{dep_code}.csv'
    if old_built_filter:
        save_name = save_name.replace('.csv','_old_built.csv')
        
    # Enregistrement des conso_5_usages en csv    
    if save_name not in existing_files:
        dpe_data, _ , _ = get_bdnb(dep_code)
        dpe_data = dpe_data[dpe_data.type_dpe=='dpe arrêté 2021 3cl logement'][['conso_5_usages_ep_m2','conso_5_usages_ef_m2','periode_construction_dpe']].compute() 
        if old_built_filter: 
            dpe_data = dpe_data[dpe_data.periode_construction_dpe.isin(['avant 1948','1948-1974'])]
        dpe_data = dpe_data[['conso_5_usages_ep_m2','conso_5_usages_ef_m2']]
        dpe_data.to_csv(os.path.join(output_folder, save_name))
        
    else:
        dpe_data = pd.read_csv(os.path.join(output_folder, save_name))
        
    return dpe_data



def formatage_dpe_data(dep_code, window_size=50):
    """
    Créé un DataFrame de la distribution des données DPE du département dep_code, avec une colonne du nombre d'observation moyen 
    et une colonne du nombre d'observation médian pour chaque valeur de consommation d'énergie primaire

    Parameters
    ----------
    dep_code : str
        code du département.
    window_size : int, optional
        taille de la fenêtre de glissement pour le rolling. The default is 50.

    Returns
    -------
    counter_df_sorted : panda DataFrame
        DataFrame du nombre d'observation exact, moyen glissant et médian glissant de chaque valeur entière de conso_5_usages_ep_m2 dans le département
    """
    
    dpe_data = get_dpe_consumption(dep_code) # c'est ca qui prend du temps
    dpe_data = dpe_data.dropna()
    dpe_data = dpe_data.map(round)
    counter_dict = dict(dpe_data.conso_5_usages_ep_m2.value_counts())
    counter_dict_sorted = {k: v for k, v in sorted(counter_dict.items(), key=lambda item: item[0])}
    counter_df_sorted = pd.DataFrame(list(counter_dict_sorted.items()), columns=["conso_5_usages_ep_m2_arrondie","nb_observations"]) # df trié par conso_ep (besoin d'un DataFrame pour utiliser rolling)
    
    #print(counter_df_sorted)
    
    # tracé des données à fit
    # plt.plot(list(counter_dict_sorted.keys()), list(counter_dict_sorted.values()), "k", label='données', linewidth = 0.5)
    
    # calcul de la moyenne/médiane glissante
    rolling_dpe = counter_df_sorted['nb_observations'].rolling(window=window_size, min_periods=1, center=True)
    counter_df_sorted["nb_obs_moyenne"] = rolling_dpe.mean()  # ajout colonne moyenne dans le DataFrame
    counter_df_sorted["nb_obs_mediane"] = rolling_dpe.median()  # ajout colonne mediane dans le DataFrame
    
    #pd.options.display.max_columns = None
    #print('counter_df_sorted pimpé \n', counter_df_sorted)
    
    return counter_df_sorted



def plot_dpe_distribution(path, dep_code, save=True, plot_mean=True, plot_median=False, window_size=50, plot_fit=True, plot_curve_fit=True, max_xlim=600) :
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
    plot_mean : boolean, optional
        tracé de la moyenne glissante. The default is True. 
    plot_median : boolean, optional
        tracé de la médiane glissante. The default is True.
    window_size : int
        taille de la fenêtre de glissement pour le rolling. 
    plot_fit : boolean, optional
        tracé du fit des données. The default is True.
    max_xlim : int, optional
        limite du graphe. The default is 600.

    Returns
    -------
    None
    """
    departement = Departement(dep_code)
    
    dpe_data = get_dpe_consumption(dep_code) # prend du temps
    dpe_data = dpe_data.dropna()
    dpe_data = dpe_data.map(round)
    nb_dpe = len(dpe_data) # pour pouvoir normaliser
    counter_dict = dict(dpe_data.conso_5_usages_ep_m2.value_counts())
    counter_dict_sorted = {k: v for k, v in sorted(counter_dict.items(), key=lambda item: item[0])}
    
    
    fig, ax = plt.subplots(figsize=(5,5), dpi=300)

    # Tracé de l'histogramme des DPEs
    for eti in etiquette_colors_dict.keys():
        inf_ep, sup_ep = etiquette_ep_dict.get(eti)
        color = etiquette_colors_dict.get(eti)
        counter_dict_eti = {k:v for k,v in counter_dict_sorted.items() if k > inf_ep and k <= sup_ep}
        ax.bar(list(counter_dict_eti.keys()), list(counter_dict_eti.values()), width=1., color=color, label=eti)
        
    ax.set_xlim([0,max_xlim])
    fig.suptitle(f"Distribution des DPE ({departement.name} - {departement.code})")
    ax.set_title(f"Taille de l'échantillon : {nb_dpe} DPE", fontsize=10)
    fig.subplots_adjust(top=0.9)
    ax.set_ylabel("Nombre d'observations")
    ax.set_xlabel("Consommation annuelle en énergie primaire (kWh.m$^{-2}$)")
    ax.set_xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [max_xlim])
    
    
    # Tracé de la moyenne/médiane glissante 
    if plot_mean:
        plt.plot(formatage_dpe_data(dep_code, window_size)["conso_5_usages_ep_m2_arrondie"], formatage_dpe_data(dep_code, window_size)["nb_obs_moyenne"], "k", label='moyenne', linewidth = 0.7)
        
    if plot_median:
        plt.plot(formatage_dpe_data(dep_code, window_size)["conso_5_usages_ep_m2_arrondie"], formatage_dpe_data(dep_code, window_size)["nb_obs_mediane"], "r", label='mediane', linewidth = 0.7)
    
    
    # tracé fit des données avec une loi beta
    if plot_fit: 
        a, b, loc, scale = beta.fit(dpe_data["conso_5_usages_ep_m2"]) # beta.fit prend comme argument des datas de type array_like
        print('beta fit', a, b, loc, scale)
        pdf = beta.pdf(formatage_dpe_data(dep_code, window_size)["conso_5_usages_ep_m2_arrondie"], a, b, loc=loc, scale=scale) # probability density function
        #plt.plot(counter_df_sorted["conso_5_usages_ep_m2_arrondie"], pdf, "k", label='beta fit', linewidth = 1)
        
        plt.plot(formatage_dpe_data(dep_code, window_size)["conso_5_usages_ep_m2_arrondie"], pdf*(nb_dpe), "k--", label='beta fit', linewidth = 1)
        plt.legend()
        
        
    if plot_curve_fit: 
        x_data = np.array(list(counter_dict_sorted.keys()))
        y_data = np.array(list(counter_dict_sorted.values()))/nb_dpe
        x_min, x_max = x_data.min(), x_data.max()
        x_normalized = (x_data - x_min) / (x_max - x_min)
        
        #print(x_normalized)
        
        plt.plot(list(counter_dict_sorted.keys()), list(counter_dict_sorted.values()), "b", label='données', linewidth = 0.7)
        
        def beta_pdf(x, a, b):
            return np.power(x,a-1) * np.power(1-x,b-1)

        param, cov = curve_fit(beta_pdf, x_normalized, y_data)
        a, b = param
        print('Valeurs des paramètres de la loi beta :', a, b, cov)
        #plt.figure()
        #plt.plot(x_normalized, y_data)
        #plt.plot(x_normalized, beta_pdf(x_normalized, a, b), '--', label='curve_fit')
        pdf = beta_pdf(x_normalized, a, b)
        plt.plot(x_data, pdf*(nb_dpe), label='curve_fit')
        #plt.xlim([0,0.4])
    
    if save:
        save_path = os.path.join(path,'distribution_dpe_{}.png'.format(dep_code))
        if plot_fit:
            save_path = os.path.join(path,'distribution_dpe_{}_fit.png'.format(dep_code))
        plt.savefig(save_path, bbox_inches='tight')


    ax.legend()
    plt.show()
    plt.close()
    
    return
    


def calcul_bunching(dep_code, method, itv_bunching=5):
    """
    Calcul du bunching avec plusieurs méthodes possibles.
    
    Parameters
    ----------
    method : str ('AMP' ou 'diff_beta')
        Nom de la méthode utilisée pour calculer le bunching.
    itv_bunching : int, optional ? XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        Taille de l'intervalle en-dessous et au-dessus des seuils sur lequel on calcule le bunching. The default is 5 kWh/m2 
       
    Returns
    -------
    bunching : panda DataFrame ? Dictionnaire ? choisir XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        Bunching pour chaque seuil : A/B, B/C, C/D, D/E, E/F et F/G.
    """
    
    departement = Departement(dep_code)
    
    dpe_data = get_dpe_consumption(dep_code)
    dpe_data = dpe_data.dropna()
    dpe_data = dpe_data.map(round)
    counter_dict = dict(dpe_data.conso_5_usages_ep_m2.value_counts())
    counter_dict_sorted = {k: v for k, v in sorted(counter_dict.items(), key=lambda item: item[0])}
    
    if method=='AMP': # méthode "Average Manipulation Density" (Civel et al.)
         bunching = []
         for seuil in etiquette_ep_seuils:
            nb_droite = sum([int(v) for k,v in counter_dict_sorted.items() if k > seuil and k <= seuil+itv_bunching])
            nb_gauche = sum([int(v) for k,v in counter_dict_sorted.items() if k > seuil-itv_bunching and k <= seuil])
            AMP = (nb_gauche - nb_droite) / (nb_gauche+nb_droite)  # average manipulation density 
            bunching.append(AMP)
            
    if method=='diff_beta': # différence d'aire sous la courbe entre les données réelles et le beta fit sur toutes les données
        bunching = []
        #plt.plot(formatage_dpe_data(dep_code, window_size)["conso_5_usages_ep_m2_arrondie"], formatage_dpe_data(dep_code, window_size)["nb_observations"]-BETA FIT)
        # completer
            
    print(f'Bunching pour {departement}, avec un intervalle de +-{itv_bunching} kWh/m2 autour des seuils : ', bunching)
    
    return bunching

   

#%% ===========================================================================
# script principal
# =============================================================================

def main():
    tic = time.time()
    
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')
    output_folder = os.path.join('output',today)
    os.makedirs(output_folder, exist_ok=True)
    
    
    #test
    if False: 
        a=2
        b=5
        
        x = np.linspace(0, 2, num=40)
        f = np.power(x,a-1) * np.power(1-x,b-1)
        
        plt.plot(x,f)
    
    # tracé de la distribution des dpe du département
    if True:
        dep = Departement(91)
        #dpe_data = get_dpe_consumption(dep.code) # cette ligne ne sert a rien car déjà dans plot_dpe_distribution ?
        plot_dpe_distribution(output_folder,dep.code)
        
    # calcul du bunching
    if False:
        calcul_bunching(dep.code, method='AMP', itv_bunching=5)
        
    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')
    
if __name__ == '__main__':
    main()
