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
from scipy.stats import beta, zscore # attention, beta (variable aléatoire) =/= sc.beta (fonction)
from scipy.optimize import curve_fit
import scipy.special as sc 
from datetime import date

from administrative import Departement, France  
from download import get_bdnb
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



def fit_dpe_data(dep_code, method):
    """
    Fit de la distribution des DPE avec plusieurs méthodes possibles.

    Parameters
    ----------
    dep_code : str
        code du département.
    method : str, optional ('beta.fit' ou 'curve_fit' ou 'curve_fit_mean')
        Nom de la méthode utilisée pour "fitter". The default is 'beta.fit'.

    Returns
    -------
    pdf : array
        Fonction de densité de probabilité des consommations d'énergie primaire.
    """
    
    dpe_data = get_dpe_consumption(dep_code) # prend du temps
    dpe_data = dpe_data.dropna()
    dpe_data = dpe_data.map(round)
    counter_dict = dict(dpe_data.conso_5_usages_ep_m2.value_counts())
    counter_dict_sorted = {k: v for k, v in sorted(counter_dict.items(), key=lambda item: item[0])}
    
    # methode beta.fit a ne pas utiliser a priori
    if method=='beta.fit':
        a, b, loc, scale = beta.fit(dpe_data["conso_5_usages_ep_m2"]) # beta.fit prend comme argument des datas de type array_like
        print('Paramètres du beta fit (a, b, loc, scale)', a, b, loc, scale)
        pdf = beta.pdf(list(counter_dict_sorted.keys()), a, b, loc=loc, scale=scale) # probability density function
        
        
    # methode à privilégier
    if method=='curve_fit':
        x_data = np.array(list(counter_dict_sorted.keys()))  # on a array of int et pas of float
        filtre = zscore(x_data)<3
        # filtre = x_data<1000
        
        x_data = x_data[filtre]
        print(x_data.max())
        
        y_data_norm = np.array(list(counter_dict_sorted.values()))/nb_dpe
        y_data_norm = y_data_norm[filtre]
        y_data_norm = y_data_norm/y_data_norm.sum()
        #XXXX 
        #pdf = 
        
        
    return pdf



def plot_dpe_distribution(path, dep_code, save=True, plot_mean=True, plot_median=True, window_size=50, plot_fit=True, plot_curve_fit=True, max_xlim=600) :
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
    
    
    # Tracé fit de toutes les données dpe_data["conso_5_usages_ep_m2"] avec une loi beta : fonction scipy.beta.fit 
    if plot_fit: 
        pdf = fit_dpe_data(dep_code, method='beta.fit')
        plt.plot(list(counter_dict_sorted.keys()), pdf*nb_dpe, "k--", label='beta fit', linewidth = 1)
            
    # Tracé fit de toutes les données : fonction curve_fit avec un modèle de loi beta    
    if plot_curve_fit: 
        x_data = np.array(list(counter_dict_sorted.keys()))  # on a array of int et pas of float
        
        # Filtrage des valeurs extrêmes de consommations d'énergie primaire
        filtre = zscore(x_data)<3 
        
        x_data = x_data[filtre]
        print(f'Le curve_fit ne prend pas en compte les DPE supérieurs à {x_data.max()} kWh/m2 (Z score > 3)')
        
        y_data_norm = np.array(list(counter_dict_sorted.values()))/nb_dpe
        y_data_norm = y_data_norm[filtre]
        y_data_norm = y_data_norm/y_data_norm.sum() # afin que l'aire sous la courbe soit bien =1
        
        
        '''
        # Fonction qui ne marche pas jsp pourquoi :
        
        def beta_pdf(x, a, b):
            x_min, x_max = x_data.min(), x_data.max()
            x_norm = (x - x_min) / (x_max - x_min)
            #print(x_norm)
            return np.power(x_norm,a-1) * np.power(1-x_norm,b-1) / (sc.beta(a, b))
        
        # en enlevant loc :
        
        first_guess = (4, 4)
        param, cov = curve_fit(beta_pdf, x_data, y_data_norm, first_guess, method='trf', bounds=(0, +np.inf))  # la méthode trf fonctionne bien 
        a, b = param
        print('Paramètres de la loi beta (a, b, cov) :', a, b, cov)
        
        
        pdf = beta_pdf(x_data, a, b)
        
        '''
        
        scale = x_data.max() # TODO: a inclure dans beta_pdf sous forme scale = x.max() ?
        
        
        def beta_pdf(x, a, b, loc):
            x_norm = (x - loc) / scale
            x_norm = x_norm.clip(min=0,max=None)
            res = np.power(x_norm,(a-1)) * np.power((1-x_norm),(b-1)) /sc.beta(a,b)/scale
            return res
    
        
        first_guess = (4, 4, 0)
        param, cov = curve_fit(beta_pdf, x_data, y_data_norm, first_guess, method='trf', bounds=(0, +np.inf))  # la méthode trf fonctionne bien 
        a, b, loc = param
        print('Paramètres de la loi beta (a, b, loc, cov) :', a, b, loc, cov)

        pdf = beta_pdf(x_data, a, b, loc)


        plt.plot(x_data, pdf*(nb_dpe), label='curve_fit')



    if save:
        save_path = os.path.join(path,'distribution_dpe_{}.png'.format(dep_code))
        if plot_fit:
            save_path = os.path.join(path,'distribution_dpe_{}_fit.png'.format(dep_code))
        # if old_built_filter:
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        plt.savefig(save_path, bbox_inches='tight')


    ax.legend()
    plt.show()
    plt.close()
    
    return
    



def calcul_bunching(path, dep_code, method, itv_bunching, max_xlim = 600):
    """
    Calcul du bunching avec plusieurs méthodes possibles.
    
    Parameters
    ----------
    method : str ('AMP' ou 'diff_beta')
        Nom de la méthode utilisée pour calculer le bunching.
    itv_bunching : int XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        Attention : l'intervalle peut être soit à gauche du seuil, soit de part et d'autres du seuil selon les méthodes. The default is 5 kWh/m2 
        Methode 'AMP' : taille de l'intervalle en-dessous et au-dessus des seuils sur lequel on calcule le bunching. 
        Methode 'diff_beta' : taille de l'intervalle à gauche de chaque seuil (utiliser plutôt 10 kWh/m2)
    Returns
    -------
    bunching : panda DataFrame ? Dictionnaire ? choisir XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        Bunching pour chaque seuil : A/B, B/C, C/D, D/E, E/F et F/G.
    """
    
    departement = Departement(dep_code)
    
    dpe_data = get_dpe_consumption(dep_code)
    dpe_data = dpe_data.dropna()
    dpe_data = dpe_data.map(round)
    nb_dpe = len(dpe_data)
    counter_dict = dict(dpe_data.conso_5_usages_ep_m2.value_counts())
    counter_dict_sorted = {k: v for k, v in sorted(counter_dict.items(), key=lambda item: item[0])}
    
    bunching = {"A/B": 0., "B/C": 0., "C/D": 0., "D/E": 0., "E/F": 0., "F/G": 0.}
    
    if method=='AMP': # méthode "Average Manipulation Density" (Civel et al.). Utilise itv_bunching
        
        for k, seuil in etiquette_ep_seuils.items():
            nb_droite = sum([int(v) for k,v in counter_dict_sorted.items() if k > seuil and k <= seuil+itv_bunching])
            nb_gauche = sum([int(v) for k,v in counter_dict_sorted.items() if k > seuil-itv_bunching and k <= seuil])
            AMP = (nb_gauche - nb_droite) / (nb_gauche+nb_droite)  # average manipulation density # on pourrait aussi diviser par nb tot DPE (AJa et al) ? 
            AMP = round(AMP,3)
            bunching[k] = AMP
            
        print(f'Bunching (méthode AMP) pour {departement}, avec un intervalle de +-{itv_bunching} kWh/m2 autour des seuils : ', bunching)

            
    if method=='diff_beta': # différence d'aire sous la courbe entre les données réelles et le beta.fit sur toutes les données. Plot et enregistre la figure
        pdf = fit_dpe_data(dep_code, method='beta.fit')
        difference = list(counter_dict_sorted.values()) - pdf*nb_dpe
        difference_dict =  dict(zip(counter_dict_sorted.keys(), difference)) # dictionnaire qui lie l'écart au beta.fit des DPE à leur conso annuelle d'ep associée
        
        # Calcul du bunching
        # méthode part excessive standardisée (Aja et al.) -> "part des DPE qui sont excessifs sur l’intervalle de 10 kWh de consommation d’énergie à gauche de chaque seuil"
        for k, seuil in etiquette_ep_seuils.items():
            part_excess = sum([float(v) for k,v in difference_dict.items() if k > seuil-itv_bunching and k <= seuil])  # omme des DPEs excessifs à gauche du seuil
            part_excess = part_excess/nb_dpe   # normalisation pour obtenir la proportion des DPEs qui seraient excessifs
            part_excess = round(part_excess,3)
            bunching[k] = part_excess
        
        print(f"Bunching (méthode part excessive) pour {departement}, sur l'intervalle de {itv_bunching} kWh/m2 à gauche de chaque seuil : ", bunching)
        

        # Tracé de l'écart entre les données réelles et le beta.fit sur toutes les données
        plt.figure()
        plt.plot(list(counter_dict_sorted.keys()), difference, linewidth = 0.7)
        
        plt.xlim([0,max_xlim])
        plt.hlines(y=0, xmin=0, xmax=max_xlim, color='k', linestyles='dashed') # tracé de l'axe y=0
        plt.title(f"Ecart au beta.fit des DPE ({departement.name} - {departement.code})")
        plt.ylabel("Nombre de DPE de différence")
        plt.xlabel("Consommation annuelle en énergie primaire (kWh.m$^{-2}$)")
        plt.xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [max_xlim])
        
        # Enregistrement de la figure
        save_path = os.path.join(path,'ecart_beta.fit_{}.png'.format(dep_code))
        plt.savefig(save_path, bbox_inches='tight')

    
    
    return bunching

   

#%% ===========================================================================
# script principal
# =============================================================================

def main():
    tic = time.time()
    
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')
    output_folder = os.path.join('output',today)
    os.makedirs(output_folder, exist_ok=True)
    
    dep = Departement(91)
    
    #test distrib beta
    if False: 
        a=2
        b=5
        
        x = np.linspace(0, 2, num=40)
        f = np.power(x,a-1) * np.power(1-x,b-1) / sc.beta(a,b)
        
        plt.plot(x,f)
    
    # tracé de la distribution des dpe du département
    if True:
        #dpe_data = get_dpe_consumption(dep.code) # cette ligne ne sert a rien car déjà dans plot_dpe_distribution ?
        plot_dpe_distribution(output_folder,dep.code, plot_mean=True, plot_median=True, plot_fit=True, plot_curve_fit=True)
        
    # calcul du bunching
    if False:
        #calcul_bunching(output_folder, dep.code, method='AMP', itv_bunching=5)
        calcul_bunching(output_folder, dep.code, method='diff_beta', itv_bunching=10)
     
        
    # Calcul bunching pour tous les départements
    if False:
        france = France()
        dict_dep_bunching = {d:0. for d in France().departements} # initialisation du dictionnaire du bunching des départements
        
        for dep in france.departements[:10] :
            dep_code = dep.code
            print(dep_code)
            bunching_dep = calcul_bunching(output_folder, dep_code, method='diff_beta', itv_bunching=10, max_xlim = 600)  # calcul du bunching : choix de la méthode et de ses paramètres
            bunching_dep_sum = sum([b for k,b in bunching_dep.items() if k in ...])  # on additionne les bunchings des 6 seuils
            dict_dep_bunching['dep'] = bunching_dep_sum
            
        print(dict_dep_bunching)
        
    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')
    
if __name__ == '__main__':
    main()
