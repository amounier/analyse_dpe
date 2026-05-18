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
from sklearn.metrics import r2_score

from administrative import Departement, France, draw_departement_map  
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



def fit_dpe_data(dep_code, method='curve_fit', verbose=True):
    """
    Fit de la distribution des DPE du département dep_code avec plusieurs méthodes possibles. 

    Parameters
    ----------
    dep_code : str
        code du département.
    method : str, optional ('beta.fit' ou 'curve_fit' ou 'curve_fit_mean')
        Nom de la méthode utilisée pour "fitter". The default is 'curve_fit'.

    Returns
    -------
    fit_dpe_data_df : pandas DataFrame
        Fit de la distribution du département = densité de probabilité des consommations d'énergie primaire dans le dep
        Colonnes : x_data (filtré tq zscore<3)  |  y_data_norm (distribution normalisée)  |  y_beta_curve_fit (fit de y_data_norm)  # todo: modifier colonne si ajout méthode curve_fit sur la moyenne 
    """
    
    dpe_data = get_dpe_consumption(dep_code) # prend du temps
    dpe_data = dpe_data.dropna()
    dpe_data = dpe_data.map(round)
    nb_dpe = len(dpe_data)
    counter_dict = dict(dpe_data.conso_5_usages_ep_m2.value_counts())
    counter_dict_sorted = {k: v for k, v in sorted(counter_dict.items(), key=lambda item: item[0])}
    
    # OLD : methode beta.fit a ne pas utiliser a priori
    if method=='beta.fit':
        a, b, loc, scale = beta.fit(dpe_data["conso_5_usages_ep_m2"]) # beta.fit prend comme argument des datas de type array_like
        print('Paramètres du beta.fit (a, b, loc, scale)', a, b, loc, scale)
        pdf = beta.pdf(list(counter_dict_sorted.keys()), a, b, loc=loc, scale=scale) # probability density function
        
        return pdf # todo: modifier pour qu'elle retourne un df, ou juste supprimer cette méthode sinon
    
        
    # methode à privilégier
    if method=='curve_fit':
        x_data = np.array(list(counter_dict_sorted.keys()))  # on a array of int et pas of float
        
        # Filtrage des valeurs extrêmes de consommations d'énergie primaire
        filtre = zscore(x_data)<3 # TODO: à enlever ? Le moins on filtre nos données le mieux c'est ?
        
        x_data = x_data[filtre]
        
        if verbose: 
            print(f'Le curve_fit ne prend pas en compte les DPE supérieurs à {x_data.max()} kWh/m2 (Z score > 3)')
        
        y_data_norm = np.array(list(counter_dict_sorted.values()))/nb_dpe
        y_data_norm = y_data_norm[filtre]
        y_data_norm = y_data_norm/y_data_norm.sum() # afin que l'aire sous la courbe soit bien =1
        
        # Création d'un DataFrame pour stocker les données
        fit_dpe_data_df = pd.DataFrame({'dep_code':dep_code, 'x_data':x_data, 'y_data_norm':y_data_norm})
        # todo: enlever dep_code car lourd pour rien ?
        
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
    
    
        # Estimation initiale des paramètres de la distribution beta
        alpha_first_guess = 4
        loc_first_guess = 0
        beta_first_guess =  alpha_first_guess / ((x_data - loc_first_guess)/x_data.max()).mean() - alpha_first_guess
        first_guess = (alpha_first_guess, beta_first_guess, loc_first_guess) 
        
        param, cov = curve_fit(beta_pdf, x_data, y_data_norm, first_guess, method='trf', bounds=(0, +np.inf), maxfev=10000)  # la méthode trf fonctionne bien 
        a, b, loc = param 
        print('Paramètres de la loi beta du curve_fit (a, b, loc, cov) :', a, b, loc, cov)


        fit_dpe_data_df['y_beta_curve_fit'] = beta_pdf(x_data, a, b, loc)


        # calcul du R2 avec les données 'réelles'
        r2_value = r2_score(fit_dpe_data_df['y_data_norm'],fit_dpe_data_df['y_beta_curve_fit'])
        print("R2 =", r2_value)
        
        return fit_dpe_data_df #, param # todo: renvoyer les paramètres




def plot_dpe_distribution(path, dep_code, save=True, plot_mean=True, plot_median=True, window_size=50, plot_fit=False, plot_curve_fit=True, max_xlim=600) :
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
        tracé du beta.fit des données. The default is False. (ancienne méthode)  # todo: à supprimer ?
    plot_curve_fit : boolean, optional
        tracé du curve_fit selon loi beta. The default is True.
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
    # todo: rajouter 
    
    # Tracé de la moyenne/médiane glissante 
    if plot_mean:
        plt.plot(formatage_dpe_data(dep_code, window_size)["conso_5_usages_ep_m2_arrondie"], formatage_dpe_data(dep_code, window_size)["nb_obs_moyenne"], "k", label='moyenne', linewidth = 0.7)
        
    if plot_median:
        plt.plot(formatage_dpe_data(dep_code, window_size)["conso_5_usages_ep_m2_arrondie"], formatage_dpe_data(dep_code, window_size)["nb_obs_mediane"], "r", label='mediane', linewidth = 0.7)
    
    
    # Tracé fit de toutes les données dpe_data["conso_5_usages_ep_m2"] avec une loi beta : fonction scipy.beta.fit 
    if plot_fit: 
        pdf = fit_dpe_data(dep_code, method='beta.fit')
        plt.plot(list(counter_dict_sorted.keys()), pdf*nb_dpe, label='beta fit', linewidth = 1)
            
    # Tracé fit de toutes les données : fonction curve_fit avec un modèle de loi beta (voir fonction fit_dpe_data)    
    if plot_curve_fit: 
        fit_dpe_data_df = fit_dpe_data(dep_code, method='curve_fit')
        x_data = fit_dpe_data_df['x_data']
        pdf = fit_dpe_data_df['y_beta_curve_fit']
        plt.plot(x_data, pdf*nb_dpe, "k--", label='curve_fit')

    ax.legend()    

    # Enregistrement de la figure
    if save:
        save_path = os.path.join(path,'distribution_dpe_{}.png'.format(dep_code))
        if plot_fit or plot_curve_fit : # todo: enlever et rajouter old_built_filter
            save_path = os.path.join(path,'distribution_dpe_{}_fit.png'.format(dep_code)) 
        # if old_built_filter:
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        plt.savefig(save_path, bbox_inches='tight')


    plt.show()
    plt.close()
    
    return
    



def calcul_bunching(dep_code, method, itv_bunching, plot_ecart, path, max_xlim = 600):
    """
    Calcul du bunching dans le département dep_code vec plusieurs méthodes possibles.
    
    Parameters
    ----------
    dep_code : str
        code du departement.
    method : str ('AMP' ou 'diff_beta_gauche' ou 'diff_beta_centre_abs')
        Nom de la méthode utilisée pour calculer le bunching.
    itv_bunching : int
        Attention : l'intervalle peut être soit à gauche du seuil, soit de part et d'autre du seuil selon les méthodes. 
        Methode 'AMP' : taille de l'intervalle en-dessous et au-dessus des seuils sur lequel on calcule le bunching. 
        Methode 'diff_beta' : taille de l'intervalle à gauche de chaque seuil (utiliser plutôt 10 kWh/m2 comme Aja et al.)
    plot_ecart : boolean
        tracé de l'écart entre les données et le fit.
    path : str
        chemin de sauvegarde.
    max_xlim : int, optional
        limite du graphe. The default is 600.    
    Returns
    -------
    bunching : panda DataFrame 
        Bunching pour chaque seuil : A/B, B/C, C/D, D/E, E/F et F/G.
        Le nom des colonnes contient la méthode utilisée (ex: A/B_method_diff_beta_gauche)
    """
    
    departement = Departement(dep_code)
    
    dpe_data = get_dpe_consumption(dep_code)
    dpe_data = dpe_data.dropna()
    dpe_data = dpe_data.map(round)
    nb_dpe = len(dpe_data)
    counter_dict = dict(dpe_data.conso_5_usages_ep_m2.value_counts())
    counter_dict_sorted = {k: v for k, v in sorted(counter_dict.items(), key=lambda item: item[0])}
    
    pd.options.display.max_columns = None
    
    
    if method=='AMP': # méthode "Average Manipulation Density" (Civel et al.).
        
        fit_dpe_data_df = fit_dpe_data(dep_code, method='curve_fit')
        fit_dpe_data_df['y_difference'] = fit_dpe_data_df.y_data_norm - fit_dpe_data_df.y_beta_curve_fit
        fit_dpe_data_df['y_difference_abs'] =  fit_dpe_data_df['y_difference'].abs()
        
        bunching_df = pd.DataFrame(index=[0]) 
        
        for k, seuil in etiquette_ep_seuils.items():
            # Création d'un DataFrame filtré sur l'intervalle à +-{itv_bunching} du seuil
            fit_dpe_data_df_filtered = fit_dpe_data_df[(fit_dpe_data_df.x_data> seuil-itv_bunching) & (fit_dpe_data_df.x_data <= seuil + itv_bunching)]
            
            # Ajout d'une colonne correspondante au seuil dans le bunching DataFrame 
            bunching_df[f'{k}_method_{method}'] = fit_dpe_data_df_filtered['y_difference_abs'].sum()
            
# =============================================================================
#     
#         for k, seuil in etiquette_ep_seuils.items():
#             nb_droite = sum([int(v) for k,v in counter_dict_sorted.items() if k > seuil and k <= seuil+itv_bunching])
#             nb_gauche = sum([int(v) for k,v in counter_dict_sorted.items() if k > seuil-itv_bunching and k <= seuil])
#             AMP = (nb_gauche - nb_droite) / (nb_gauche+nb_droite)  # average manipulation density # on pourrait aussi diviser par nb tot DPE (AJa et al) ? 
#             AMP = round(AMP,3)
#             bunching[k] = AMP
# =============================================================================
            
        print(f'Bunching (méthode {method}) pour {departement}, avec un intervalle de +-{itv_bunching} kWh/m2 autour des seuils : \n', bunching_df)

        return bunching_df
    # todo modifier pour que ce redonne comme methode diff_beta_gauche et faire les calculs sur les densité et pas les nb d'observations (pour comparer entre les départements)
            
    
    
    if method=='diff_beta_gauche': # différence d'aire sous la courbe entre les données réelles et le curve_fit sur les données, dans les intervalles à gauche des seuils. Plot et enregistre la figure de l'écart données/curve_fit

        fit_dpe_data_df = fit_dpe_data(dep_code, method='curve_fit')
        fit_dpe_data_df['y_difference'] = fit_dpe_data_df.y_data_norm - fit_dpe_data_df.y_beta_curve_fit
        
        #print(fit_dpe_data_df)
        
        # Calcul du bunching 
        # méthode part excessive standardisée (Aja et al.) -> "part des DPE qui sont excessifs sur l’intervalle de 10 kWh de consommation d’énergie à gauche de chaque seuil"
        
        bunching_df = pd.DataFrame(index=[0]) 
        
        for k, seuil in etiquette_ep_seuils.items():
            # Création d'un DataFrame filtré sur l'intervalle à gauche du seuil
            fit_dpe_data_df_filtered = fit_dpe_data_df[(fit_dpe_data_df.x_data> seuil-itv_bunching) & (fit_dpe_data_df.x_data <= seuil)]
            
            # Ajout d'une colonne correspondante au seuil dans le bunching DataFrame 
            bunching_df[f'{k}_method_{method}'] = fit_dpe_data_df_filtered['y_difference'].sum()
            
        #bunching_df.round(3) # todo: ne marche pas ? de toute facon on ne veut pas perdre d'info , utiliser plutot f'{var:.2f}'
        print(f"Bunching (méthode {method}) pour {departement}, sur l'intervalle de {itv_bunching} kWh/m2 à gauche de chaque seuil : \n", bunching_df)
        
        
        # Remarque : on somme sur y_difference, donc on obtient le bunching normalisé
        
        
        # Tracé de l'écart entre les données réelles et le fit sur toutes les données
        if plot_ecart:
            plt.figure()
            plt.plot(fit_dpe_data_df.x_data, fit_dpe_data_df.y_difference, linewidth = 0.7)
            
            plt.xlim([0,max_xlim])
            plt.hlines(y=0, xmin=0, xmax=max_xlim, color='k', linestyles='dashed', zorder=-1) # tracé de l'axe y=0 en arrière-plan
            plt.title(f"Ecart entre la distribution des DPE et la distribution beta\n({departement.name} - {departement.code})")
            plt.ylabel("Nombre de DPE de différence, normalisé")
            plt.xlabel("Consommation annuelle en énergie primaire (kWh.m$^{-2}$)")
            plt.xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [max_xlim])
            
            # Enregistrement de la figure
            save_path = os.path.join(path,'ecart_curve_fit_{}.png'.format(dep_code))
            plt.savefig(save_path, bbox_inches='tight')
    
        return bunching_df
    
    
      
    if method=='diff_beta_centre_abs': # différence absolue d'aire sous la courbe entre les données réelles et le curve_fit sur les données, dans les intervalles à gauche ET A DROITE des seuils. 
        
        fit_dpe_data_df = fit_dpe_data(dep_code, method='curve_fit')
        fit_dpe_data_df['y_difference'] = fit_dpe_data_df.y_data_norm - fit_dpe_data_df.y_beta_curve_fit
        fit_dpe_data_df['y_difference_abs'] =  fit_dpe_data_df['y_difference'].abs()
        
        bunching_df = pd.DataFrame(index=[0]) 
        
        for k, seuil in etiquette_ep_seuils.items():
            # Création d'un DataFrame filtré sur l'intervalle à +-{itv_bunching} du seuil
            fit_dpe_data_df_filtered = fit_dpe_data_df[(fit_dpe_data_df.x_data> seuil-itv_bunching) & (fit_dpe_data_df.x_data <= seuil + itv_bunching)]
            
            # Ajout d'une colonne correspondante au seuil dans le bunching DataFrame 
            bunching_df[f'{k}_method_{method}'] = fit_dpe_data_df_filtered['y_difference_abs'].sum()
            
        print(f'Bunching (méthode {method}) pour {departement}, avec un intervalle de +-{itv_bunching} kWh/m2 autour des seuils : \n', bunching_df)


      # Tracé de l'écart entre les données réelles et le fit sur toutes les données
        if plot_ecart:
          plt.figure()
          plt.plot(fit_dpe_data_df.x_data, fit_dpe_data_df.y_difference, linewidth = 0.7)
          
          plt.xlim([0,max_xlim])
          plt.ylim([-0.003, 0.009]) # correspond aux bornes pour la Haute-Marne 52 (max bunching)
          # plt.ylim([-0.003=2, 0.006]) # correspond aux bornes pour la Vendée 85 (max somme bunching)
          plt.hlines(y=0, xmin=0, xmax=max_xlim, color='k', linestyles='dashed', zorder=-1) # tracé de l'axe y=0 en arrière-plan
          plt.title(f"Ecart entre la distribution des DPE et la distribution beta\n({departement.name} - {departement.code})")
          plt.ylabel("Nombre de DPE de différence, normalisé")
          plt.xlabel("Consommation annuelle en énergie primaire (kWh.m$^{-2}$)")
          plt.xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [max_xlim])
          
          # Enregistrement de la figure
          save_path = os.path.join(path,'ecart_curve_fit_{}.png'.format(dep_code))   # todo: modifier nom !!
          plt.savefig(save_path, bbox_inches='tight')

        return bunching_df
            
    
    
# =============================================================================

#     if method == 'diff_beta_old': # OLD VERSION AVEC BETA.FIT ET PAS DE DATAFRAME
#         
#         pdf = fit_dpe_data(dep_code, method='beta.fit')
#         difference = list(counter_dict_sorted.values()) - pdf*nb_dpe
#         difference_dict =  dict(zip(counter_dict_sorted.keys(), difference)) # dictionnaire qui lie l'écart au beta.fit des DPE à leur conso annuelle d'ep associée
#         
#         # Calcul du bunching
#         # méthode part excessive standardisée (Aja et al.) -> "part des DPE qui sont excessifs sur l’intervalle de 10 kWh de consommation d’énergie à gauche de chaque seuil"
#         for k, seuil in etiquette_ep_seuils.items():
#             part_excess = sum([float(v) for k,v in difference_dict.items() if k > seuil-itv_bunching and k <= seuil])  # omme des DPEs excessifs à gauche du seuil
#             part_excess = part_excess/nb_dpe   # normalisation pour obtenir la proportion des DPEs qui seraient excessifs
#             part_excess = round(part_excess,3)
#             bunching[k] = part_excess
#         
#         pd.options.display.max_columns = None
#         print(f"Bunching (méthode part excessive) pour {departement}, sur l'intervalle de {itv_bunching} kWh/m2 à gauche de chaque seuil : ", bunching)
#         
#         if plot_ecart:
#             # Tracé de l'écart entre les données réelles et le beta.fit sur toutes les données
#             plt.figure()
#             plt.plot(list(counter_dict_sorted.keys()), difference, linewidth = 0.7)
#             
#             plt.xlim([0,max_xlim])
#             plt.hlines(y=0, xmin=0, xmax=max_xlim, color='k', linestyles='dashed') # tracé de l'axe y=0
#             plt.title(f"Ecart au beta.fit des DPE ({departement.name} - {departement.code})")
#             plt.ylabel("Nombre de DPE de différence")
#             plt.xlabel("Consommation annuelle en énergie primaire (kWh.m$^{-2}$)")
#             plt.xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [max_xlim])
#             
#             # Enregistrement de la figure
#             save_path = os.path.join(path,'ecart_beta.fit_{}.png'.format(dep_code))
#             plt.savefig(save_path, bbox_inches='tight')
#     
#         return bunching
# 
# =============================================================================

#%%


def calcul_bunching_france(path, method, itv_bunching, max_xlim = 600, verbose=False):
    
    # todo: rajouter definition de cette fonction
    
    #path : chemin de sauvegarde des figures (pas du dictionnaire bunching)
    

    # Définition du chemin de sauvegarde des bunchings en .csv
    output_folder_bunching = os.path.join('output', 'buching')
    os.makedirs(output_folder_bunching, exist_ok=True)
    existing_files = os.listdir(output_folder_bunching)
    
    # Définition du nom du fichier final
    save_name = f'france_bunching_method_{method}_itv_bunching_{itv_bunching}_kWh.csv'
    #if old_built_filter:
        #save_name = save_name.replace('.csv','_old_built.csv')
    
    
    if save_name not in existing_files:
        france = France()
        
        # Initialisation du dictionnaire de bunching avec des listes vides
        dict_france_bunching = {'dep_code':[]}
        for k in etiquette_ep_seuils.keys():
            dict_france_bunching[f'{k}_method_{method}'] = []
        
        # Implémentation du dictionnaire grâce à calcul_bunching 
        for dep in france.departements :
            dep_code = dep.code
            print(dep)
            bunching_dep = calcul_bunching(dep_code, method, itv_bunching, plot_ecart=False, path=path, max_xlim=max_xlim) # calcul du bunching : choix de la méthode et de ses paramètres
            dict_france_bunching['dep_code'].append(dep_code)
            # Implémentation de la liste des bunchings du département
            for k in etiquette_ep_seuils.keys():
                dict_france_bunching[f'{k}_method_{method}'].append(bunching_dep[f'{k}_method_{method}'].values[0])
        
        # Transformation du dictionnaire de listes en DataFrame
        france_bunching = pd.DataFrame().from_dict(dict_france_bunching)
        france_bunching = france_bunching.set_index('dep_code')
        
        # Enregistrement du DataFrame du bunching en .csv
        france_bunching.to_csv(os.path.join(output_folder_bunching, save_name))

        
        if verbose:
            print('dict_dep APRES', dict_france_bunching)
        
        
    else:
        france_bunching = pd.read_csv(os.path.join(output_folder_bunching, save_name), index_col='dep_code')

        
    return france_bunching


# def fonction_affichage:
    # prend en entrée df (read csv ou le calcule)
    # zip parcourt deux listes de la même manière en meme temps
    # dict_dep_bunching = {Departement(dep_code):bunching_AB for dep_code, bunching_AB in zip(df.dep_code, df.bunching_AB)}
    # définir l'index du df : ici c'est le département df  = df.set_index('dep_code') comme ça quand on fait des calculs il prend pas en compte l'index (somme sur les colonnes par ex)
    # creer nouvelle colonne dans le df qui correspond a somme_bunching
    # attention axis=1 pour moyenne sur les départements et pas les bunching
    # pas grave si on enregistre pas les colonnes sommes"
    
    
#%% ===========================================================================
# script principal
# =============================================================================

def main():
    tic = time.time()
    
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')
    output_folder = os.path.join('output',today)
    os.makedirs(output_folder, exist_ok=True)
    
    dep = Departement('52')
    
    
    # tracé de la distribution des dpe du département
    if True:
        #dpe_data = get_dpe_consumption(dep.code) # cette ligne ne sert a rien car déjà dans plot_dpe_distribution ?
        plot_dpe_distribution(output_folder,dep.code, plot_mean=True, plot_median=True, plot_fit=False, plot_curve_fit=True, max_xlim=600) # todo: specifier 
    
        
    # BUNCHING
        
    # choix des paramètres de mesure du bunching
    method='diff_beta_centre_abs'
    itv_bunching = 5    
    
        
    # calcul du bunching du département
    if True:
        #calcul_bunching(dep.code, method='AMP', itv_bunching=5, path=output_folder)
        bunching_dep = calcul_bunching(dep.code, method=method, itv_bunching=itv_bunching, plot_ecart = True, path=output_folder)
        bunching_dep_sum = bunching_dep.sum(axis=1) # on somme le bunching de l'ensemble des 6 seuils        

        print('bunching_dep_sum =', bunching_dep_sum.iloc[0])
            
        
    # carte du bunching     
    if False:
        today = pd.Timestamp(date.today()).strftime('%Y%m%d')
        output_folder = os.path.join('output',today)
        os.makedirs(output_folder, exist_ok=True)
        
        france_bunching = calcul_bunching_france(output_folder, method=method, itv_bunching=itv_bunching, max_xlim=600)
        
        
        # un dictionnaire par cartes, mais autant de méthodes qu'on veut a partir 

        
        # Somme sur l'ensemble des 6 seuils
        france_bunching[f'Somme_method_{method}'] = france_bunching.sum(axis=1) # on somme sur les lignes et non les colonnes
        dict_dep_bunching = {Departement(dep_code):bunching_somme for dep_code, bunching_somme in zip(france_bunching.index, france_bunching[f'Somme_method_{method}'])}
        draw_departement_map(dict_dep_bunching,output_folder,save=f"Carte somme du bunching sur l'ensemble des seuils (Méthode {method}, intervalle de bunching = {itv_bunching} kWh.m-2))", map_title=f"Somme du bunching sur l'ensemble des seuils\n(Méthode {method}, intervalle de bunching = {itv_bunching} kWh.m$^{-2}$))")


        # Moyenne des methodes
        france_bunching[f'Moyenne_method_{method}'] = france_bunching.mean(axis=1)
        dict_dep_bunching = {Departement(dep_code):bunching_mean for dep_code, bunching_mean in zip(france_bunching.index, france_bunching[f'Moyenne_method_{method}'])}
        draw_departement_map(dict_dep_bunching,output_folder,save=f"Carte moyenne du bunching sur l'ensemble des seuils (Méthode {method}, intervalle de bunching = {itv_bunching} kWh.m-2))", map_title=f"Moyenne du bunching sur l'ensemble des seuils\n(Méthode {method}, intervalle de bunching = {itv_bunching} kWh.m$^{-2}$))")
      
        
        # Bunching sur le seuil E/F seulement
        #dict_dep_bunching = {Departement(dep_code):bunching_somme for dep_code, bunching_somme in zip(france_bunching.index, france_bunching[f'Somme_method_{method}'])}
        # todo :modif 


        #france_bunching = france_bunching_method1.join(france_bunching_method2)
        # pour pouvoir faire des stats sur l'ensemble des methodes
        
       
        
    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')
    
if __name__ == '__main__':
    main()
