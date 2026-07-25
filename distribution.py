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
from scipy.stats import beta, zscore, pearsonr # attention, beta (variable aléatoire) =/= sc.beta (fonction)
from scipy.optimize import curve_fit
import scipy.special as sc 
from datetime import date
from sklearn.metrics import r2_score
import seaborn as sns
import tqdm

from administrative import  list_dep_code, Departement, France, draw_departement_map
from download import get_bdnb
from utils import etiquette_colors_dict,etiquette_ep_dict,etiquette_ep_seuils
from manipulation_dpe import dicts_dep_gain_moyen_etiquette


methods_dict = {'diff_simple':'différence simple',
           'diff_simple_nb_dpe':'différence simple *',
           'diff_beta_gauche': 'écart distribution bêta, gauche',
           'diff_beta': 'écart distribution bêta',
           'diff_beta_itv': 'écart distribution bêta *',
           'diff_moyenne': 'écart moyenne glissante',
           'diff_moyenne_itv': 'écart moyenne glissante *',
           'diff_moyenne_gauche_itv': 'écart moyenne glissante, gauche',
           #'diff_moyenne_classes' : 'écart moyenne glissante, classes' 
           }



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
    # save_name = f'conso_5_usages_millesime_2026-02_dep{dep_code}.csv'
    
    if old_built_filter:
        save_name = save_name.replace('.csv','_old_built.csv')
        
    # Enregistrement des conso_5_usages en csv    
    if save_name not in existing_files:
        dpe_data, _ , _ = get_bdnb(dep_code)
        dpe_data = dpe_data[dpe_data.type_dpe=='dpe arrêté 2021 3cl logement'][['conso_5_usages_ep_m2','conso_5_usages_ef_m2','periode_construction_dpe','surface_habitable_logement','date_etablissement_dpe']].compute() 
        if old_built_filter: 
            dpe_data = dpe_data[dpe_data.periode_construction_dpe.isin(['avant 1948','1948-1974'])]
        
        #dpe_data = dpe_data[date_etablissement_dpe < 2024] # filtre qui enlève DPE après 2024 (réforme petites surfaces) --> mais enlève trop de données 
        #dpe_data = dpe_data[dpe_data.surface_habitable_logement > 40.] # todo: filtre qui enlève petits logements (en dessous de 40 m2) , garde Nan ou pas ?
        #dpe_data = dpe_data[(dpe_data.surface_habitable_logement > 40.) & (date_etablissement_dpe < 2024)] # filtrage des petits logements après 2024 car n'ont pas les même seuils entre les classes
        
        dpe_data = dpe_data[['conso_5_usages_ep_m2','conso_5_usages_ef_m2']]
        dpe_data.to_csv(os.path.join(output_folder, save_name))
        
    else:
        dpe_data = pd.read_csv(os.path.join(output_folder, save_name))
        
    return dpe_data



def formatage_dpe_data(dep_code, window_size, old_built_filter=False):
    """
    Créé un DataFrame de la distribution des données DPE du département dep_code, avec une colonne du nombre d'observation moyen 
    et une colonne du nombre d'observation médian pour chaque valeur de consommation d'énergie primaire. 
    Glissement sur window_size à l'aide de la méthode rolling.

    Parameters
    ----------
    dep_code : str
        code du département.
    window_size : int, optional
        taille de la fenêtre de glissement pour le rolling.
    old_built_filter : boolean, optional
        filtrage : on ne garde que les logements construits avant 1974. The default is False.

    Returns
    -------
    counter_df_sorted : panda DataFrame
        DataFrame du nombre d'observation exact, moyen glissant et médian glissant de chaque valeur entière de conso_5_usages_ep_m2 dans le département
        Colonnes :  conso_5_usages_ep_m2_arrondie  |  nb_observations  |  nb_obs_moyenne  |  nb_obs_mediane
    """
    
    dpe_data = get_dpe_consumption(dep_code, old_built_filter=old_built_filter) # prend du temps
    dpe_data = dpe_data.dropna()
    dpe_data = dpe_data.map(round)
    counter_dict = dict(dpe_data.conso_5_usages_ep_m2.value_counts())
    counter_dict_sorted = {k: v for k, v in sorted(counter_dict.items(), key=lambda item: item[0])}
    counter_df_sorted = pd.DataFrame(list(counter_dict_sorted.items()), columns=["conso_5_usages_ep_m2_arrondie","nb_observations"]) # df trié par conso_ep (besoin d'un DataFrame pour utiliser rolling)
    
    # tracé des données à fit
    # plt.plot(list(counter_dict_sorted.keys()), list(counter_dict_sorted.values()), "k", label='données', linewidth = 0.5)
    
    # calcul de la moyenne/médiane glissante
    rolling_dpe = counter_df_sorted['nb_observations'].rolling(window=window_size, min_periods=1, center=True) 
    counter_df_sorted["nb_obs_moyenne"] = rolling_dpe.mean()  # ajout colonne moyenne dans le DataFrame
    counter_df_sorted["nb_obs_mediane"] = rolling_dpe.median()  # ajout colonne mediane dans le DataFrame
    
    return counter_df_sorted


#%%


def fit_dpe_data(dep_code, method='curve_fit', old_built_filter=False, verbose=True):
    """
    Fit de la distribution des DPE du département dep_code avec plusieurs méthodes possibles. 

    Parameters
    ----------
    dep_code : str
        code du département.
    method : str, optional ('beta.fit' ou 'curve_fit')
        Nom de la méthode utilisée pour "fitter". The default is 'curve_fit'.
    old_built_filter : boolean, optional
        filtrage : on ne garde que les logements construits avant 1974. The default is False.
    verbose : boolean
        affiche les print lors du run de la fonction. 

    Returns
    -------
    fit_dpe_data_df : pandas DataFrame
        Fit de la distribution du département = densité de probabilité des consommations d'énergie primaire dans le dep
        Colonnes : x_data (filtré tq zscore<3)  |  y_data_norm (distribution normalisée)  |  y_beta_curve_fit (fit de y_data_norm)
    r2_value : float
        Coefficient de régression entre le curve_fit et les données réelles
    param : array of floats --> array([ a, b, loc])
        Paramètres alpha, beta et loc de la distribution beta fitée sur les données DPE.
    nb_dpe_filtre : int
        nombre de DPE restants une fois qu'on a enlevé les zscore > 3 pour le fit
    """
    
    dpe_data = get_dpe_consumption(dep_code, old_built_filter=old_built_filter) # prend du temps
    dpe_data = dpe_data.dropna()
    dpe_data = dpe_data.map(round)
    nb_dpe = len(dpe_data)
    counter_dict = dict(dpe_data.conso_5_usages_ep_m2.value_counts())
    counter_dict_sorted = {k: v for k, v in sorted(counter_dict.items(), key=lambda item: item[0])} 

    
    # OLD : methode beta.fit a ne pas utiliser a priori
    if method=='beta.fit':
        a, b, loc, scale = beta.fit(dpe_data["conso_5_usages_ep_m2"]) # beta.fit prend comme argument des datas de type array_like
        if verbose: 
            print('Paramètres du beta.fit (a, b, loc, scale)', a, b, loc, scale)
        pdf = beta.pdf(list(counter_dict_sorted.keys()), a, b, loc=loc, scale=scale) # probability density function
        
        return pdf # todo: modifier pour qu'elle retourne un df, ou juste supprimer cette méthode sinon
    
        
    # methode à privilégier
    if method=='curve_fit':
        x_data = np.array(list(counter_dict_sorted.keys()))  # it is an array of int and not float
        
        # Filtrage des valeurs extrêmes de consommations d'énergie primaire
        filtre = zscore(x_data)<3 # todo: à enlever ? Le moins on filtre nos données le mieux c'est ?
        
        x_data = x_data[filtre]
        if verbose: 
            print(f'Le curve_fit ne prend pas en compte les DPE supérieurs à {x_data.max()} kWh/m2 (Z score > 3)')
        
        
        y_data = np.array(list(counter_dict_sorted.values()))[filtre]
        
        # Calcul du nombre de DPE une fois qu'on a filtré les zscore(x_data)<3 (à faire avant de normaliser y_data par nb_dpe)
        nb_dpe_filtre = y_data.sum()
        
        y_data_norm = y_data/nb_dpe
        y_data_norm = y_data_norm/y_data_norm.sum() # afin que l'aire sous la courbe soit bien =1
        
        
        # Création d'un DataFrame pour stocker les données
        fit_dpe_data_df = pd.DataFrame({'x_data':x_data, 'y_data':y_data, 'y_data_norm':y_data_norm})
        
        
        scale = x_data.max()
        
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
        if verbose: 
            print('Paramètres de la loi beta du curve_fit (a, b, loc, cov) :', a, b, loc, cov)


        fit_dpe_data_df['y_beta_curve_fit'] = beta_pdf(x_data, a, b, loc)

        # ajout d'une colonne non normalisée du beta_fit --> on repasse à l'échelle initiale
        fit_dpe_data_df['y_beta_curve_fit_scale_up'] = fit_dpe_data_df.y_beta_curve_fit * nb_dpe_filtre

        # calcul du R2 entre les données et le curve_fit
        r2_value = r2_score(fit_dpe_data_df['y_data_norm'],fit_dpe_data_df['y_beta_curve_fit'])
        if verbose: 
            print("R2 data/curve_fit =", r2_value)
        
# =============================================================================
#         # calcul du R2 entre la moyenne des données et le curve_fit

#         y_moyenne = formatage_dpe_data(dep_code, window_size=window_size, old_built_filter=old_built_filter)["nb_obs_moyenne"]/nb_dpe
#         y_moyenne = y_moyenne[filtre]
#         y_moyenne = y_moyenne/y_moyenne.sum()
#         print('somme y_moyenne', y_moyenne.sum())

#         r2_value = r2_score(y_moyenne, fit_dpe_data_df['y_beta_curve_fit'])
#         print("R2 moyenne/curve_fit =", r2_value)

#         # todo: pertinence du R2 avec la moyenne ? il faut alors justifier la fenêtre de rolling adaptée 
#         # mais en meme temps R2 avec données brutes n'a pas trop de sens non plus
# =============================================================================
        
        
        return fit_dpe_data_df, r2_value, param, nb_dpe_filtre 


#%% 


def plot_dpe_distribution(path, dep_code, save, plot_mean, plot_median, window_size, plot_curve_fit=True, old_built_filter=False, max_xlim=600) :
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
    plot_curve_fit : boolean, optional
        tracé du curve_fit selon loi beta. The default is True.
    old_built_filter : boolean, optional
        filtrage : on ne garde que les logements construits avant 1974. The default is False.
    max_xlim : int, optional
        limite du graphe. The default is 600.

    Returns
    -------
    None
    """
    departement = Departement(dep_code)
    
    dpe_data = get_dpe_consumption(dep_code, old_built_filter=old_built_filter) 
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
    fig.suptitle(f"{departement.name} - {departement.code}")
    if old_built_filter:
        ax.set_title(f"Construction avant 1974. Taille de l'échantillon : {nb_dpe} DPE", fontsize=10)
    else : 
        ax.set_title(f"Taille de l'échantillon : {nb_dpe} DPE", fontsize=10)
    fig.subplots_adjust(top=0.9)
    ax.set_ylabel("Nombre d'observations")
    ax.set_xlabel("Consommation annuelle en énergie primaire (kWh.m$^{-2}$)")
    ax.set_xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [max_xlim])
    
    # Tracé de la moyenne/médiane glissante 
    if plot_mean:
        ax.plot(formatage_dpe_data(dep_code, window_size, old_built_filter=old_built_filter)["conso_5_usages_ep_m2_arrondie"], formatage_dpe_data(dep_code, window_size, old_built_filter=old_built_filter)["nb_obs_moyenne"], "k", label='moyenne', linewidth = 1)
        
    if plot_median:
        ax.plot(formatage_dpe_data(dep_code, window_size, old_built_filter=old_built_filter)["conso_5_usages_ep_m2_arrondie"], formatage_dpe_data(dep_code, window_size, old_built_filter=old_built_filter)["nb_obs_mediane"], "r", label='mediane', linewidth = 1)
            
    # Tracé fit de toutes les données : fonction curve_fit avec un modèle de loi beta (voir fonction fit_dpe_data)    
    if plot_curve_fit: 
        fit_dpe_data_df, r2_value, param, nb_dpe_filtre = fit_dpe_data(dep_code, method='curve_fit', old_built_filter=old_built_filter, verbose=False)
        x_data = fit_dpe_data_df['x_data']
        pdf = fit_dpe_data_df['y_beta_curve_fit_scale_up']
        ax.plot(x_data, pdf, "k--", label=f'curve_fit\n(R$^{{2}}$={r2_value:.2f})')
  
# =============================================================================
#     # Pour illustration méthode diff_simple

#     plt.vlines(320.5, 0, counter_dict_sorted[320], color='k', linestyles='dashed') 
#     plt.vlines(340.5, 0, counter_dict_sorted[340], color='k', linestyles='dashed') 
#     x_translation = [x-9 for x in counter_dict_sorted.keys()]
#     ax.plot(x_translation, list(counter_dict_sorted.values()), "k--", ds='steps-mid', linewidth = 0.7)
#     ax.annotate("$b_{E/F}$",
#         xy=(0.375, 0.38),
#         xycoords=ax.transAxes,
#         ha = 'center',
#         fontsize=20)
#     ax.set_xlim([310,350])
#     ax.set_ylim([0,260])
# =============================================================================
        
    ax.legend()    

    # Enregistrement de la figure
    if save:
        if old_built_filter:
            save_path = os.path.join(path,'distribution_dpe_{}_old_built.png'.format(dep_code)) 
        else: 
            save_path = os.path.join(path,'distribution_dpe_{}.png'.format(dep_code))
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    plt.close()
    
    return
    

#%%


def calcul_bunching(dep_code, method, itv_bunching, window_size, plot_ecart, path, old_built_filter, max_xlim = 600, verbose=False):
    """
    Calcul du bunching dans le département dep_code avec plusieurs méthodes possibles.
    
    Parameters
    ----------
    dep_code : str
        code du departement.
    method : str 
        nom de la méthode utilisée pour calculer le bunching.
        ('diff_simple' ou 'diff_simple_nb_dpe' 
        ou 'diff_beta_gauche' ou 'diff_beta' ou 'diff_beta_itv' 
        ou 'diff_moyenne' ou 'diff_moyenne_itv' ou 'diff_moyenne_classes' ou 'diff_moyenne_gauche_itv')
    itv_bunching : int
        Attention : l'intervalle peut être soit à gauche du seuil, soit de part et d'autre du seuil selon les méthodes. 
        Methodes 'diff_simple' et 'diff_simple_nb_dpe' : taille de l'intervalle de part et d'autre des seuils. 
        Methode 'diff_beta_gauche' : taille de l'intervalle à gauche de chaque seuil (utiliser plutôt 10 kWh/m2 comme Aja et al.).
        Methodes 'diff_beta' ou 'diff_beta_itv': taille de l'intervalle de part et d'autre des seuils.
        Methodes 'diff_moyenne', 'diff_moyenne_itv' et 'diff_moyenne_classes' : taille de l'intervalle de part et d'autre des seuils
        Methode 'diff_moyenne_gauche_itv' : taille de l'intervalle à gauche de chaque seuil.
    window_size : int
        taille de la fenêtre de glissement pour le rolling (moyenne glissante). Utile uniquement pour la méthode 'diff_moyenne'.
    plot_ecart : boolean
        tracé de l'écart entre les données et le fit.
    path : str
        chemin de sauvegarde des figures.
    old_built_filter : boolean
        filtrage : on ne garde que les logements construits avant 1974.
    max_xlim : int, optional
        limite du graphe. The default is 600.    
        
    Returns
    -------
    bunching : panda DataFrame 
        Bunching du département pour chaque seuil :  A/B_{method}  |  B/C_{method}  |  C/D_{method}  |  D/E_{method}  |  E/F_{method}  |  F/G_{method}
        Le nom des colonnes contient la méthode utilisée (ex: A/B_method_diff_beta_gauche)
    """
    
    departement = Departement(dep_code)
    
    dpe_data = get_dpe_consumption(dep_code, old_built_filter=old_built_filter)
    dpe_data = dpe_data.dropna()
    dpe_data = dpe_data.map(round)
    nb_dpe = len(dpe_data)
    counter_dict = dict(dpe_data.conso_5_usages_ep_m2.value_counts())
    counter_dict_sorted = {k: v for k, v in sorted(counter_dict.items(), key=lambda item: item[0])}
    
    pd.options.display.max_columns = None # pour que les bunching_df s'affichent entièrement lors des print (si verbose == True)

    
    # METHODES DIFFERENCE SIMPLE
    
    
    if method=='diff_simple': # méthode différence simple autour du seuil (utilisée par Civel et al. 2025)
        
        bunching_df = pd.DataFrame(index=[0]) # initialisation d'un DataFrame
        
        for seuil, valeur_seuil in etiquette_ep_seuils.items():
            nb_droite = sum([int(v) for k,v in counter_dict_sorted.items() if k > valeur_seuil and k <= valeur_seuil+itv_bunching])
            nb_gauche = sum([int(v) for k,v in counter_dict_sorted.items() if k > valeur_seuil-itv_bunching and k <= valeur_seuil])
               
            if verbose:
                print(f'nb_gauche_{seuil}', nb_gauche)
                print(f'nb_droite_{seuil}', nb_droite)

            diff_simple = (nb_gauche - nb_droite) / (nb_gauche+nb_droite)
         
            # Ajout d'une colonne correspondante au seuil dans le bunching DataFrame 
            bunching_df[f'{seuil}_method_{method}'] = diff_simple
            
        if verbose:  
            print(f'Bunching (méthode {method}) pour {departement}, avec un intervalle de +-{itv_bunching} kWh/m2 autour des seuils, old_built_filter = {old_built_filter} : \n', bunching_df)

    
    if method=='diff_simple_nb_dpe': # méthode différence simple, mais divisée par le nombe total de DPE (utilisée par Aja et al. 2024)
        
        bunching_df = pd.DataFrame(index=[0]) # initialisation d'un DataFrame
        
        for seuil, valeur_seuil in etiquette_ep_seuils.items():
            nb_droite = sum([int(v) for k,v in counter_dict_sorted.items() if k > valeur_seuil and k <= valeur_seuil+itv_bunching])
            nb_gauche = sum([int(v) for k,v in counter_dict_sorted.items() if k > valeur_seuil-itv_bunching and k <= valeur_seuil])
            
            if verbose:
                print(f'nb_gauche_{seuil}', nb_gauche)
                print(f'nb_droite_{seuil}', nb_droite)

            diff_simple_nb_dpe = (nb_gauche - nb_droite) / nb_dpe  
         
            # Ajout d'une colonne correspondante au seuil dans le bunching DataFrame 
            bunching_df[f'{seuil}_method_{method}'] = diff_simple_nb_dpe
        
        if verbose:
            print(f'Bunching (méthode {method}) pour {departement}, avec un intervalle de +-{itv_bunching} kWh/m2 autour des seuils, old_built_filter = {old_built_filter} : \n', bunching_df)



    # METHODES ECART A UNE DISTRIBUTION BETA           

    
    if method=='diff_beta_gauche': # différence d'aire sous la courbe entre les données réelles et le curve_fit sur les données, dans les intervalles à gauche des seuils. 
        # méthode part excessive standardisée (Aja et al.) -> "part des DPE qui sont excessifs sur l’intervalle de 10 kWh de consommation d’énergie à gauche de chaque seuil"
        # Plot et enregistre la figure de l'écart données/curve_fit
        # Attention : il n'y a pas de valeur absolue dans cette méthode


        fit_dpe_data_df, r2_value, param, nb_dpe_filtre = fit_dpe_data(dep_code, method='curve_fit', old_built_filter=old_built_filter, verbose=verbose)
        fit_dpe_data_df['y_difference'] = fit_dpe_data_df.y_data_norm - fit_dpe_data_df.y_beta_curve_fit # remarque : ici le bunching est déjà normalisé par nb_dpe_filtre
        
        # Calcul du bunching 
        bunching_df = pd.DataFrame(index=[0]) # initialisation d'un DataFrame
        
        for seuil, valeur_seuil in etiquette_ep_seuils.items():
            # Création d'un DataFrame filtré sur l'intervalle à gauche du seuil
            fit_dpe_data_df_filtered = fit_dpe_data_df[(fit_dpe_data_df.x_data> valeur_seuil-itv_bunching) & (fit_dpe_data_df.x_data <= valeur_seuil)]
            
            # Ajout d'une colonne correspondante au seuil dans le bunching DataFrame 
            bunching_df[f'{seuil}_method_{method}'] = fit_dpe_data_df_filtered['y_difference'].sum()
            # bunching normalisé par nb_dpe_filtre
        
        if verbose:
            print(f"Bunching (méthode {method}) pour {departement}, sur l'intervalle de {itv_bunching} kWh/m2 à gauche de chaque seuil, old_built_filter = {old_built_filter} : \n", bunching_df)        
        
        
        # Tracé de l'écart entre les données réelles et le fit sur toutes les données
        if plot_ecart:
            fig, ax = plt.subplots(dpi=300)

            ax.plot(fit_dpe_data_df.x_data, fit_dpe_data_df.y_difference, linewidth = 0.7)
            
            ax.set_xlim([0,max_xlim])
            #ax.set_ylim([-0.003, 0.009]) # correspond aux bornes pour la Haute-Marne 52 (max bunching)
            ax.set_ylim([-0.002, 0.006]) # correspond aux bornes pour la Vendée 85 (max somme bunching)
            ax.hlines(y=0, xmin=0, xmax=max_xlim, color='k', linestyles='dashed', zorder=-1) # tracé de l'axe y=0 en arrière-plan
            
            # remplissage de l'aire du bunching
            # for valeur_seuil in etiquette_ep_seuils.values():
            for valeur_seuil in list(etiquette_ep_seuils.values())[3:6]: # pour avoir seulement à partir seuil D/E
                   condition_alentours_seuils = (fit_dpe_data_df.x_data > valeur_seuil - itv_bunching) & (fit_dpe_data_df.x_data <= valeur_seuil)
                   ax.fill_between(fit_dpe_data_df.x_data, y1=fit_dpe_data_df.y_difference, y2=0, where=condition_alentours_seuils, alpha=0.3, color='red')
            
            if old_built_filter:
                ax.set_title(f"Ecart entre la distribution des DPE (avant 1974) et la distribution beta\n({departement.name} - {departement.code})")
            else :
                ax.set_title(f"Ecart entre la distribution des DPE et la distribution beta\n({departement.name} - {departement.code})")
            ax.set_ylabel("Nombre de DPE de différence, normalisé")
            ax.set_xlabel("Consommation annuelle en énergie primaire (kWh.m$^{-2}$)")
            ax.set_xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [max_xlim])
            
            # Enregistrement de la figure
            if old_built_filter:
                save_path = os.path.join(path,f'ecart_curve_fit_{dep_code}_method_{method}_itv_{itv_bunching}_kWh_old_built.png')
            else:
                save_path = os.path.join(path,f'ecart_curve_fit_{dep_code}_method_{method}_itv_{itv_bunching}_kWh.png')
            plt.savefig(save_path, bbox_inches='tight')
    
    
    
    if method=='diff_beta': # différence absolue d'aire sous la courbe entre les données réelles et le curve_fit sur les données, dans les intervalles à gauche ET A DROITE des seuils. 
        
        fit_dpe_data_df, r2_value, param, nb_dpe_filtre = fit_dpe_data(dep_code, method='curve_fit', old_built_filter=old_built_filter, verbose=verbose)
        fit_dpe_data_df['y_difference'] = fit_dpe_data_df.y_data_norm - fit_dpe_data_df.y_beta_curve_fit # remarque : ici le bunching est déjà normalisé par nb_dpe_filtre
        fit_dpe_data_df['y_difference_abs'] =  fit_dpe_data_df['y_difference'].abs()
        
        bunching_df = pd.DataFrame(index=[0]) 
        
        for seuil, valeur_seuil in etiquette_ep_seuils.items():
            # Création d'un DataFrame filtré sur l'intervalle à +-{itv_bunching} du seuil
            fit_dpe_data_df_filtered = fit_dpe_data_df[(fit_dpe_data_df.x_data> valeur_seuil-itv_bunching) & (fit_dpe_data_df.x_data <= valeur_seuil + itv_bunching)]
            
            # Ajout d'une colonne correspondante au seuil dans le bunching DataFrame 
            bunching_df[f'{seuil}_method_{method}'] = fit_dpe_data_df_filtered['y_difference_abs'].sum()
            # bunching normalisé par nb_dpe_filtre
            
        if verbose:
            print(f'Bunching (méthode {method}) pour {departement}, avec un intervalle de +-{itv_bunching} kWh/m2 autour des seuils, old_built_filter = {old_built_filter} : \n', bunching_df)


      # Tracé de l'écart entre les données réelles et le fit sur toutes les données
        if plot_ecart:
            fig, ax = plt.subplots(dpi=300)
            ax.plot(fit_dpe_data_df.x_data, fit_dpe_data_df.y_difference, linewidth = 0.7)
            
            ax.set_xlim([0,max_xlim])
            #ax.set_ylim([-0.003, 0.009]) # correspond aux bornes pour la Haute-Marne 52 (max bunching)
            ax.set_ylim([-0.002, 0.006]) # correspond aux bornes pour la Vendée 85 (max somme bunching)
            ax.hlines(y=0, xmin=0, xmax=max_xlim, color='k', linestyles='dashed', zorder=-1) # tracé de l'axe y=0 en arrière-plan
              
            # remplissage de l'aire du bunching
            # for valeur_seuil in etiquette_ep_seuils.values():
            for valeur_seuil in list(etiquette_ep_seuils.values())[3:6]: # pour avoir seulement à partir seuil D/E
                   condition_alentours_seuils = (fit_dpe_data_df.x_data > valeur_seuil - itv_bunching) & (fit_dpe_data_df.x_data <= valeur_seuil + itv_bunching)
                   ax.fill_between(fit_dpe_data_df.x_data, y1=fit_dpe_data_df.y_difference, y2=0, where=condition_alentours_seuils, alpha=0.3, color='red')
                   
            if old_built_filter:
                ax.set_title(f"Ecart entre la distribution des DPE (avant 1974) et la distribution beta\n({departement.name} - {departement.code})")
            else :
                ax.set_title(f"{departement.name} - {departement.code}")
                # ax.set_title(f"Ecart entre la distribution des DPE et la distribution beta\n({departement.name} - {departement.code})")
            ax.set_ylabel("Nombre de DPE de différence, normalisé")
            ax.set_xlabel("Consommation annuelle en énergie primaire (kWh.m$^{-2}$)")
            ax.set_xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [max_xlim])
             
            # Enregistrement de la figure
            if old_built_filter:
                save_path = os.path.join(path,f'ecart_curve_fit_{dep_code}_method_{method}_itv_{itv_bunching}_kWh_old_built.png')  
            else:
                save_path = os.path.join(path,f'ecart_curve_fit_{dep_code}_method_{method}_itv_{itv_bunching}_kWh.png') 
            plt.savefig(save_path, bbox_inches='tight')



    if method=='diff_beta_itv': # différence absolue d'aire sous la courbe entre les données réelles et le curve_fit sur les données, 
    # dans les intervalles à gauche ET A DROITE des seuils, 
    # normalisé par le nombre de DPE dans l'intervalle à +-{itv_bunching} du seuil étudié
        
        fit_dpe_data_df, r2_value, param, nb_dpe_filtre = fit_dpe_data(dep_code, method='curve_fit', old_built_filter=old_built_filter, verbose=verbose)
        fit_dpe_data_df['y_difference'] = fit_dpe_data_df.y_data - fit_dpe_data_df.y_beta_curve_fit_scale_up  # ATTENTION : à ce stade, le bunching n'est pas normalisé
        fit_dpe_data_df['y_difference_abs'] =  fit_dpe_data_df['y_difference'].abs()
        
        bunching_df = pd.DataFrame(index=[0]) 
        
        for seuil, valeur_seuil in etiquette_ep_seuils.items():
            # Création d'un DataFrame filtré sur l'intervalle à +-{itv_bunching} du seuil
            fit_dpe_data_df_filtered = fit_dpe_data_df[(fit_dpe_data_df.x_data> valeur_seuil-itv_bunching) & (fit_dpe_data_df.x_data <= valeur_seuil + itv_bunching)]
            
            nb_dpe_itv = fit_dpe_data_df_filtered.y_data.sum() # nombre total de DPE dans l'intervalle à +-{itv_bunching} du seuil
            
            # Ajout d'une colonne correspondante au seuil dans le bunching DataFrame 
            bunching_df[f'{seuil}_method_{method}'] = fit_dpe_data_df_filtered['y_difference_abs'].sum() / nb_dpe_itv
            # bunching normalisé par nb_dpe_itv
            
        if verbose:
            print(f'Bunching (méthode {method}) pour {departement}, avec un intervalle de +-{itv_bunching} kWh/m2 autour des seuils, old_built_filter = {old_built_filter} : \n', bunching_df)


    
    # METHODES ECART A LA MOYENNE GLISSANTE
    
    
    if method =='diff_moyenne': # difference entre les données et la moyenne glissante, normalisé par nb_dpe (nb total de DPE dans le département) 
        
        dpe_data_df = formatage_dpe_data(dep_code=dep_code, window_size=window_size, old_built_filter=old_built_filter)  # moyenne glissante sur une fenêtre de taille window_size
        
        # Normalisation par nb total de DPE
        dpe_data_df['y_diff_moyenne_norm'] = (dpe_data_df.nb_observations - dpe_data_df.nb_obs_moyenne)/nb_dpe 
        dpe_data_df['y_diff_moyenne_norm_abs'] = dpe_data_df['y_diff_moyenne_norm'].abs() 

        
        bunching_df = pd.DataFrame(index=[0]) # initialisation d'un DataFrame
        
        for seuil, valeur_seuil in etiquette_ep_seuils.items():
            # Création d'un DataFrame filtré sur l'intervalle à +-{itv_bunching} du seuil
            dpe_data_df_filtered = dpe_data_df[(dpe_data_df.conso_5_usages_ep_m2_arrondie > valeur_seuil-itv_bunching) & (dpe_data_df.conso_5_usages_ep_m2_arrondie <= valeur_seuil + itv_bunching)]
            
            # Ajout d'une colonne correspondante au seuil dans le bunching DataFrame 
            bunching_df[f'{seuil}_method_{method}'] = dpe_data_df_filtered['y_diff_moyenne_norm_abs'].sum() 
                
        if verbose:
            print(f'Bunching (méthode {method}, window_size {window_size} kWh) pour {departement}, avec un intervalle de +-{itv_bunching} kWh/m2 autour des seuils, old_built_filter = {old_built_filter} : \n', bunching_df)

      # Tracé de l'écart entre les données réelles et la moyenne glissante sur window_size
        if plot_ecart:
            fig, ax = plt.subplots(dpi=300)
            ax.plot(dpe_data_df.conso_5_usages_ep_m2_arrondie, dpe_data_df.y_diff_moyenne_norm, linewidth = 0.7)
            
            ax.set_xlim([0,max_xlim])
            #ax.set_ylim([-0.003, 0.009]) # correspond aux bornes pour la Haute-Marne 52 (max bunching)
            ax.set_ylim([-0.002, 0.006]) # correspond aux bornes pour la Vendée 85 (max somme bunching)
            ax.hlines(y=0, xmin=0, xmax=max_xlim, color='k', linestyles='dashed', zorder=-1) # tracé de l'axe y=0 en arrière-plan
            
            # remplissage de l'aire du bunching
            # for valeur_seuil in etiquette_ep_seuils.values():
            for valeur_seuil in list(etiquette_ep_seuils.values())[3:6]: # pour avoir seulement à partir seuil D/E
                   condition_alentours_seuils = (dpe_data_df.conso_5_usages_ep_m2_arrondie > valeur_seuil - itv_bunching) & (dpe_data_df.conso_5_usages_ep_m2_arrondie <= valeur_seuil + itv_bunching)
                   ax.fill_between(dpe_data_df.conso_5_usages_ep_m2_arrondie, y1=dpe_data_df.y_diff_moyenne_norm, y2=0, where=condition_alentours_seuils, alpha=0.3, color='red')
                   
            if old_built_filter:
                ax.set_title(f"Ecart entre la distribution des DPE (avant 1974) et la moyenne glissante\n({departement.name} - {departement.code})")
            else :
                ax.set_title(f"{departement.name} - {departement.code}")
                # ax.set_title(f"Ecart entre la distribution des DPE et la moyenne glissante\n({departement.name} - {departement.code})") # todo: afficher window_size partout
            ax.set_ylabel("Nombre de DPE de différence, normalisé")
            ax.set_xlabel("Consommation annuelle en énergie primaire (kWh.m$^{-2}$)")
            ax.set_xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [max_xlim])
            
            # Enregistrement de la figure
            if old_built_filter:
                save_path = os.path.join(path,f'ecart_moy_gliss_sur_{window_size}_kWh_{dep_code}_method_{method}_itv_{itv_bunching}_kWh_old_built.png')  
            else:
                save_path = os.path.join(path,f'ecart_moy_gliss_sur_{window_size}_kWh_{dep_code}_method_{method}_itv_{itv_bunching}_kWh.png')   
            plt.savefig(save_path, bbox_inches='tight')
            
            
           
    if method =='diff_moyenne_itv': # difference entre les données et la moyenne glissante
    # dans les intervalles à gauche ET A DROITE des seuils, 
    # normalisé par le nombre de DPE dans l'intervalle à +-{itv_bunching} du seuil étudié (nb_dpe_itv)
        
        dpe_data_df = formatage_dpe_data(dep_code=dep_code, window_size=window_size, old_built_filter=old_built_filter)  # moyenne glissante sur une fenêtre de taille window_size
        
        dpe_data_df['y_diff_moyenne'] = (dpe_data_df.nb_observations - dpe_data_df.nb_obs_moyenne) 
        dpe_data_df['y_diff_moyenne_abs'] = dpe_data_df['y_diff_moyenne'].abs()  # Rq : à ce stade, le bunching n'est pas encore normalisé

        
        bunching_df = pd.DataFrame(index=[0]) # initialisation d'un DataFrame
        
        for seuil, valeur_seuil in etiquette_ep_seuils.items():
            # Création d'un DataFrame filtré sur l'intervalle A GAUCHE du seuil, de largeur {itv_bunching}
            dpe_data_df_filtered = dpe_data_df[(dpe_data_df.conso_5_usages_ep_m2_arrondie > valeur_seuil-itv_bunching) & (dpe_data_df.conso_5_usages_ep_m2_arrondie <= valeur_seuil + itv_bunching)]
            
            nb_dpe_itv = dpe_data_df_filtered.nb_observations.sum() # nombre total de DPE dans l'intervalle à +-{itv_bunching} du seuil
            
            # Ajout d'une colonne correspondante au seuil dans le bunching DataFrame et normalisation 
            bunching_df[f'{seuil}_method_{method}'] = dpe_data_df_filtered['y_diff_moyenne_abs'].sum() / nb_dpe_itv
                
        if verbose:
            print(f"Bunching (méthode {method}, window_size {window_size} kWh) pour {departement}, sur l'intervalle de {itv_bunching} kWh/m2 à gauche de chaque seuils, old_built_filter = {old_built_filter} : \n", bunching_df)
            

            

    if method =='diff_moyenne_classes': # difference entre les données et la moyenne glissante, normalisé par le nombre de DPE dans les classes de part et d'autres du seuil étudié (nb_dpe_classes) 
        
        dpe_data_df = formatage_dpe_data(dep_code=dep_code, window_size=window_size, old_built_filter=old_built_filter)  # moyenne glissante sur une fenêtre de taille window_size
        
        dpe_data_df['y_diff_moyenne'] = (dpe_data_df.nb_observations - dpe_data_df.nb_obs_moyenne)
        dpe_data_df['y_diff_moyenne_abs'] = dpe_data_df['y_diff_moyenne'].abs() # Rq : à ce stade, le bunching n'est pas encore normalisé

        
        bunching_df = pd.DataFrame(index=[0]) # initialisation d'un DataFrame
        
        for seuil, valeur_seuil in etiquette_ep_seuils.items():
            # Création d'un DataFrame filtré sur l'intervalle  +-{itv_bunching} du seuil
            dpe_data_df_filtered = dpe_data_df[(dpe_data_df.conso_5_usages_ep_m2_arrondie > valeur_seuil-itv_bunching) & (dpe_data_df.conso_5_usages_ep_m2_arrondie <= valeur_seuil + itv_bunching)]
            
            # TODO : a généraliser ? (CAD ?)
            borne_inf_classe_gauche = etiquette_ep_dict.get(seuil.split('/')[0])[0]
            borne_sup_classe_droite = etiquette_ep_dict.get(seuil.split('/')[1])[1]
            nb_dpe_classes = dpe_data_df[(dpe_data_df.conso_5_usages_ep_m2_arrondie > borne_inf_classe_gauche) & (dpe_data_df.conso_5_usages_ep_m2_arrondie <= borne_sup_classe_droite)].nb_observations.sum()
            
            # Ajout d'une colonne correspondante au seuil dans le bunching DataFrame et normalisation
            bunching_df[f'{seuil}_method_{method}'] = dpe_data_df_filtered['y_diff_moyenne_abs'].sum() /nb_dpe_classes 
                
        if verbose:
            print(f'Bunching (méthode {method}, window_size {window_size} kWh) pour {departement}, avec un intervalle de +-{itv_bunching} kWh/m2 autour des seuils, old_built_filter = {old_built_filter} : \n', bunching_df)



    if method =='diff_moyenne_gauche_itv': # difference entre les données et la moyenne glissante, normalisé par le nombre de DPE dans l'intervalle itv_bunching A GAUCHE du seuil étudié (nb_dpe_itv) 
        
        dpe_data_df = formatage_dpe_data(dep_code=dep_code, window_size=window_size, old_built_filter=old_built_filter)  # moyenne glissante sur une fenêtre de taille window_size
        
        dpe_data_df['y_diff_moyenne'] = (dpe_data_df.nb_observations - dpe_data_df.nb_obs_moyenne) 
        dpe_data_df['y_diff_moyenne_abs'] = dpe_data_df['y_diff_moyenne'].abs()  # Rq : à ce stade, le bunching n'est pas encore normalisé

        
        bunching_df = pd.DataFrame(index=[0]) # initialisation d'un DataFrame
        
        for seuil, valeur_seuil in etiquette_ep_seuils.items():
            # Création d'un DataFrame filtré sur l'intervalle A GAUCHE du seuil, de largeur {itv_bunching}
            dpe_data_df_filtered = dpe_data_df[(dpe_data_df.conso_5_usages_ep_m2_arrondie > valeur_seuil-itv_bunching) & (dpe_data_df.conso_5_usages_ep_m2_arrondie <= valeur_seuil)]
            
            nb_dpe_itv_gauche = dpe_data_df_filtered.nb_observations.sum() # nombre total de DPE dans l'intervalle A GAUCHE du seuil
            
            # Ajout d'une colonne correspondante au seuil dans le bunching DataFrame et normalisation 
            bunching_df[f'{seuil}_method_{method}'] = dpe_data_df_filtered['y_diff_moyenne_abs'].sum() / nb_dpe_itv_gauche
                
        if verbose:
            print(f"Bunching (méthode {method}, window_size {window_size} kWh) pour {departement}, sur l'intervalle de {itv_bunching} kWh/m2 à gauche de chaque seuils, old_built_filter = {old_built_filter} : \n", bunching_df)
            

    
    return bunching_df
            
    
    
# =============================================================================
# OLD VERSION AVEC BETA.FIT ET PAS DE DATAFRAME

#     if method == 'diff_beta_old': 
#         
#         pdf = fit_dpe_data(dep_code, method='beta.fit')
#         difference = list(counter_dict_sorted.values()) - pdf*nb_dpe
#         difference_dict =  dict(zip(counter_dict_sorted.keys(), difference)) # dictionnaire qui lie l'écart au beta.fit des DPE à leur conso annuelle d'ep associée
#         
#         # Calcul du bunching
#         # méthode part excessive standardisée (Aja et al.) -> "part des DPE qui sont excessifs sur l’intervalle de 10 kWh de consommation d’énergie à gauche de chaque seuil"
#         for seuil, valeur_seuil in etiquette_ep_seuils.items():
#             part_excess = sum([float(v) for k,v in difference_dict.items() if k > valeur_seuil-itv_bunching and k <= valeur_seuil])  # omme des DPEs excessifs à gauche du seuil
#             part_excess = part_excess/nb_dpe   # normalisation pour obtenir la proportion des DPEs qui seraient excessifs
#             part_excess = round(part_excess,3)
#             bunching[seuil] = part_excess
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
# =============================================================================

#%%


def calcul_bunching_france(path, method, itv_bunching, window_size, old_built_filter, max_xlim = 600, verbose=False, force=False):
    """
    Calcul le bunching à chaque seuil sur l'ensemble des départements de l'hexagone pour une méthode au choix.

    Parameters
    ----------
    path : str
        Chemin de sauvegarde des FIGURES issues de calcul_bunching (écart données/fit) (et pas du dictionnaire bunching). N'est pas utile en pratique car par défaut, plot_ecart=False.
    method : str 
        nom de la méthode utilisée pour calculer le bunching.
        ('diff_simple' ou 'diff_simple_nb_dpe' 
        ou 'diff_beta_gauche' ou 'diff_beta' ou 'diff_beta_itv' 
        ou 'diff_moyenne' ou 'diff_moyenne_itv' ou 'diff_moyenne_classes' ou 'diff_moyenne_gauche_itv')
    itv_bunching : int
        Attention : l'intervalle peut être soit à gauche du seuil, soit de part et d'autre du seuil selon les méthodes. 
        Methodes 'diff_simple' et 'diff_simple_nb_dpe' : taille de l'intervalle de part et d'autre des seuils. 
        Methode 'diff_beta_gauche' : taille de l'intervalle à gauche de chaque seuil (utiliser plutôt 10 kWh/m2 comme Aja et al.).
        Methodes 'diff_beta' ou 'diff_beta_itv': taille de l'intervalle de part et d'autre des seuils.
        Methodes 'diff_moyenne', 'diff_moyenne_itv' et 'diff_moyenne_classes' : taille de l'intervalle de part et d'autre des seuils
        Methode 'diff_moyenne_gauche_itv' : taille de l'intervalle à gauche de chaque seuil.
     window_size : int
        taille de la fenêtre de glissement pour le rolling (moyenne glissante). Utile uniquement pour la méthode 'diff_moyenne'.
    old_built_filter : boolean
        filtrage : on ne garde que les logements construits avant 1974.
    max_xlim : int, optional
        limite du graphe. The default is 600.    
    verbose : boolean, optional
        affiche les print() dans la console. The default is False.

    Returns
    -------
    france_bunching : pandas DataFrame
        Bunching pour chaque seuil pour chaque département, avec la méthode et l'itv_bunching spécifié
        Colonnes :  dep_code  |  A/B_method  |  B/C_method  |  C/D_method  |  D/E_method  |  E/F_method  |  F/G_method
    """    

    # Définition du chemin de sauvegarde des bunchings en .csv
    output_folder_bunching = os.path.join('output', 'buching')
    os.makedirs(output_folder_bunching, exist_ok=True)
    existing_files = os.listdir(output_folder_bunching)
    
    # Définition du nom du fichier final
    save_name = f'france_bunching_method_{method}_itv_bunching_{itv_bunching}_kWh.csv'
    if old_built_filter:
        save_name = save_name.replace('.csv','_old_built.csv')
    
    if save_name not in existing_files or force:
        
        france = France()
        
        # Initialisation du dictionnaire de bunching avec des listes vides
        dict_france_bunching = {'dep_code':[]}
        for k in etiquette_ep_seuils.keys():
            dict_france_bunching[f'{k}_method_{method}'] = []
        
        # Implémentation du dictionnaire grâce à calcul_bunching 
        for dep in tqdm.tqdm(france.departements) :
            dep_code = dep.code
            # print(dep)
            bunching_dep = calcul_bunching(dep_code, method, itv_bunching, window_size, plot_ecart=False, path=path, old_built_filter=old_built_filter, max_xlim=max_xlim) # calcul du bunching de chaque département
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



def cut_france_bunching(france_bunching, seuils):
    """
    Rogne le DataFrame france_bunching pour ne conserver que les colonnes correspondant aux seuils demandés.

    Parameters
    ----------
    france_bunching : panda DataFrame
        dataframe du bunching pour chaque seuil pour chaque département, issu de calcul_bunching_france().
    seuils : list
        liste des colonnes seuils à conserver dans france_bunching_cut.

    Returns
    -------
    france_bunching_cut : panda DataFrame
        DataFrame france_bunching rogné.
    seuils_sans_slash : str
        chaîne de caractère simplifiée pour identifier les seuils.
    """
        
    # formatage d'une chaîne de caractère simplifiée pour identifier les seuils
    seuils_sans_slash = '_'.join(seuils)
    seuils_sans_slash = seuils_sans_slash.replace("/","")
    
    # implémentation d'une liste du nom exact des colonnes de france_bunching à sélectionner
    noms_colonnes_seuils = []
    for seuil in seuils:
        colonne_seuil = [colonne for colonne in france_bunching.columns if colonne.startswith(f'{seuil}')]
        nom_colonne = colonne_seuil[0]
        noms_colonnes_seuils.append(nom_colonne)

    france_bunching_cut = france_bunching.filter(items = noms_colonnes_seuils)  
    
    return france_bunching_cut, seuils_sans_slash
    
    
  
    
#%% 
    
    
def calcul_nb_dpe_filtre(old_built_filter):
    '''
    Calcule le nb de DPE total utilisé pour le curve_fit (donc après filtrage), dans chaque département.
    
    Parameters
    ----------
    old_built_filter : boolean
        filtrage : on ne garde que les logements construits avant 1974.

    Returns
    -------
    dict_dep_nb_dpe_filtre : dictionnaire (compatible avec draw_departement_map)
        {Ain (01) (Departement) : nombre de DPE (np.int), 
         etc...}
    '''
    
    france = France()
    dict_dep_nb_dpe_filtre = {d:0. for d in france.departements} 
    
    print('Calcul du nombre de DPE par département...')
    for dep in tqdm.tqdm(france.departements) :
        dep_code = dep.code
        # print(dep)
        _, _, _, nb_dpe_filtre = fit_dpe_data(dep_code, method='curve_fit', old_built_filter=old_built_filter, verbose=False)  # attention : si on utilisé méthode diff_simple ou diff_moyenne, pas forcément pertinent de filtrer
        dict_dep_nb_dpe_filtre[dep] = nb_dpe_filtre 
    
    return dict_dep_nb_dpe_filtre 




def get_nb_dpe(old_built_filter=True,seuils=['D/E', 'E/F', 'F/G']):
    '''
    Calcule le nb de DPE total dans l'ensemble des classes de la liste seuil (donc à partir du plus petit seuil).
    
    Parameters
    ----------
    old_built_filter : boolean
        filtrage : on ne garde que les logements construits avant 1974.

    Returns
    -------
    dict_dep_nb_dpe_filtre : dictionnaire (compatible avec draw_departement_map)
        {Ain (01) (Departement) : nombre de DPE dans l'ensemble des classes de seuils (np.int), 
         etc...}
    '''
    
    france = France()
    dict_dep_nb_dpe_filtre = {d:0. for d in france.departements} 
    
    for dep in france.departements:
        dep_code = dep.code
        
        dpe_data = get_dpe_consumption(dep_code, old_built_filter=old_built_filter) # prend du temps
        dpe_data = dpe_data.dropna()
        dpe_data = dpe_data.map(round)
        counter_dict = dict(dpe_data.conso_5_usages_ep_m2.value_counts())
        
        min_seuil = np.inf
        for s in seuils :
            min_seuil = min(min_seuil,etiquette_ep_dict.get(s.split('/')[0])[0])
            
        nb_dpe = sum([nb for s,nb in counter_dict.items() if s>=min_seuil])
        dict_dep_nb_dpe_filtre[dep] = nb_dpe 
        
    return dict_dep_nb_dpe_filtre 
        
    
        
        

#%%


def df_compare_methods(seuils, path, old_built_filter, itv_bunching=10, force=False): # todo : changer nb_dpe_filtre
    """
    Création d'un DataFrame du bunching selon l'ensemble des méthodes, en sommant sur les seuils de la liste seuils.
    ATTENTION ! La valeur de window_size a été fixée à 50 kWh pour le calcul de la moyenne glissante

    Parameters
    ----------
    seuils : list
        liste des seuils à prendre en compte pour le calcul de la somme du bunching.
    path : str
        Chemin de sauvegarde des FIGURES issues de calcul_bunching (écart données/fit) (et pas du dictionnaire bunching). N'est pas utile en pratique car par défaut, plot_ecart=False.
    old_built_filter : boolean
        filtrage : on ne garde que les logements construits avant 1974.
    itv_bunching : int
        Attention : l'intervalle peut être soit à gauche du seuil, soit de part et d'autre du seuil selon les méthodes. 

    Returns
    -------
    df_compare : pandas DataFrame
        Bunching par départements selon différentes méthodes de calcul
        Colonnes : dep_code  |  Département  |  nb_dpe_filtre  |  Méthode diff_simple  |  Méthode diff_simple_nb_dpe  |  Méthode diff_beta  |  Méthode diff_moyenne  |  Méthode diff_moyenne_classes  |  Méthode diff_moyenne_gauche_itv  |  Méthode diff_beta_gauche
    """
    
    # Définition du chemin de sauvegarde des df_compare en .csv
    output_folder_bunching = os.path.join('output', '0. df_compare')
    os.makedirs(output_folder_bunching, exist_ok=True)
    existing_files = os.listdir(output_folder_bunching)
    
    # formatage d'une chaîne de caractère simplifiée pour identifier les seuils
    seuils_sans_slash = '_'.join(seuils)
    seuils_sans_slash = seuils_sans_slash.replace("/","")
    
    # Définition du nom du fichier final
    save_name = f'df_compare_seuils_{seuils_sans_slash}_itv_{itv_bunching}_kWh.csv'  
    if old_built_filter:
        save_name = save_name.replace('.csv','_old_built.csv')
    
    list_methods = ['diff_simple',
               'diff_simple_nb_dpe',
               'diff_beta_gauche',
               'diff_beta',
               'diff_beta_itv',
               'diff_moyenne',
               'diff_moyenne_itv',
               'diff_moyenne_gauche_itv',
               #'diff_moyenne_classes' 
               ]
    
    if save_name not in existing_files or force:
    
        # Initialisation du DataFrame de comparaison des méthodes
        df_compare = pd.DataFrame(index=list_dep_code) 
        df_compare.index.name = 'dep_code'
        df_compare['département'] = [Departement(f'{n}') for n in df_compare.index]
        
        # Ajout d'une colonne nombre de DPE 
        # todo : ne pas forcément filtrer le nb_DPE ? utile que pour méthode diff_beta --> mettre simplement nb_dpe ?
        dict_dep_nb_dpe_filtre = calcul_nb_dpe_filtre(old_built_filter)
        df_compare['nb_dpe_filtre'] = dict_dep_nb_dpe_filtre.values()
        
        # Ajout d'une colonne pour chacune des méthodes
        for method in list_methods:  
            print(f'Calcul du bunching pour la méthode {method}...')
            france_bunching = calcul_bunching_france(path, method=method, itv_bunching=itv_bunching, window_size = 50, old_built_filter=old_built_filter, max_xlim=600)
            france_bunching_cut, seuils_sans_slash = cut_france_bunching(france_bunching, seuils)
            df_compare[f'Méthode {method}'] = france_bunching_cut.sum(axis=1) # on somme sur les lignes des colonnes conservées


        # Enregistrement du DataFrame du bunching en .csv
        df_compare.to_csv(os.path.join(output_folder_bunching, save_name))
        
    else:
        df_compare = pd.read_csv(os.path.join(output_folder_bunching, save_name), index_col='dep_code') 
    
    return df_compare
   
   
   
   
        
#%% ===========================================================================wwwwwwwwwwwwwwwwwwww
# SCRIPT PRINCIPAL
# =============================================================================




def main():
    tic = time.time()
    
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')
    output_folder = os.path.join('output',today)
    os.makedirs(output_folder, exist_ok=True)
    
    dep = Departement('85')
    old_built_filter = True
    window_size = 50  # fenêtre de la moyenne glissante (rolling de la méthode 'diff_moyenne')
    
    
    # DISTRIBUTION DES DPE
    
    # tracé de la distribution des dpe du département
    if False:
        #dpe_data = get_dpe_consumption(dep.code)
        plot_dpe_distribution(output_folder,dep.code, save=True, plot_mean=False, plot_median=False, window_size = window_size, plot_curve_fit=True, old_built_filter=old_built_filter, max_xlim=600)
    
    
        
    # CALCUL DU BUNCHING
        
    # choix des paramètres de mesure du bunching
    # method='diff_beta'
    # method='diff_beta_itv'
    # method = 'diff_simple_nb_dpe'
    method = 'diff_simple'
    # itv_bunching = 5
    itv_bunching = 10

    
    
    # calcul du bunching du département
    if False:
        bunching_dep = calcul_bunching(dep.code, method=method, itv_bunching=itv_bunching, window_size=window_size, plot_ecart = True, path=output_folder, old_built_filter=old_built_filter, verbose=True)
        
        # Affichage somme du bunching sur l'ensemble des seuils
        bunching_dep_sum = bunching_dep.sum(axis=1) # on somme le bunching de l'ensemble des 6 seuils 
        print('bunching_dep_sum =', bunching_dep_sum.iloc[0])
                
        # Affichage somme du bunching des seuils D/E à F/G
        seuils = ['D/E', 'E/F', 'F/G']
        bunching_dep_cut, seuils_sans_slash = cut_france_bunching(bunching_dep, seuils)
        bunching_dep_sum = bunching_dep_cut.sum(axis=1)
        print(f'bunching_dep_sum_{seuils_sans_slash} =', bunching_dep_sum.iloc[0])
        
        
        
        
    # CARTES DU BUNCHING 

      
    if False:
        today = pd.Timestamp(date.today()).strftime('%Y%m%d')
        output_folder = os.path.join('output',today)
        os.makedirs(output_folder, exist_ok=True)
        
        france_bunching = calcul_bunching_france(output_folder, method=method, itv_bunching=itv_bunching, window_size=window_size, old_built_filter=old_built_filter, max_xlim=600,force=True)
        
                
        # Somme sur l'ensemble des 6 seuils
        if False: 
            france_bunching[f'Somme_method_{method}'] = france_bunching.sum(axis=1) # on somme sur les lignes et non les colonnes
            dict_dep_bunching = {Departement(dep_code):bunching_somme for dep_code, bunching_somme in zip(france_bunching.index, france_bunching[f'Somme_method_{method}'])}
            
            if old_built_filter:
                save=f"Carte somme du bunching sur l'ensemble des seuils (Méthode {method}, intervalle bunching = {itv_bunching} kWh.m-2, anciens logements)"
                map_title=f"Somme du bunching sur l'ensemble des seuils\n(Méthode {method}, intervalle bunching = {itv_bunching}"" kWh.m$^{-2}$, anciens logements)"
            else :
                save=f"Carte somme du bunching sur l'ensemble des seuils (Méthode {method}, intervalle bunching = {itv_bunching} kWh.m-2)"
                map_title=f"Somme du bunching sur l'ensemble des seuils\n(Méthode {method}, intervalle bunching = {itv_bunching}"" kWh.m$^{-2}$)"
        
            draw_departement_map(dict_dep_bunching,output_folder,save=save, map_title=map_title)
        
                    
        # Moyenne des methodes
        if False: 
            france_bunching[f'Moyenne_method_{method}'] = france_bunching.mean(axis=1)
            dict_dep_bunching = {Departement(dep_code):bunching_mean for dep_code, bunching_mean in zip(france_bunching.index, france_bunching[f'Moyenne_method_{method}'])}
            
            if old_built_filter:
                save=f"Carte moyenne du bunching sur l'ensemble des seuils (Méthode {method}, intervalle bunching = {itv_bunching} kWh.m-2, anciens logements)"
                map_title=f"Moyenne du bunching sur l'ensemble des seuils\n(Méthode {method}, intervalle bunching = {itv_bunching}"" kWh.m$^{-2}$, anciens logements)"
            else :
                save=f"Carte moyenne du bunching sur l'ensemble des seuils (Méthode {method}, intervalle bunching = {itv_bunching} kWh.m-2)"
                map_title=f"Moyenne du bunching sur l'ensemble des seuils\n(Méthode {method}, intervalle bunching = {itv_bunching}"" kWh.m$^{-2}$)"
        
            draw_departement_map(dict_dep_bunching,output_folder,save=save, map_title=map_title)


        # Bunching sur un seuil seulement
        if False: 
            seuil = 'B/C'
            seuil_sans_slash = seuil.replace("/","")
            
            colonne_seuil = [colonne for colonne in france_bunching.columns if colonne.startswith(f'{seuil}')]
            nom_colonne = colonne_seuil[0]
            dict_dep_bunching = {Departement(dep_code):bunching_EF for dep_code, bunching_EF in zip(france_bunching.index, france_bunching[nom_colonne])}
   
            if old_built_filter:
                save=f"Carte du bunching au seuil {seuil_sans_slash} (Méthode {method}, intervalle bunching = {itv_bunching} kWh.m-2, anciens logements)"
                map_title=f"Bunching au seuil {seuil}\n(Méthode {method}, intervalle bunching = {itv_bunching}"" kWh.m$^{-2}$, anciens logements)"
            else :
                save=f"Carte du bunching au seuil {seuil_sans_slash} (Méthode {method}, intervalle bunching = {itv_bunching} kWh.m-2)"
                map_title=f"Bunching au seuil {seuil}\n(Méthode {method}, intervalle bunching = {itv_bunching}"" kWh.m$^{-2}$)"

            draw_departement_map(dict_dep_bunching,output_folder,save=save, map_title=map_title)


        # Somme sur une sélection de seuils seulement
        if True:
            seuils = ['D/E', 'E/F', 'F/G']
            
            france_bunching_cut, seuils_sans_slash = cut_france_bunching(france_bunching, seuils)
            france_bunching[f'Somme_seuils_{seuils_sans_slash}_method_{method}'] = france_bunching_cut.sum(axis=1) # on somme sur les lignes des colonnes conservées
            print('france_bunching :\n', france_bunching)
            dict_dep_bunching = {Departement(dep_code):bunching_somme for dep_code, bunching_somme in zip(france_bunching.index, france_bunching[f'Somme_seuils_{seuils_sans_slash}_method_{method}'])}

            if old_built_filter:
                save=f"Carte somme du bunching aux seuils {seuils_sans_slash} (Méthode {method}, intervalle bunching = {itv_bunching} kWh.m-2, anciens logements)"
                # map_title=f"Somme du bunching aux seuils {seuils}\n(Méthode {method}, intervalle bunching = {itv_bunching}"" kWh.m$^{-2}$, anciens logements)"
                map_title=f"Méthode {method} (construction avant 1974)"
            else :
                save=f"Carte somme du bunching aux seuils {seuils_sans_slash} (Méthode {method}, intervalle bunching = {itv_bunching} kWh.m-2)"
                # map_title=f"Somme du bunching aux seuils {seuils}\n(Méthode {method}, intervalle bunching = {itv_bunching}"" kWh.m$^{-2}$)"
                map_title=f"Méthode {method}"
        
            draw_departement_map(dict_dep_bunching,output_folder,save=save) #, map_title=map_title)
            plt.show()
            plt.close()
            
            # Regplot entre bunching et nb_dpe_filtre
            if True: # todo : creer une fonction qui fait ça cut_france
                
                df_bunching = pd.DataFrame().from_dict(dict_dep_bunching, orient='index', columns=[f'Somme_seuils_{seuils_sans_slash}_method_{method}'])
                
                # dict_dep_nb_dpe_filtre=calcul_nb_dpe_filtre(old_built_filter)
                # df_nb_dpe_filtre = pd.DataFrame().from_dict(dict_dep_nb_dpe_filtre, orient='index', columns=['nb_dpe_filtre'])
                
                df_nb_dpe = pd.DataFrame().from_dict(get_nb_dpe(old_built_filter,['A/B']), orient='index', columns=['nb_dpe'])
                # df_nb_dpe_seuils = pd.DataFrame().from_dict(get_nb_dpe(old_built_filter,seuils), orient='index', columns=['nb_dpe_seuils'])
                
    
                # Création d'une colonne dep_code commune aux deux df afin de pouvoir merge
                df_bunching['dep_code'] = [dep.code for dep in df_bunching.index]  # Extrait le code du département
                #df_bunching = df_bunching.reset_index(drop=True)  # Réinitialise l'index
                
                df_nb_dpe['dep_code'] = [dep.code for dep in df_nb_dpe.index]  # Extrait le code du département
                # df_nb_dpe_seuils['dep_code'] = [dep.code for dep in df_nb_dpe_seuils.index]
                
    
                df_bunching = df_bunching.merge(df_nb_dpe, on='dep_code')
                # df_bunching = df_bunching.merge(df_nb_dpe_seuils, on='dep_code')
                df_bunching = df_bunching.set_index("dep_code")  # Réinitialise l'index
                
                # df_bunching[f'Somme_seuils_{seuils_sans_slash}_method_{method}_nb_seuils_calibrated'] = df_bunching[f'Somme_seuils_{seuils_sans_slash}_method_{method}'] * df_bunching.nb_dpe / df_bunching.nb_dpe_seuils
    
    
                #sns.set()  
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)                  
                sns.regplot(data=df_bunching, x="nb_dpe", y=f'Somme_seuils_{seuils_sans_slash}_method_{method}',ax=ax)
                ax.set_ylabel(f'Somme_seuils_{seuils_sans_slash}_method_{method}', fontsize=10)
                ax.set_title(f"Corrélation entre le nombre de DPE et\nle bunching aux seuils {seuils_sans_slash}, méthode {method}")
                # p.set_ylim(0.01, 0.04)
                plt.show()
                plt.close()
                
                # fig,ax = plt.subplots(figsize=(5,5),dpi=300)                  
                # sns.regplot(data=df_bunching, x="nb_dpe", y=f'Somme_seuils_{seuils_sans_slash}_method_{method}_nb_seuils_calibrated',ax=ax)
                # ax.set_ylabel(f'Somme_seuils_{seuils_sans_slash}_method_{method}_nb_seuils_calibrated', fontsize=10)
                # ax.set_title(f"Corrélation entre le nombre de DPE et\nle bunching aux seuils {seuils_sans_slash}, méthode {method}")
                # # p.set_ylim(0.01, 0.04)
                # plt.show()
                # plt.close()
                
                df_bunching['inv_nb_dpe'] = 1/df_bunching.nb_dpe
                
                # # 1/nb_dpe_filtre
                # fig,ax = plt.subplots(figsize=(5,5),dpi=300)                  
                # sns.regplot(data=df_bunching, x="inv_nb_dpe_filtre", y=f'Somme_seuils_{seuils_sans_slash}_method_{method}',ax=ax)
                # ax.set_ylabel(f'Somme_seuils_{seuils_sans_slash}_method_{method}', fontsize=10)
                # ax.set_title(f"Corrélation entre l'inverse du nombre de DPE et\nle bunching aux seuils {seuils_sans_slash}, méthode {method}")
                # ax.set_xlim([0., 1e-4])


        #france_bunching = france_bunching_method1.join(france_bunching_method2)
        # pour pouvoir faire des stats sur l'ensemble des methodes
        
    
    
    
    # CARTE DU NB_DPE_FILTRE

    if False :
        today = pd.Timestamp(date.today()).strftime('%Y%m%d')
        output_folder = os.path.join('output',today)
        os.makedirs(output_folder, exist_ok=True)
   
        draw_departement_map(calcul_nb_dpe_filtre(old_built_filter),output_folder,save="Carte du nombre de DPE sur lesquels on fit une distribution beta (curve_fit)", map_title="Carte du nombre de DPE sur lesquels on fit une distribution beta (curve_fit)") 
        
        
        
    # CALCUL DE DF_COMPARE POUR L'ENSEMBLE DES MÉTHODES (prend un peu de temps)
    
    if False:
        seuils = ['D/E', 'E/F', 'F/G']
        itv_bunching = 10   # attention : on a le même itv_bunching pour toutes les méthodes ! (même si gauche ou centré)

        df_compare = df_compare_methods(seuils, output_folder, old_built_filter=old_built_filter, itv_bunching=itv_bunching, force=True)        

        
        
    # SEABORN COMPARAISON METHODES

    
    if False: # corrélation entre deux méthodes
    
        seuils = ['D/E', 'E/F', 'F/G']
        method_1 = 'diff_moyenne' 
        method_2 = 'diff_beta'
        itv_bunching = 10   # attention : on a le même itv_bunching pour toutes les méthodes ! (même si gauche ou centré)

        # formatage d'une chaîne de caractère simplifiée pour identifier les seuils
        seuils_sans_slash = '_'.join(seuils)
        seuils_sans_slash = seuils_sans_slash.replace("/","")

        df_compare = df_compare_methods(seuils, output_folder, old_built_filter=old_built_filter, itv_bunching=itv_bunching)        

        fig,ax = plt.subplots(figsize=(5,5),dpi=300)                  
        sns.regplot(data=df_compare, x=f"Méthode {method_1}", y=f"Méthode {method_2}",ax=ax)
        ax.set_xlabel(f'Méthode {methods_dict[method_1]}', fontsize = 12)
        ax.set_ylabel(f'Méthode {methods_dict[method_2]}', fontsize = 12)

        # calcul du coefficient de corrélation (r) et de la valeur-p
        correlation, p_value = pearsonr(df_compare[f"Méthode {method_1}"], df_compare[f"Méthode {method_2}"])
        ax.text(
            0.05, 0.95, 
            f"r = {correlation:.2f}\np = {p_value:.2e}", 
            transform=ax.transAxes, 
            fontsize=12, 
            verticalalignment='top'
            )
        
        if old_built_filter:
            save_path = os.path.join(output_folder,f'Correlation_methodes_{method_1}_et_{method_2}_seuils_{seuils_sans_slash}_old_built.png') 
        else: 
            save_path = os.path.join(output_folder,f'Correlation_methodes_{method_1}_et_{method_2}_seuils_{seuils_sans_slash}.png') 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        
    if False: # corrélation d'un groupe de méthodes avec pairplot
        seuils = ['D/E', 'E/F', 'F/G']
        list_methods = ['diff_simple_nb_dpe', 'diff_moyenne', 'diff_beta']
        # seuils = ['E/F']
        itv_bunching = 10   # attention : on a le même itv_bunching pour toutes les méthodes !

    
        # formatage d'une chaîne de caractère simplifiée pour identifier les seuils
        seuils_sans_slash = '_'.join(seuils)
        seuils_sans_slash = seuils_sans_slash.replace("/","")
        
        df_compare = df_compare_methods(seuils, output_folder, old_built_filter=old_built_filter, itv_bunching=itv_bunching)        
        
        data = df_compare[[f'Méthode {method}' for method in list_methods]]
        p = sns.pairplot(data=data, kind='reg', plot_kws={'marker': '+'}, diag_kind='kde') #, hue='nb_dpe_filtre')    
        # p.fig.suptitle(f"Corrélation entre les différentes méthodes de mesure du bunching\naux seuils {seuils_sans_slash}, old_built_filter = {old_built_filter}")
        # p.fig.subplots_adjust(top=0.92)
        
        # Fonction qui affiche r, coef de corrélation de pearson, et la p-value
        def corrfunc(x, y, **kws):
            r, p = pearsonr(x, y)
            ax = plt.gca()
            ax.annotate(
                f"r = {r:.2f}\np = {p:.2e}",
                xy=(0.05, 0.85),
                xycoords=ax.transAxes,
                fontsize=10
            )
            
        p.map_upper(corrfunc)

        if old_built_filter:
            save_path = os.path.join(output_folder,f'Pairplot_correlation_methodes_{list_methods}_seuils_{seuils_sans_slash}_old_built.png') 
        else: 
            save_path = os.path.join(output_folder,f'Pairplot_correlation_methodes_{list_methods}_seuils_{seuils_sans_slash}.png') 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
        plt.show()
        # plt.close()
        
        
        
    # COMPARAISON METHODES BUNCHING ET GAIN MOYEN ETIQUETTE 
    # todo: a déplacer dans manipulation_dpe.py ?
    
    if False : 
        seuils = ['D/E', 'E/F', 'F/G']
        list_methods = ['diff_simple_nb_dpe', 'diff_moyenne', 'diff_beta']
        itv_bunching = 10
        period = 20
        # attention : dcp là on a le même itv_bunching pour toutes les méthodes !!
    
        # formatage d'une chaîne de caractère simplifiée pour identifier les seuils
        seuils_sans_slash = '_'.join(seuils)
        seuils_sans_slash = seuils_sans_slash.replace("/","")
        
        df_compare = df_compare_methods(seuils, output_folder, old_built_filter=old_built_filter, itv_bunching=itv_bunching)        
        
        dict_part_dpe_stables, dict_gain_moyen_etiquette, dict_gain_moyen_etiquette_parmi_modif = dicts_dep_gain_moyen_etiquette(period)
        df_compare['gain_moyen_etiquette'] = dict_gain_moyen_etiquette.values()
        df_compare['part_dpe_stables'] = dict_part_dpe_stables.values()  # todo : comment etre sure que les departements sont bien alignés ?
         
        data = df_compare[[f'Méthode {method}' for method in list_methods]+['gain_moyen_etiquette','part_dpe_stables']]
    
        p = sns.pairplot(data=data, kind='reg', plot_kws={'marker': '+'}, diag_kind='kde')#, hue='nb_dpe_filtre')    
        # p.fig.suptitle(f"Corrélation entre les différentes méthodes de mesure du bunching\naux seuils {seuils_sans_slash}, old_built_filter = {old_built_filter}")
        # p.fig.subplots_adjust(top=0.92)
        
        # Fonction qui affiche r, coef de corrélation de pearson, et la p-value
        def corrfunc(x, y, **kws):
            r, p = pearsonr(x, y)
            ax = plt.gca()
            ax.annotate(
                f"r = {r:.2f}\np = {p:.2e}",
                xy=(0.05, 0.85),
                xycoords=ax.transAxes,
                fontsize=10
            )
            
        p.map_upper(corrfunc)
    
    
        if old_built_filter:
            save_path = os.path.join(output_folder,f'Pairplot_correlation_methodes_{list_methods}_seuils_{seuils_sans_slash}_old_built.png') 
        else: 
            save_path = os.path.join(output_folder,f'Pairplot_correlation_methodes_{list_methods}_seuils_{seuils_sans_slash}.png') 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
        plt.show()
        # plt.close()
      
        
        
    # TEST CLASSE GES VS CLASSE ENERGIE
    
    if False:
        
        # france = France()
        
        # for dep in france.departements :
            # dep_code = dep.code
            dep_code = '75'
            dpe_data, _ , _ = get_bdnb(dep_code)
            dpe_data = dpe_data[dpe_data.type_dpe=='dpe arrêté 2021 3cl logement'][['conso_5_usages_ep_m2','periode_construction_dpe','surface_habitable_logement','date_etablissement_dpe', 'classe_bilan_dpe', 'classe_emission_ges']].compute() 
            dpe_data = dpe_data.dropna()
            #dpe_data = dpe_data.map(round)
            
            print('nb_tot_dpe : ', len(dpe_data))
            
            # Création des bins et des labels pour pandas.cut
            bins = [0, 70, 110, 180, 250, 330, 420, np.inf]
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
            
            dpe_data['classe_energie_calcul'] = pd.cut(dpe_data['conso_5_usages_ep_m2'], bins=bins,labels=labels, include_lowest=True)
            
            # nb de logements dont le DPE ne correspond ni à l'étiquette énergie, ni à l'étiquette ges
            dpe_data_cut_1 = dpe_data[(dpe_data['classe_energie_calcul'] != dpe_data['classe_bilan_dpe']) & (dpe_data['classe_emission_ges'] != dpe_data['classe_bilan_dpe'])]
            
            nb_dpe_bizarres = len(dpe_data_cut_1)
            print("Nombre de DPE qui ne correspondent ni à l'étiquette énergie, ni à l'étiquette GES :", nb_dpe_bizarres)
            
            
            # nb de logements dont la classe GES > classe énergie --> c'est la classe GES qui limite le DPE global
            dpe_data_cut_2 = dpe_data[dpe_data['classe_emission_ges'] > dpe_data['classe_energie_calcul']]
            
            nb_dpe_lim_ges = len(dpe_data_cut_2)
            print("Nombre de DPE limités par l'étiquette GES : ", nb_dpe_lim_ges)
            
            # vérification : nb_bugs doit être nul
            dpe_data_cut_2_bugs = dpe_data_cut_2[dpe_data_cut_2['classe_emission_ges'] != dpe_data_cut_2['classe_bilan_dpe']]
            print('nb_bugs : ', len(dpe_data_cut_2_bugs))
            
            # todo idée : tracer des pie-charts de répartition des logements par départements 

    

    # CALCUL PART DES DPE METHODE 3CL 2021 (prend 45 min)
    
    if True:
    
        france = France()
    
        n_total = 0
        n_filtre = 0
        
        for dep in france.departements :
            dep_code = dep.code
            dpe_data, _ , _ = get_bdnb(dep_code)
            n_total = n_total + len(dpe_data)
            n_filtre = n_filtre + len(dpe_data[dpe_data.type_dpe=='dpe arrêté 2021 3cl logement'])
        
        part_dpe_3CL = n_filtre / n_total
        print(n_total, n_filtre)
        print(part_dpe_3CL) # = 0.591

    


    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')

if __name__ == '__main__':
    main()
