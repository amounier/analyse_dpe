#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:23:12 2026

@author: audrey
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib.request import urlopen, Request
from datetime import date
from scipy.stats import zscore
import seaborn as sns
import plotly.io as pio
pio.renderers.default='browser'
import plotly.graph_objects as go
from pySankey.sankey import sankey
import json
import jsondiff as jd
from jsondiff import diff
import tqdm

from utils import etiquette_colors_dict, etiquette_ep_dict, etiquette_ep_seuils
from administrative import Departement, France, draw_departement_map
from download import get_bdnb
#from distribution import cut_france_bunching # ne peux pas marcher car distribution a aussi besoin de manipulation_dpe


def filter_bdnb_individual(dep_code, force):
    """
    Filtrage des DPE 2021 3CL associés à des logements individuels (maisons) et enregistrement en .csv pour ne pas avoir à compute à chaque fois. 

    Parameters
    ----------
    dep_code : str
        code du departement.
    force : boolean
        force le compute et le ré-enregistrement en .csv

    Returns
    -------
    bdnb_filter_individual : pandas DataFrame
        identifiants DPE des logements individuels, associés à leur identifiant bâtiment.
        Colonnes :  batiment_groupe_id  |  ffo_bat_nb_log  |  identifiant_dpe  |  adresse_brut  |  type_batiment_dpe  |  date_etablissement_dpe  |  classe_bilan_dpe  |  surface_habitable_logement
    """
    
    # Définition du chemin de sauvegarde du DataFrame en .csv
    output_folder = os.path.join('data', 'BDNB', 'filter_bdnb_individual')
    os.makedirs(output_folder, exist_ok=True)
    existing_files = os.listdir(output_folder)
    
    # Définition du nom du fichier final
    save_name = f'bdnb_filter_individual_dep{dep_code}'
    
    if save_name not in existing_files or force:

        bdnb_dpe_logement, bdnb_rel_batiment_groupe_dpe_logement, bdnb_batiment_groupe_compile = get_bdnb(dep_code)
    
        # Filtrage des bâtiments correspondants à des logements individuels 
        bdnb_batiment_groupe_compile = bdnb_batiment_groupe_compile[bdnb_batiment_groupe_compile.ffo_bat_nb_log == 1][['batiment_groupe_id','ffo_bat_nb_log']]
        bdnb_batiment_groupe_compile = bdnb_batiment_groupe_compile.compute() 
    
        bdnb_rel_batiment_groupe_dpe_logement = bdnb_rel_batiment_groupe_dpe_logement[['batiment_groupe_id','identifiant_dpe','adresse_brut']]
        bdnb_rel_batiment_groupe_dpe_logement = bdnb_rel_batiment_groupe_dpe_logement.compute()
        #doublons = bdnb_rel_batiment_groupe_dpe_logement[bdnb_rel_batiment_groupe_dpe_logement.duplicated(subset = ["identifiant_dpe"], keep=False)]
    
        # Filtrage des DPE arrêté 2021 méthode 3CL logement
        bdnb_dpe_logement = bdnb_dpe_logement[(bdnb_dpe_logement.type_dpe =='dpe arrêté 2021 3cl logement') & (bdnb_dpe_logement.type_batiment_dpe == 'maison')][['identifiant_dpe','type_batiment_dpe', 'date_etablissement_dpe', 'conso_5_usages_ep_m2', 'classe_bilan_dpe','surface_habitable_logement']]
        bdnb_dpe_logement = bdnb_dpe_logement.compute()
        bdnb_dpe_logement.dropna(inplace = True) # certains DPE n'ont pas de surface associée (2021) 
        
        # Jointure des différentes bases
        bdnb_join_id_dpe = bdnb_batiment_groupe_compile.merge(bdnb_rel_batiment_groupe_dpe_logement, how='inner', on='batiment_groupe_id') # 'inner' --> intersection des deux car bdnb_batiment_groupe_compile contient des id bâtiments sans DPE, et bdnb_rel_batiment_groupe_dpe_logement contient des id_DPE de logements collectifs
        
        bdnb_filter_individual = bdnb_join_id_dpe.merge(bdnb_dpe_logement, how='inner', on='identifiant_dpe') # intersection : on conserve seulement les DPE méthode 3CL 2021 correspondant à des logements individuels
        #doublons = bdnb_filter_individual[bdnb_filter_individual.duplicated(subset = ["identifiant_dpe"], keep=False)] # pour observer les id_dpe en doublons (car correspondants à plusieurs bâtiments à la fois)
        bdnb_filter_individual.drop_duplicates(subset = ["identifiant_dpe"], keep=False, inplace = True) # pour supprimer tous les id_dpe associés à plusieurs bâtiments à la fois
                
        # Enregistrement en csv
        bdnb_filter_individual.to_csv(os.path.join(output_folder, save_name), index = False) # index = False permet de ne pas enregistrer la colonne d'index, qui sera recréée automatiquement lors de read_csv

    
    else:
        bdnb_filter_individual = pd.read_csv(os.path.join(output_folder, save_name), parse_dates=['date_etablissement_dpe'], date_format='%Y-%m-%d %H:%M:%S')

            
    return bdnb_filter_individual


# %% SELECTION DES PAIRES DE DPE SUCESSIFS

def filter_manipulated(dep_code, period = 20): 
# todo : exclure DPE identiques fait le meme jour = doublons ? verifier que meme infos détaillées ou pas -> a priori il n'y a pas tant de doublons
    """
    Identification des bâtiments ayant calculés plusieurs DPE.
    
    Parameters
    ----------
    dep_code : str
        code du departement.
    period : int
        écart de temps maximal entre deux DPE successifs avant de considérer que des rénovations énergétiques ont pu avoir lieu.

    Returns
    -------
    df_epc_evolution : pandas DataFrame
        DataFrame des paires de DPE successifs pour chaque bâtiment du département.
        Colonnes :
            - batiment_groupe_id (index) : identifiant du batiment
            - first_epc_id : identifiant du premier DPE calculé
            - first_epc_surf : surface renseignée lors du calcul du 1er DPE
            - first_epc : classe du premier DPE calculé (compris entre ‘A’ et ‘G’)
            - second_epc_id : identifiant du deuxième DPE calculé
            - second_epc_surf : surface renseignée lors du calcul du 2e DPE
            - second_epc : classe du deuxième DPE calculé (compris entre ‘A’ et ‘G’).
    """
    
    # departement = Departement(dep_code)
    bdnb_df = filter_bdnb_individual(dep_code, force=False) # prend du temps si non déjà stocké en .csv
    
    # Filtrage des valeurs aberrantes
    filtre = zscore(bdnb_df.conso_5_usages_ep_m2)<5 # todo : il faudrait peut-être aussi filtrer les valeurs trop faibles ?
    bdnb_df = bdnb_df[filtre]
    
    # on conserve uniquement les bâtiments qui ont plus d'un DPE
    bat_many_dpe = bdnb_df['batiment_groupe_id'].value_counts() # décompte du nb de dpe par bâtiment
    bat_many_dpe = bat_many_dpe[bat_many_dpe >= 2].index # index des batiments avec plusieurs DPE
    bdnb_df = bdnb_df[bdnb_df['batiment_groupe_id'].isin(bat_many_dpe)]
    
    # tri chronologique du dataframe
    df_sorted = bdnb_df.sort_values(by=['batiment_groupe_id', 'date_etablissement_dpe'])
        
    def extract_consecutive_dpe(group, period):
        consecutive_pairs = []
        for i in range(len(group)-1):
            date_diff = (group.iloc[i+1]['date_etablissement_dpe'] - group.iloc[i]['date_etablissement_dpe']).days
            if date_diff < period : 
                consecutive_pairs.append({
                    'adresse_brut': group.iloc[i]['adresse_brut'],
                    'first_epc_id': group.iloc[i]['identifiant_dpe'],
                    'epc_date_1': group.iloc[i]['date_etablissement_dpe'].strftime("%d %b %Y"), # on enlève l'heure qui n'est pas pertinente dans la base de données
                    'epc_cons_1': group.iloc[i]['conso_5_usages_ep_m2'],
                    'first_epc': group.iloc[i]['classe_bilan_dpe'],
                    'surface_1' : group.iloc[i]['surface_habitable_logement'],
                    'second_epc_id': group.iloc[i+1]['identifiant_dpe'],
                    'epc_date_2': group.iloc[i+1]['date_etablissement_dpe'].strftime("%d %b %Y"),
                    'epc_cons_2': group.iloc[i+1]['conso_5_usages_ep_m2'],
                    'second_epc': group.iloc[i+1]['classe_bilan_dpe'],
                    'surface_2' : group.iloc[i+1]['surface_habitable_logement'],
                    'date_diff' : date_diff
                }) #'batiment_groupe_id': group.name, # groupby fait des Series (cf ci-dessous)

        return pd.DataFrame(consecutive_pairs)

    df_epc_evolution = df_sorted.groupby('batiment_groupe_id').apply(extract_consecutive_dpe, period=period)
    
    # Formatage de l'index du dataframe
    df_epc_evolution = df_epc_evolution.reset_index(drop=False)
    df_epc_evolution.rename(columns={"level_1": "numero_paire_de_dpe"}, inplace=True)
    df_epc_evolution.numero_paire_de_dpe = df_epc_evolution.numero_paire_de_dpe + 1

    # ajout colonne des variables_diff # todo : supprimer car c'est fait dans plot_variable_diff ?
    # df_epc_evolution = variable_diff(df_epc_evolution, variable = 'epc_cons') 
    # df_epc_evolution = variable_diff(df_epc_evolution, variable = 'surface')

    return df_epc_evolution   


def filter_manipulated_national(period, force = False): # attention : prend du temps (~5 min pour period = 20)
    """
    Creation du DataFrame des DPE successifs sur l'ensemble des départements.

    Parameters
    ----------
    period : int
        écart de temps maximal entre deux DPE successifs avant de considérer que des rénovations énergétiques ont pu avoir lieu.
    force : boolean
        force le compute et le ré-enregistrement en .csv
        
    Returns
    -------
    df_epc_evolution_national
        DataFrame des paires de DPE successifs pour chaque bâtiment.
    """
    
    # Définition du chemin de sauvegarde du DataFrame en .csv
    output_folder = os.path.join('data', 'BDNB', 'df_epc_evolution')
    os.makedirs(output_folder, exist_ok=True)
    existing_files = os.listdir(output_folder)
    
    # Définition du nom du fichier final
    save_name = f'df_epc_evolution_national_period_{period}'
    
    if save_name not in existing_files or force:
    
        france=France()
        df_epc_evolution_national = pd.DataFrame()
        for dep in tqdm.tqdm(france.departements):
            dep_code = dep.code
            df_epc_evolution_dep = filter_manipulated(dep_code, period=period)
            df_epc_evolution_national = pd.concat([df_epc_evolution_national, df_epc_evolution_dep])
        
        # Enregistrement en csv
        df_epc_evolution_national.to_csv(os.path.join(output_folder, save_name), index = False) # index = False permet de ne pas enregistrer la colonne d'index, qui sera recréée automatiquement lors de read_csv

    else:
        df_epc_evolution_national = pd.read_csv(os.path.join(output_folder, save_name)) 

    return df_epc_evolution_national   


# %% ANALYSE DE L'EVOLUTION DE CERTAINES VARIABLES ENTRE DPE SUCCESSIFS

# Dictionnaire pour le tracé des figures
variable_dict = {'epc_cons':'conso. énergétique',
           'epc_date':'date',
           'surface':'surface'}


def variable_diff(df_epc_evolution, variable = 'epc_cons'):
    """
    Ajout des colonnes {variable}_diff et {variable}_diff_rel (différence relative à la MOYENNE des deux DPE) au DataFrame df_epc_evolution

    Parameters
    ----------
    df_epc_evolution : pandas DataFrame
        DataFrame initial, comportant deux colonnes nommées '{variable}_1' et '{variable}_2'
    variable : str, optional
        nom de la variable d'intérêt : 'epc_cons', 'epc_date' ou 'surface'. The default is 'epc_cons'.
   
    Returns
    -------
    df_epc_evolution : pandas DataFrame
        tableau avec deux colonnes {variable}_diff et {variable}_diff_rel en plus.
    """

    df_epc_evolution[f'{variable}_diff'] =  df_epc_evolution[f"{variable}_2"] - df_epc_evolution[f"{variable}_1"]
    df_epc_evolution[f'{variable}_diff_rel'] =  df_epc_evolution[f'{variable}_diff'] / ((df_epc_evolution[f"{variable}_1"]+df_epc_evolution[f"{variable}_2"])/2) *100  # écart relatif par rapport à la moyenne des variables entre les deux DPE


    return df_epc_evolution



def plot_variable_diff(national_scale, period, dep_code=None, variable='epc_cons', ecart_relatif = True):
    """
    Trace l'histogramme des variations de {variable} entre paires de DPE.

    Parameters
    ----------
    national_scale : boolean
        trace à l'échelle nationale (utilise filter_manipulated_national). Le paramètre dep_code est alors inutile.
    period : int
        écart de temps maximal entre deux DPE successifs avant de considérer que des rénovations énergétiques ont pu avoir lieu.
    dep_code : TYPE, optional
        code du departement, si national_scale = False. The default is None.
    variable : TYPE, optional
        nom de la variable d'intérêt : 'epc_cons', 'epc_date' ou 'surface'. The default is 'epc_cons'.
    ecart_relatif : boolean, optional
        trace l'histogramme des variations RELATIVES (attention : différence relative à la MOYENNE des deux DPE, cf fonction variable_diff). The default is True.

    Returns
    -------
    None
    """
    
    # Définition du dataframe des paires de DPE à considérer
    if national_scale:
        df_epc_evolution = filter_manipulated_national(period)
    else:
        df_epc_evolution = filter_manipulated(dep_code, period)
        departement = Departement(dep_code)

    # Ajout des colonnes variable_diff
    df_epc_evolution = variable_diff(df_epc_evolution, variable) 
    
    # Calculs de statistiques générales
    count_diff = len(df_epc_evolution[df_epc_evolution[f'{variable}_diff'] != 0])
    diff_percent = count_diff / len(df_epc_evolution) *100
    print(f'Nombre de modifications de {variable} non nulles :', count_diff, f'parmi N={len(df_epc_evolution)} ({diff_percent:.1f} %)')
    print(f'Nombre de modifications de {variable} nulles :', len(df_epc_evolution)-count_diff, f'parmi N={len(df_epc_evolution)} ({100-diff_percent:.1f} %)') 
    
    count_diff_neg = len(df_epc_evolution[df_epc_evolution[f'{variable}_diff'] < 0])
    diff_neg_percent = count_diff_neg / len(df_epc_evolution) *100
    
    count_diff_pos = len(df_epc_evolution[df_epc_evolution[f'{variable}_diff'] > 0])
    diff_pos_percent = count_diff_pos / len(df_epc_evolution) *100
    
    if variable == 'surface':
        surface_manip_count = len(df_epc_evolution) - len(df_epc_evolution[(-1 < df_epc_evolution.surface_diff) & (df_epc_evolution.surface_diff < 1)])
        surface_manip_count_percent = surface_manip_count / len(df_epc_evolution) *100
        print('Nombre de modifications de surface supérieures à +-1 m2 :', surface_manip_count, f'parmi N={len(df_epc_evolution)} ({surface_manip_count_percent:.1f} %)')

    # Calcul de la moyenne et de l'écart-type pour la colonne d'intérêt
    if ecart_relatif:
        data_col = f'{variable}_diff_rel'
        unit = "%"
    else:
        data_col = f'{variable}_diff'
        unit = "kWh/m2" 

    mean_val = df_epc_evolution[data_col].mean()
    std_val = df_epc_evolution[data_col].std()

    # Alternative pour une distribution non normale : calcul des percentiles
    percentile_2_5 = df_epc_evolution[data_col].quantile(0.025)
    percentile_97_5 = df_epc_evolution[data_col].quantile(0.975)
    
    print(f"Moyenne de {data_col} : {mean_val:.2f} {unit}")
    print(f"Écart-type de {data_col} : {std_val:.2f} {unit}")
    print(f"Intervalle à 95% (percentiles 2.5 et 97.5) : [{percentile_2_5:.2f}, {percentile_97_5:.2f}] {unit}")


    # Tracé de l'histogramme
    
    fig, ax = plt.subplots(figsize=(5,5), dpi=300)
    
    # Définition des bins pour avoir histogramme centré en 0 
    max_variable = int(max(np.abs(df_epc_evolution[data_col]))) 
    bins_width = 1
    bins= np.asarray(range(-max_variable//bins_width*bins_width, max_variable//bins_width*bins_width+1, bins_width))
    bins = bins +bins_width/2

    df_epc_evolution.hist(column=data_col, ax=ax, bins=bins, grid=False, color=plt.get_cmap('viridis')(0.4))
        
    if ecart_relatif :
        ax.set_title(None)
        if national_scale==False:
            ax.set_title(f"{departement.name} - {departement.code}, N={len(df_epc_evolution)}") 
        ax.set_ylabel("Nombre d'observations")
        ax.set_xlabel(f"Ecart relatif de {variable_dict[variable]} entre DPE successifs, en %")
        
        ax.set_xlim([-100,100])
    else:
        ax.set_title(None)
        if national_scale==False:
            ax.set_title(f"{departement.name} - {departement.code}, N={len(df_epc_evolution)}") 
        ax.set_ylabel("Nombre d'observations")
        ax.set_xlabel(f"Ecart de {variable_dict[variable]} entre DPE successifs")
        
        # ax.set_xlim([-max_variable,max_variable])
        ax.set_xlim([percentile_2_5-10,percentile_97_5+10])
    
    ax.set_yscale('log')
    ax.set_ylim(bottom=1)

    
    # Ajout d'une annotation sur le graphique pour l'intervalle
    ax.axvline(x=mean_val, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Moyenne = {mean_val:.1f}')
    ax.axvline(x=percentile_2_5, color='blue', linestyle=':', linewidth=1, alpha=0.5, label=f'95% : [{percentile_2_5:.0f}, {percentile_97_5:.0f}]')
    ax.axvline(x=percentile_97_5, color='blue', linestyle=':', linewidth=1, alpha=0.5)
            
    ax.annotate(f"1$^{{er}}$ > 2$^{{e}}$ : {diff_neg_percent:.1f} %", 
        xy=(0.25, 0.7),
        xycoords=ax.transAxes,
        ha = 'center',
        fontsize=10)
# =============================================================================
#     ax.annotate(f"DPE à {variable} stable : {len(df_epc_evolution[df_epc_evolution[f'{variable}_diff'] == 0])}", #f"DPE à variable inchangée : {100 - diff_percent:.1f} %",
#         xy = (0.45, 0.99),
#         xytext=(0.5, 0.9),
#         xycoords=ax.transAxes,
#         ha = 'center',
#         fontsize=10)
# =============================================================================
    ax.annotate(f"1$^{{er}}$ < 2$^{{e}}$ : {diff_pos_percent:.1f} %",
        xy=(0.75, 0.7),
        xycoords=ax.transAxes,
        ha = 'center',
        fontsize=10)
    
    ax.legend(loc='lower right')
    
    
    # Enregistrement de la figure
    output_folder_hist_variations = os.path.join('output', '6. hist_variations_entre_DPE_successifs')
    os.makedirs(output_folder_hist_variations, exist_ok=True)
    
    if ecart_relatif :
        save_name = f'Ecart_relatif_{variable}_entre_DPE_successifs_{period}_dep{dep_code}.png'
    else:
        save_name = f'Ecart_{variable}_entre_DPE_successifs_{period}_dep{dep_code}.png'
    if national_scale:
        save_name = save_name.replace(f'dep{dep_code}.png','national.png')
    plt.savefig(os.path.join(output_folder_hist_variations,save_name), bbox_inches='tight')
    
    
    return 


# %% 

def plot_distrib_dpe_sucessifs(national_scale, period, dep_code='91', max_xlim =600):
    if national_scale:
        df_epc_evolution = filter_manipulated_national(period)
    else:
        df_epc_evolution = filter_manipulated(dep_code, period)
        departement = Departement(dep_code)
    
    bins = list(range(0,round(max(df_epc_evolution.epc_cons_1))))
    fig, ax = plt.subplots(figsize=(5,5), dpi=300)
    df_epc_evolution.hist('epc_cons_1', bins=bins, ax=ax, label = 'Premiers DPE', color= 'r', alpha = 0.6, grid=False)
    df_epc_evolution.hist('epc_cons_2', bins=bins, ax=ax, label = 'Seconds DPE', color= 'blue', alpha = 0.5, grid=False)
    if national_scale:
        ax.set_title(f"Ensemble des départements sur {period} jours, N={len(df_epc_evolution)}")
    else:
        ax.set_title(f"{departement.name} - {departement.code} sur {period} jours, N={len(df_epc_evolution)}")
    ax.set_ylabel("Nombre d'observations")
    ax.set_xlabel("Consommation annuelle en énergie primaire (kWh.m$^{-2}$)")
    ax.set_xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [max_xlim])
    ax.set_xlim(0, max_xlim)
    # plt.grid(False, alpha=0.3, zorder=-1)
    plt.legend()
    plt.show()
    
    return



def formatage_dpe_successifs_data(df_epc_evolution, window_size):
    """
    Création d'un DataFrame df_dpe_successifs permettant de calculer le bunching des deux distributions par la suite.

    Parameters
    ----------
    df_epc_evolution : TYPE
        DESCRIPTION.
    window_size : TYPE
        DESCRIPTION.

    Returns
    -------
    df_dpe_successifs : pandas DataFrame
        DESCRIPTION.
    """
    nb_dpe = len(df_epc_evolution)
    
    # Initialisation d'un dataframe des conso_5_usages
    max_epc_cons = max(df_epc_evolution['epc_cons_1'].max(),df_epc_evolution['epc_cons_2'].max())    
    df_dpe_successifs = pd.DataFrame(index=np.arange(0, int(max_epc_cons) + 1))
    df_dpe_successifs.rename_axis('conso_5_usages_ep_m2', inplace=True)

    # Dictionnaires du nombre de DPE par valeur de conso_5_usages
    count_epc_cons_1 = df_epc_evolution.epc_cons_1.map(round).value_counts()
    count_epc_cons_1 = df_dpe_successifs.join(count_epc_cons_1)
    count_epc_cons_1.rename(columns={"count": "count_epc_cons_1"}, inplace=True)

    count_epc_cons_2 = df_epc_evolution.epc_cons_2.map(round).value_counts()
    count_epc_cons_2 = df_dpe_successifs.join(count_epc_cons_2)
    count_epc_cons_2.rename(columns={"count": "count_epc_cons_2"}, inplace=True)

    df_dpe_successifs = count_epc_cons_1.join(count_epc_cons_2)
    df_dpe_successifs = df_dpe_successifs.fillna(0)
    
    # Différence des deux distributions
    df_dpe_successifs['diff_distrib_dpe_sucessifs'] = df_dpe_successifs.count_epc_cons_2 - df_dpe_successifs.count_epc_cons_1 
    
    # Moyenne glissante de chacune des distribution
    rolling_dpe = df_dpe_successifs['count_epc_cons_1'].rolling(window=window_size, min_periods=1, center=True) 
    df_dpe_successifs["moyenne_distrib_1"] = rolling_dpe.mean()  
    
    rolling_dpe = df_dpe_successifs['count_epc_cons_2'].rolling(window=window_size, min_periods=1, center=True) 
    df_dpe_successifs["moyenne_distrib_2"] = rolling_dpe.mean()  
    
    # Normalisation par nb total de DPE
    df_dpe_successifs['y_diff_moyenne_norm_1'] = (df_dpe_successifs.count_epc_cons_1 - df_dpe_successifs.moyenne_distrib_1)/nb_dpe 
    df_dpe_successifs['y_diff_moyenne_norm_abs_1'] = df_dpe_successifs['y_diff_moyenne_norm_1'].abs() 
    df_dpe_successifs['y_diff_moyenne_norm_2'] = (df_dpe_successifs.count_epc_cons_2 - df_dpe_successifs.moyenne_distrib_2)/nb_dpe 
    df_dpe_successifs['y_diff_moyenne_norm_abs_2'] = df_dpe_successifs['y_diff_moyenne_norm_2'].abs() 

    
    return df_dpe_successifs



def plot_diff_distrib_dpe_successifs(period, max_xlim=600): # echelle nationale
    
    df_epc_evolution = filter_manipulated_national(period)
    df_dpe_successifs = formatage_dpe_successifs_data(df_epc_evolution)
    
    fig, ax = plt.subplots(figsize=(5,5), dpi=300)
    # ax.plot(df_dpe_successifs.index, df_dpe_successifs.diff_distrib_dpe_sucessifs, color=plt.get_cmap('viridis')(0.1), linewidth = 0.7)
    
    ax.plot(df_dpe_successifs.index, df_dpe_successifs.count_epc_cons_1, color=plt.get_cmap('viridis')(0.1), linewidth = 0.7, label = 'premiers DPE')
    ax.plot(df_dpe_successifs.index, df_dpe_successifs.moyenne_distrib_1,"k", label='moyenne', linewidth = 1)

    
    # ax.plot(df_dpe_successifs.index, df_dpe_successifs.count_epc_cons_2, color=plt.get_cmap('viridis')(0.1), linewidth = 0.7, label = 'seconds DPE')
    # ax.plot(df_dpe_successifs.index, df_dpe_successifs.moyenne_distrib_2,"k", label='moyenne', linewidth = 1)

    
    ax.set_xlim([0,max_xlim])
    ax.set_ylim([0,800])

    # ax.hlines(y=0, xmin=0, linewidth = 1, xmax=max_xlim, color='k', alpha=0.4, zorder=-1) # tracé de l'axe y=0 en arrière-plan
    ax.set_ylabel("Nombre de DPE de différence")
    ax.set_xlabel("Consommation annuelle en énergie primaire (kWh.m$^{-2}$)")
    ax.set_xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [max_xlim])
    ax.set_title(f"Ensemble des départements sur {period} jours, N={len(df_epc_evolution)}")
    plt.legend()     
    
    # # Définition du chemin de sauvegarde des histogrammes des champs modifiés
    # output_folder = os.path.join('output', '') #todo : modifier
    # os.makedirs(output_folder, exist_ok=True)

    # # Enregistrement de la figure
    # save_path = os.path.join(output_folder,f'diff_distrib_dpe_successifs.png')
    # plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return




def calcul_bunching_dpe_successifs(df_epc_evolution, seuils, itv_bunching, window_size):
    
    df_dpe_successifs = formatage_dpe_successifs_data(df_epc_evolution, window_size=window_size)
    
    # méthode différence simple (utilisée par Civel et al. 2025)
    
    method ='diff_simple'
    
    bunching_dpe_succ = pd.DataFrame(index=[0]) # initialisation d'un DataFrame
    
    bunching = 0
    
    for seuil in seuils:
        valeur_seuil = etiquette_ep_seuils[seuil]
        print(bunching_dpe_succ)
        
        nb_gauche = df_dpe_successifs[(df_dpe_successifs.index > valeur_seuil-itv_bunching) & (df_dpe_successifs.index <= valeur_seuil)].sum()
        nb_droite = df_dpe_successifs[(df_dpe_successifs.index > valeur_seuil) & (df_dpe_successifs.index <= valeur_seuil+itv_bunching)].sum() # todo : pb il faut dire quelle colonne

        diff_simple = (nb_gauche - nb_droite) / (nb_gauche+nb_droite) 
        bunching += diff_simple

        bunching_dpe_succ['diff_simple'] += diff_simple
        
        
    method ='diff_moyenne'
      
# =============================================================================
#     # Initialisation des DataFrame
#     bunching_df = pd.DataFrame(index=['Premier DPE','Second DPE'])
#     # bunching_df_2 = pd.DataFrame(index=[0]) 
#     
#     for k, seuil in etiquette_ep_seuils.items():
#         # Création d'un DataFrame filtré sur l'intervalle à +-{itv_bunching} du seuil
#         df_dpe_successifs_filtered = df_dpe_successifs[(df_dpe_successifs.index  > seuil-itv_bunching) & (df_dpe_successifs.index <= seuil + itv_bunching)]
#         
#         # Ajout d'une colonne correspondante au seuil dans le bunching DataFrame 
#         bunching_df[1][f'{k}_method_{method}'] = df_dpe_successifs_filtered['y_diff_moyenne_norm_abs_1'].sum() 
#         bunching_df_2[f'{k}_method_{method}'] = df_dpe_successifs_filtered['y_diff_moyenne_norm_abs_2'].sum() 
# 
#     bunching_df_cut, seuils_sans_slash = cut_france_bunching(bunching_df, seuils)
#     
# =============================================================================
    
    # Initialisation des DataFrame
    bunching_df_1 = pd.DataFrame(index=[0])
    bunching_df_2 = pd.DataFrame(index=[0]) 
    
    for k, seuil in etiquette_ep_seuils.items():
        # Création d'un DataFrame filtré sur l'intervalle à +-{itv_bunching} du seuil
        df_dpe_successifs_filtered = df_dpe_successifs[(df_dpe_successifs.index  > seuil-itv_bunching) & (df_dpe_successifs.index <= seuil + itv_bunching)]
        
        # Ajout d'une colonne correspondante au seuil dans le bunching DataFrame 
        bunching_df_1[f'{k}_method_{method}'] = df_dpe_successifs_filtered['y_diff_moyenne_norm_abs_1'].sum() 
        bunching_df_2[f'{k}_method_{method}'] = df_dpe_successifs_filtered['y_diff_moyenne_norm_abs_2'].sum() 

    # bunching_df_1[:, [3:6]]
    bunching_df_cut_1, seuils_sans_slash = cut_france_bunching(bunching_df_1, seuils)
    bunching_premier_dpe = bunching_df_cut_1.iloc[0].sum() #axis=1) 
    bunching_df_cut_2, seuils_sans_slash = cut_france_bunching(bunching_df_2, seuils)
    bunching_second_dpe = bunching_df_cut_2.iloc[0].sum()

    print('Bunching méthode diff_moyenne sur la distribution des premier DPE :', bunching_premier_dpe)
    print('Bunching méthode diff_moyenne sur la distribution des second DPE :', bunching_second_dpe)

    pourcent_variation_bunching = (bunching_second_dpe - bunching_premier_dpe)/ bunching_premier_dpe *100
    print(f'Augmentation du bunching de {pourcent_variation_bunching:.2f} %')
    
    return


# =============================================================================
# NE SERT A RIEN CAR ON A DEJA PLOT VARIABLE DIFF 
# def plot_distrib_ecart_conso(national_scale, period, dep_code='91', max_xlim =600):
#     if national_scale:
#         df_epc_evolution = filter_manipulated_national(period)
#     else:
#         df_epc_evolution = filter_manipulated(dep_code, period)
#         departement = Departement(dep_code)
#     
#     
#     
#     return
# =============================================================================


def plot_gain_period(national_scale, dep_code='91', period_max=120, bins_size = 5):
    
    #todo : probleme de dimension de x et y si pas assez de donnees et donc que certains bins n'ont aucun dpe dedans (ex 75) --> augmenter bins_size dns ce cas ? ou creer x_value qui dépend directement du calcul de df_gain_moyen_bins
    
    if national_scale:
        df_epc_evolution = filter_manipulated_national(period_max)
    else:
        df_epc_evolution = filter_manipulated(dep_code, period_max)
        departement = Departement(dep_code)
    
    df_epc_evolution['ecart_date'] = pd.to_datetime(df_epc_evolution.epc_date_2, format="%d %b %Y") - pd.to_datetime(df_epc_evolution.epc_date_1, format="%d %b %Y")
    df_epc_evolution['ecart_date'] = df_epc_evolution['ecart_date'].dt.days
    # Calcul du gain moyen d'étiquette sur l'ensemble des paires de DPE successifs
    epc_order = {'A':7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1} # on attribue valeur à chaque classe
    df_epc_evolution['first_val'] = df_epc_evolution['first_epc'].map(epc_order) 
    df_epc_evolution['second_val'] = df_epc_evolution['second_epc'].map(epc_order)
    df_epc_evolution['gain_etiquette'] = df_epc_evolution['second_val'] - df_epc_evolution['first_val']


    # Tracé histogramme des ecart_date
    # bins_sequence = list(range(0,period+1)) 
    # fig,ax = plt.subplots(figsize=(5,5), dpi=300) 
    # df_epc_evolution.hist(column='ecart_date', ax=ax, bins=bins_sequence, color='k')
    # df_epc_evolution.hist(column='ecart_date', ax=ax, bins=20, color='k') # autre version

    # Tracé du gain moyen d'étiquette en fonction de l'écart entre les deux DPE successifs 
    #df_gain_moyen = df_epc_evolution.groupby(['ecart_date'])['gain_etiquette'].mean() # moyenne sans grouper par périodes
    
    # Tracé du gain moyen d'étiquette en fonction de la période d'écart entre les deux DPE successifs 
    bins = list(range(0, period_max+1, bins_size))
    df_epc_evolution['bins_ecart_date'] = pd.cut(df_epc_evolution.ecart_date, bins=bins, include_lowest=True) # include_lowest=True pour inclure 0 dans le premier bin
    stats_gain_moyen = df_epc_evolution.groupby(['bins_ecart_date'])['gain_etiquette'].agg(['mean', 'std'])
    # df_gain_moyen_bins = df_gain_moyen_bins.append([0]) # todo : rajouter un 0 pour que la courbe soit plus belle
    #df_gain_moyen_bins_std = df_epc_evolution.groupby(['bins_ecart_date'])['gain_etiquette'].std()
    
    # x_values = bins[1:]
    # x_values = bins[:-1]
    x_values = [interval.left for interval in stats_gain_moyen.index]  # extraction des bornes inférieures pour l'axe x
    y_mean = stats_gain_moyen['mean'].values
    y_std = stats_gain_moyen['std'].values # Écart-type
    
    fig,ax = plt.subplots(figsize=(5,5), dpi=300)  
    ax.plot(x_values, y_mean, ds='steps-post', label="Gain d'étiquette moyen")
    ax.fill_between(x_values, y_mean - y_std, y_mean + y_std, step = 'post', alpha=0.2, label='Ecart-type')
    ax.set_xticks(ticks=list(range(0, period_max+1, 10)))
    #ax.errorbar(yerr=)
    if national_scale:
        ax.set_title(f"Ensemble des départements, N={len(df_epc_evolution)}")
    else:
        ax.set_title(f"{departement.name} - {departement.code}, N={len(df_epc_evolution)}")
    ax.set_xlim(0, period_max)
    #ax.set_ylim(bottom=0)
    ax.set_xlabel('Écart entre les DPE successifs (jours)')
    ax.set_ylabel('Gain moyen d\'étiquette')
    
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.show()

    
    # sns.regplot(data=df_epc_evolution, x='ecart_date', y='gain_etiquette') 

    return


def analyse_gain_etiquette(dep_code, period):
    
    df_epc_evolution = filter_manipulated(dep_code, period = period)[['first_epc','second_epc']] # version rapide non nettoyée
    N = len(df_epc_evolution)

    # Calcul de la part des DPE stables
    stable_mask = df_epc_evolution['first_epc'] == df_epc_evolution['second_epc']
    part_dpe_stables = stable_mask.sum() / N
    
    # Calcul du gain moyen d'étiquette sur l'ensemble des paires de DPE successifs
    epc_order = {'A':7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1} # on attribue valeur à chaque classe
    df_epc_evolution['first_val'] = df_epc_evolution['first_epc'].map(epc_order) 
    df_epc_evolution['second_val'] = df_epc_evolution['second_epc'].map(epc_order)
    df_epc_evolution['gain_etiquette'] = df_epc_evolution['second_val'] - df_epc_evolution['first_val']
    gain_moyen_etiquette = df_epc_evolution['gain_etiquette'].mean()

    # Calcul du gain moyen d'étiquette parmi les DPE modifiés uniquement
    modif_mask = ~stable_mask
    gain_moyen_etiquette_parmi_modif = df_epc_evolution.loc[modif_mask,'gain_etiquette'].mean()
    
    return part_dpe_stables, gain_moyen_etiquette, gain_moyen_etiquette_parmi_modif



def plot_gain_period_cumule(dep_code):
    period_list = list(range(0,120,5))
    part_dpe_stables = []
    gain_moyen_etiquette = []
    gain_moyen_etiquette_parmi_modif = []
    for period in tqdm.tqdm(period_list):
        x1, x2, x3 = analyse_gain_etiquette(dep_code, period)
        part_dpe_stables.append(x1)
        gain_moyen_etiquette.append(x2)
        gain_moyen_etiquette_parmi_modif.append(x3)
        
    fig,ax = plt.subplots(figsize=(5,5), dpi=300)  
    ax.plot(period_list, gain_moyen_etiquette, label = 'gain_moyen_etiquette')
    ax.plot(period_list, gain_moyen_etiquette_parmi_modif, label = 'gain_moyen_etiquette_parmi_modif')
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.show()
    
    fig,ax = plt.subplots(figsize=(5,5), dpi=300)  
    ax.plot(period_list, part_dpe_stables, label = 'part_dpe_stables')
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.show()
    
    return


def dicts_dep_gain_moyen_etiquette(period, save_json=False):
    
    france = France()
    
    dict_part_dpe_stables  = {d:[] for d in france.departements} 
    dict_gain_moyen_etiquette  = {d:[] for d in france.departements} 
    dict_gain_moyen_etiquette_parmi_modif  = {d:[] for d in france.departements} 

    for dep in tqdm.tqdm(france.departements) :
        dep_code = dep.code
        part_dpe_stables, gain_moyen_etiquette, gain_moyen_etiquette_parmi_modif = analyse_gain_etiquette(dep_code, period=period)
        dict_part_dpe_stables[dep] = part_dpe_stables*100
        dict_gain_moyen_etiquette[dep] = gain_moyen_etiquette
        dict_gain_moyen_etiquette_parmi_modif[dep] = gain_moyen_etiquette_parmi_modif
    
    # todo : enregistrer en json mais besoin departement --> dep code
    # if save_json:
    #     dict_part_dpe_stables
    #     print(json.dumps(dict_part_dpe_stables, indent=4)) 
    
    # Tracé et enregistrement des cartes
    output_folder = os.path.join('output', '4. cartes_analyse_gain_etiquettes')
    os.makedirs(output_folder, exist_ok=True)

    save = f'carte_part_dpe_stables_sur_{period}_jours'
    map_title = f'Part des DPE stables sur {period} jours (%)'
    draw_departement_map(dict_part_dpe_stables, output_folder,save=save, map_title=map_title) # todo : fixer les bornes de la colorbar ? 
    plt.show()
    plt.close()
    
    save = f'carte_gain_moyen_etiquette_sur_{period}_jours'
    map_title ="Maisons individuelles"
    cbar_label = f"Gain moyen d'étiquette sur {period} jours" # todo : pareil partout

    draw_departement_map(dict_gain_moyen_etiquette, output_folder, save=save, map_title=map_title, cbar_label=cbar_label)
    plt.show()
    plt.close()
    
    save = f'carte_gain_moyen_etiquette_parmi_modif_sur_{period}_jours'
    map_title = f"Gain moyen d'étiquette parmi les DPE modifiés sur {period} jours"
    draw_departement_map(dict_gain_moyen_etiquette_parmi_modif, output_folder, save=save, map_title=map_title)
    plt.show()
    plt.close()

    return dict_part_dpe_stables, dict_gain_moyen_etiquette, dict_gain_moyen_etiquette_parmi_modif


    


# %% HEATMAP ET DIAGRAMME DE SANKEY


def plot_heatmap(national_scale, dep_code=None, frequency=True, period = 20):
    """
    Tracé de la heatmap de comparaison des paires de DPE successifs.

    Parameters
    ----------
    national_scale : boolean
        trace à l'échelle nationale (utilise filter_manipulated_national). Le paramètre dep_code est alors inutile.
    dep_code : TYPE, optional
        code du departement, si national_scale = False. The default is None.
    period : int
        écart de temps maximal entre deux DPE successifs avant de considérer que des rénovations énergétiques ont pu avoir lieu. 
    frequency : boolean
        if True, trace la heatmap en fréquence et non en absolu

    Returns
    -------
    None
    """
    
    if national_scale:
        df_epc_evolution = filter_manipulated_national(period)
    else:
        departement = Departement(dep_code)
        
        # Version rapide non nettoyée
        df_epc_evolution = filter_manipulated(dep_code, period = period)  
        
        # Version nettoyée des DPE en double : (beaucoup plus long car il faut télécharger tous les json) mais ne change pas grand chose --> a eviter
        # df_epc_evolution = delete_dpe_copies(dep_code, plot_surface_evolution = False, period = period)
    
    # Décompte de la fréquence des transitions avec crosstab()
    df_heatmap = pd.crosstab(
        index=df_epc_evolution['second_epc'],  # Lignes = 2e DPE
        columns=df_epc_evolution['first_epc'],  # Colonnes = 1er DPE
    )
    
    # Remplissage des classes manquantes avec 0
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    df_heatmap = df_heatmap.reindex(index=classes, columns=classes, fill_value=0)
    if frequency:
        df_heatmap = (df_heatmap/len(df_epc_evolution))*100        

    # Pour afficher seulement les valeurs non nulles de la matrice
    annot = df_heatmap
    annot = np.round(annot, 1)
    annot = np.where(annot != 0, annot, "")

    # Tracé de la figure    
    fig,ax = plt.subplots(figsize=(5,5), dpi=300)

    cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
    posn = ax.get_position()
    cbar_ax.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
    
    if frequency:
        ax = sns.heatmap(df_heatmap, ax=ax, vmin=0, vmax=25, annot=annot, fmt="", cmap='bone_r', cbar=True, cbar_ax=cbar_ax, cbar_kws={'label':'Pourcentage (%)'})
    else:
        ax = sns.heatmap(df_heatmap, ax=ax, annot=annot, fmt="", cmap='bone_r', cbar_ax=cbar_ax,cbar=True,cbar_kws={'label':"Nombre d'observations"})
    
    if national_scale==False: 
        ax.set_title(f'Logements individuels, {departement.name} - {departement.code}\nPériode de {period} jours, N={len(df_epc_evolution)}')
    for spine in ax.spines.values():
        spine.set_visible(True)
    for spine in cbar_ax.spines.values():
        spine.set_visible(True)
    ax.set_ylabel('Second DPE')
    ax.set_xlabel('Premier DPE')
    
    
    # Définition du chemin de sauvegarde des heatmap
    output_folder_heatmap = os.path.join('output', '1. heatmap')
    os.makedirs(output_folder_heatmap, exist_ok=True)
    existing_files = os.listdir(output_folder_heatmap)
    
    # Enregistrement de la figure
    if national_scale : 
        save_name = f'DPE_manipulation_classes_national_sur_{period}_jours.png'
    else : 
        save_name = f'DPE_manipulation_classes_{dep_code}_sur_{period}_jours.png'
    if frequency:
        save_name = save_name.replace('.png','_frequency.png')

    plt.savefig(os.path.join(output_folder_heatmap,save_name), bbox_inches='tight')
    
    plt.show()

    return 



def plotly_sankey(national_scale, period, dep_code=None):
    """
    Tracé du diagramme de Sankey de l'évolution des classes énergétiques entre DPE successifs dans une fenêtre web.

    Parameters
    ----------
    national_scale : boolean
        trace à l'échelle nationale (utilise filter_manipulated_national). Le paramètre dep_code est alors inutile.
    period : int
        écart de temps maximal entre deux DPE successifs avant de considérer que des rénovations énergétiques ont pu avoir lieu.
    dep_code : TYPE, optional
        code du departement, si national_scale = False. The default is None.

    Returns
    -------
    None
    """

    
    departement = Departement(dep_code)
    output_folder_sankey = os.path.join('output', '2. sankey diagram')
    os.makedirs(output_folder_sankey, exist_ok=True)
    
    # Définition du dataframe des paires de DPE à considérer
    if national_scale:
        df_epc_evolution = filter_manipulated_national(period)
    else:
        df_epc_evolution = filter_manipulated(dep_code, period)
    
    # todo : rajouter ici un delete_dpe_copies ?
        
    # Décompte des transitions entre chaque paire de classes DPE
    transition_counts = df_epc_evolution.groupby(['first_epc', 'second_epc']).size().reset_index(name='count')
    
    # Labels des noeuds (7 initiaux + 7 finaux)
    labels = [f"{classe}_initial" for classe in ['A', 'B', 'C', 'D', 'E', 'F', 'G']] + [f"{classe}_final" for classe in ['A', 'B', 'C', 'D', 'E', 'F', 'G']]
    label_to_index = {label: idx for idx, label in enumerate(labels)} # indices correspondants à chaque noeud
    
    sources = transition_counts['first_epc'].map(lambda x: label_to_index[f"{x}_initial"])
    targets = transition_counts['second_epc'].map(lambda x: label_to_index[f"{x}_final"])
    values = transition_counts['count']
    
    # Couleur des noeuds
    node_colors = etiquette_colors_dict.values()
    # Conversion en format RGBA
    node_colors_rgba = [f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 1)" for r, g, b in node_colors]
    
    # Couleur de chaque lien en fonction de sa classe initiale
    link_colors = transition_counts['first_epc'].map(etiquette_colors_dict)
    # Conversion en format RGBA (opacité à 0.8)
    link_colors_rgba = [f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.8)" for r, g, b in link_colors]
    
    
    fig = go.Figure(go.Sankey(
        arrangement = 'snap',
        node=dict(
            label=[label.replace("_initial", "").replace("_final", "") for label in labels],
            color= node_colors_rgba + node_colors_rgba
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors_rgba 
        )
    ))
    
    
    if national_scale:
        title_text=f"Transitions de classes entre DPE successifs, France hexagonale, N={len(df_epc_evolution)}, écart max. entre DPE = {period} jours)"
    else : 
        title_text=f"Transitions de classes entre DPE successifs ({departement.name} - {departement.code}, N={len(df_epc_evolution)}, écart max. entre DPE = {period} jours)"
    
    # Personnalisation de la disposition pour séparer les deux groupes de nœuds
    fig.update_layout(
        title_text=title_text,
        font_size=30,
        title_font_size=30,
        #font_color = 'black', 
        font_shadow = "auto", # 'None' si pas d'ombre autour des labels
        )
    
    fig.show()
    
    return


# %%


# Définition des champs à ne pas prendre en compte car non manipulable directement par les diagnostiqueurs pour influencer les calculs
set_admin_and_geog = {'numero_dpe',
                    'date_derniere_modification_dpe',
                    'date_visite_diagnostiqueur',
                    'date_etablissement_dpe',
                    'date_reception_dpe',
                    'date_fin_validite_dpe',
                    # Bilan DPE
                    'etiquette_dpe',
                    'etiquette_ges',
                    # Localisation
                    'adresse_ban',
                    'numero_voie_ban',
                    'nom_rue_ban',
                    'nom_commune_ban',
                    'code_postal_ban',
                    'code_insee_ban',
                    'code_departement_ban',
                    'code_region_ban',
                    'identifiant_ban',
                    'coordonnee_cartographique_x_ban',
                    'coordonnee_cartographique_y_ban',
                    'score_ban',
                    'statut_geocodage',
                    'adresse_brut',
                    'adresse_complete_brut',
                    'nom_commune_brut',
                    'code_postal_brut',
                    'numero_etage_appartement',
                    'position_logement_dans_immeuble',
                    'nom_residence',
                    'complement_adresse_batiment',
                    'complement_adresse_logement',
                    '_geopoint'
                    }
                    
                       
set_variables_intermediaires = {
                       # Déperdition
                       'deperditions_enveloppe', # correspond à la somme des autres déperditions 
                       # Isolation
                       'qualite_isolation_enveloppe',
                       'ubat_w_par_m2_k',
                       # Apport et besoin
                       'besoin_chauffage',
                       # Consommation en énergie primaire
                       'conso_5_usages_ep',
                       'conso_5_usages_par_m2_ep',
                       'conso_chauffage_ep',
                       'conso_ecs_ep',
                       'conso_refroidissement_ep',
                       'conso_eclairage_ep',
                       'conso_auxiliaires_ep',
                       # Consommation en énergie finale
                       'conso_5_usages_ef',
                       'conso_5_usages_par_m2_ef',
                       'conso_chauffage_ef',
                       'conso_ecs_ef',
                       'conso_refroidissement_ef',
                       'conso_eclairage_ef',
                       'conso_auxiliaires_ef',
                       # Emissions de GES
                       'emission_ges_5_usages',
                       'emission_ges_5_usages_par_m2',
                       'emission_ges_chauffage',
                       'emission_ges_ecs',
                       'emission_ges_refroidissement',
                       'emission_ges_eclairage',
                       'emission_ges_auxiliaires',
                       # Bilan par énergie
                       'conso_5_usages_ef_energie_n1',
                       'conso_chauffage_ef_energie_n1',
                       'conso_ecs_ef_energie_n1',
                       'cout_total_5_usages_energie_n1',
                       'cout_chauffage_energie_n1',
                       'cout_ecs_energie_n1',
                       'emission_ges_5_usages_energie_n1',
                       'emission_ges_chauffage_energie_n1',
                       'emission_ges_ecs_energie_n1',
                       'conso_5_usages_ef_energie_n2',
                       'conso_chauffage_ef_energie_n2',
                       'conso_ecs_ef_energie_n2',
                       'cout_total_5_usages_energie_n2',
                       'cout_chauffage_energie_n2',
                       'cout_ecs_energie_n2',
                       'emission_ges_5_usages_energie_n2',
                       'emission_ges_chauffage_energie_n2',
                       'emission_ges_ecs_energie_n2',
                       'conso_5_usages_ef_energie_n3',
                       'conso_chauffage_ef_energie_n3',
                       'conso_ecs_ef_energie_n3',
                       'cout_total_5_usages_energie_n3',
                       'cout_chauffage_energie_n3',
                       'cout_ecs_energie_n3',
                       'emission_ges_5_usages_energie_n3',
                       'emission_ges_chauffage_energie_n3',
                       'emission_ges_ecs_energie_n3',
                       # Coûts
                       'cout_total_5_usages',
                       'cout_chauffage',
                       'cout_ecs',
                       'cout_refroidissement',
                       'cout_eclairage',
                       'cout_auxiliaires',
                       # Chauffage
                       'conso_chauffage_installation_chauffage_n1',
                       'conso_chauffage_generateur_n1_installation_n1',
                       'conso_chauffage_generateur_n2_installation_n1',
                       'conso_chauffage_installation_chauffage_n2',
                       'conso_chauffage_generateur_n1_installation_n2',
                       'conso_chauffage_generateur_n2_installation_n2',
                       # ECS
                       'conso_ef_installation_ecs_n1',
                       'conso_ef_generateur_n1_ecs_n1',
                       'conso_ef_generateur_n2_ecs_n1',
                       # Autres
                       '_score',
                       '_rand',
                       '_i',
                       '_id'
                       }




def download_dpe_json(dpe_id, force=False):
    """
    Téléchargement des fichiers de sorties des DPE (au format json) depuis l'API de l'ADEME.

    Parameters
    ----------
    dpe_id : str
        identifiant du dpe.
    force : boolean, optional
        force le re-téléchargement du .json. The default is False.

    Returns
    -------
    None.

    """
    output_folder_dpe_details = os.path.join('data', 'DPE', 'JSON')
    os.makedirs(output_folder_dpe_details, exist_ok=True)
    
    if '{}.json'.format(dpe_id) in os.listdir(output_folder_dpe_details) or force:
        return
    
    
    else:
        dls = f"https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines?numero_dpe_eq={dpe_id}" 
        req = Request(dls)
        content = urlopen(req)
        
# =============================================================================
#         if content.status_code == 429:
#                 retry_after = int(content.headers.get("Retry-After", 1))
#                 print(f"Rate limited. Sleeping for {retry_after} seconds...")
#                 time.sleep(retry_after * (2 ** i))  # exponential backoff
# =============================================================================


        # with open(os.path.join('data','DPE','XLS','{}.xlsx'.format(dpe_id)), 'wb') as output:
        with open(os.path.join('data','DPE','JSON','{}.json'.format(dpe_id)), 'wb') as output:
            output.write(content.read())
            
            # todo : rajouter ici le fait de nettoyer les json ? et de supprimer les json vide ?
    
    return


def download_all_dpe_json(df_epc_evolution):
    """
    Téléchargement des fichiers de sorties des DPE (au format json) depuis l'API de l'ADEME.

    Parameters
    ----------
    df_epc_evolution : pandas DataFrame
        DESCRIPTION.

    Returns
    -------
    ensemble_dpe : set
        Ensemble des DPE présents dans la table df_epc_evolution.
    """
    
    output_folder_dpe_details = os.path.join('data', 'DPE', 'JSON')
    os.makedirs(output_folder_dpe_details, exist_ok=True)
    printed=False
    
    ensemble_dpe = set(pd.concat([df_epc_evolution["first_epc_id"], df_epc_evolution["second_epc_id"]])) 
    for dpe_id in ensemble_dpe :       
        if '{}.json'.format(dpe_id) not in os.listdir(output_folder_dpe_details):
            download_dpe_json(dpe_id) # download prend 0.3s par dpe_id environ --> 200 req/min, or 600 requêtes/min max
            time.sleep(0.01) # ralentir pour éviter erreur 429 (ne marche pas tout à fait ?)
            if not printed:
                print("Wait... Downloading the json files. This can take a few minutes per department.")
                printed = True
    
    return ensemble_dpe



# %%



def diff_dpe_data(dpe_id1,dpe_id2):
    """
    Retourne les différences entre deux DPE successifs (json) à l'aide de jsondiff.

    Parameters
    ----------
    dpe_id1 : TYPE
        DESCRIPTION.
    dpe_id2 : TYPE
        DESCRIPTION.

    Returns
    -------
    json_dpe1 : TYPE
        DESCRIPTION.
    json_dpe2 : TYPE
        DESCRIPTION.
    dpe_diff : TYPE
        DESCRIPTION.
    """
    
    # Load the JSON files    
    with open(os.path.join('data','DPE','JSON','{}.json'.format(dpe_id1)), 'r') as f:
        json_dpe1 = json.load(f)['results']
    with open(os.path.join('data','DPE','JSON','{}.json'.format(dpe_id2)), 'r') as f:
        json_dpe2 = json.load(f)['results']
    
    # Compute the differences using jsondiff
    dpe_diff = diff(a=json_dpe1, b=json_dpe2, syntax='symmetric') #, marshal=True) # marshal permet d'avoir '$delete' et '$insert' en str (et non objets jsondiff)
    
    # todo : nettoyer dpe_diff[0][jd.delete] et dpe_diff[0][jd.insert]
    
    return json_dpe1, json_dpe2, dpe_diff



def compare_dpe_data(dpe_id1,dpe_id2): # todo : inutile mtn qu'on a syntax = symmetric ?
    """
    Création d'un DataFrame pour comparer les variables modifiées entre deux DPE successifs. Ne tient pas compte des variables finales issues de calculs (conso_5_usages etc)

    Parameters
    ----------
    dpe_id1 : TYPE
        DESCRIPTION.
    dpe_id2 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    

    json_dpe1, json_dpe2, dpe_diff = diff_dpe_data(dpe_id1,dpe_id2)
    
    if dpe_diff['results'] == []:
        print('Les deux DPE sont exactement identiques.') # todo : modifier : pas identique mais pb 
        return 

    # Initialisation DataFrame des variables modifiées # todo : enlever set_to_not_consider mais laisser infos importantes (numero_dpe, date)
    list_changing_variables = [k for k in dpe_diff['results'][0].keys()] # todo : faire un code plus beau sans dictionnaire de dictionnaire ?
    for variable in set_variables_intermediaires :
        if variable in list_changing_variables:
            list_changing_variables.remove(variable)
        else:
            print(variable)
    
    df_changing_variables = pd.DataFrame(index = list_changing_variables)
    
    # Jointure des détails des DPE successifs
    df_dpe1 = pd.DataFrame().from_dict(json_dpe1['results'][0], orient='index', columns =['First DPE'])
    df_dpe2 = pd.DataFrame().from_dict(json_dpe2['results'][0], orient='index', columns =['Second DPE'])

    comparison_df = df_dpe1.join(df_dpe2, how='outer') # on conserve tous les champs, y compris ceux qui ne sont renseignés que dans un seul des deux df
    comparison_df = df_changing_variables.join(comparison_df)
    
    print(comparison_df)

    return comparison_df



def delete_dpe_copies(dep_code, period): # todo : prendre plutot df_epc_evolution en argument ? changer le nom
    """
    Nettoie df_epc_evolution en enlevant les DPE successifs qui sont en réalité identiques.

    Parameters
    ----------
    dep_code : str
        code du departement.

    Returns
    -------
    df_epc_evolution : pandas DataFrame
        DataFrame des paires de DPE successifs avec ajout d'une colonne "dpe_diff" listant les champs modifiés.
    """
    
    # Récupération des DPE successifs effectués LE MEME JOUR (period = 0) #todo modifier ce commentaire
    df_epc_evolution = filter_manipulated(dep_code, period = period)
    
    
    # Téléchargement de tous les fichiers json nécessaires au calcul dpe_diff
    download_all_dpe_json(df_epc_evolution)
    
    # Création colonne "dpe_diff" listant les champs modifiés entre les deux DPE successifs
    liste_dpe_diff = []
    for index, row in df_epc_evolution.iterrows() :
        dpe_id1 = row["first_epc_id"]
        dpe_id2 = row["second_epc_id"]
        json_dpe1, json_dpe2, dpe_diff = diff_dpe_data(dpe_id1, dpe_id2)
        
        # Suppression des DPE pour lesquels au moins un des json est vide
        if json_dpe1==[] or json_dpe2 ==[] : 
            df_epc_evolution.drop(labels=index, inplace = True) 
        else:
            liste_dpe_diff_ligne = [k for k in dpe_diff[0].keys()] # todo : ajouter dictionnaire dpe_diff[0] carrement
            # Suppression des DPE dont les différences ne sont pas liées à des manipulations 
            for variable in set_variables_intermediaires:
                if variable in liste_dpe_diff_ligne:
                    liste_dpe_diff_ligne.remove(variable)
            if liste_dpe_diff_ligne == []:
                df_epc_evolution.drop(labels=index, inplace = True)
            else:
                liste_dpe_diff.append(liste_dpe_diff_ligne) # liste des champs modifiés entre les deux DPE

    df_epc_evolution["dpe_diff"] = liste_dpe_diff # todo : pas top de faire ça car on est pas sûr que la ligne corresponde bien a l'index ? ce serait mieux de manipuler les df directement
    # todo: transformer liste_dpe_diff en série qu'on ajoutera a la fin au dataframe

    
    return df_epc_evolution


def plot_nb_champs_modifies(dep_code, period): #todo : enlever insert et delete
    df_epc_evolution = delete_dpe_copies(dep_code, period = period)
    counts.drop(index = set_admin_and_geog, inplace = True, errors='ignore') # todo : enlever les champs administratifs qui nous intéressent pas ?
    df_epc_evolution['nb_champs_modifies'] = df_epc_evolution.dpe_diff.map(len)
    

def hist_champs_modifies(dep_code, period, filter_dpe = None, top_n = None):
    
    departement = Departement(dep_code)
    df_epc_evolution = delete_dpe_copies(dep_code, period = period)
    
    # Filtrage des DPE qui sont passés dans une meilleure classe ou inversement
    if filter_dpe == 'better_dpe_only':
        df_epc_evolution = df_epc_evolution[df_epc_evolution.second_epc < df_epc_evolution.first_epc]
    elif filter_dpe == 'worse_dpe_only':
        df_epc_evolution = df_epc_evolution[df_epc_evolution.second_epc > df_epc_evolution.first_epc]
    
    # Dépliage de la colonne de listes en autant de lignes qu'il y a d'éléments par liste
    df_exploded = df_epc_evolution.explode('dpe_diff')
    # Décompte des occurrences de chaque champ2491E0874510Q
    counts = df_exploded['dpe_diff'].value_counts() #ascending = True)
    counts.drop(index = set_admin_and_geog, inplace = True, errors='ignore')
    
    counts_norm = counts / len(df_epc_evolution) *100 # en % du nb de paires de DPE successifs 
       
    
    # Tracé histogramme (bar chart) des {top_n} champs les plus fréquemment modifiés
    # Titre figure : Champs les plus modifiés entre deux DPE successifs de moins de {period} jours\n{departement.name} - {departement.code}, N={len(df_epc_evolution)}
    fig,ax = plt.subplots(figsize=(10, 2/10*len(counts_norm.head(top_n))))                  
    counts_norm.head(top_n).plot(kind='barh')
    if filter_dpe == 'better_dpe_only': 
        ax.set_title(f'DPE successifs améliorés en moins de {period} jours ({departement.name} - {departement.code}, N={len(df_epc_evolution)})\n ')
    elif filter_dpe == 'worse_dpe_only': # todo: changer titre car pas convaincue
        ax.set_title(f'DPE successifs empirés en moins de {period} jours ({departement.name} - {departement.code}, N={len(df_epc_evolution)})\n ')
    else :   
        ax.set_title(f'{departement.name} - {departement.code}, N={len(df_epc_evolution)}. period = {period} jours') #, fontsize=10)
        # fig.suptitle(f'DPE successifs de moins de {period} jours ({departement.name} - {departement.code}, N={len(df_epc_evolution)})\n ') #, fontsize=10)
    ax.set_xlabel("Nombre d'occurrences (%)")
    ax.set_ylabel(None)
    ax.set_xlim(0, max(counts_norm)+5)
    for container in ax.containers:
        ax.bar_label(container, fmt= "%.1f", padding = 1)
    ax.invert_yaxis()
    plt.tight_layout()
    
    
    # Définition du chemin de sauvegarde des histogrammes des champs modifiés
    output_folder_ACM = os.path.join('output', '5. analyse_champs_modifies')
    os.makedirs(output_folder_ACM, exist_ok=True)
    existing_files = os.listdir(output_folder_ACM)
    
    # Enregistrement de la figure
    save_name = f'Hist_champs_modifies_sur_{period}_jours_dep_{dep_code}.pdf'
    if filter_dpe == 'better_dpe_only':
        save_name = save_name.replace('.pdf','_better_dpe_only.pdf')
    elif filter_dpe == 'worse_dpe_only':
        save_name = save_name.replace('.pdf','_worse_dpe_only.pdf')

    plt.savefig(os.path.join(output_folder_ACM,save_name), bbox_inches='tight')
    
    
    plt.show()
    
    return
    



def regplot_influence_variable(dep_code, variable, period, display_class, relatif, non_zero_variation_only):
    # todo : regression sans prendre en compte variation de surface nulle ?
    
    departement = Departement(dep_code)
    df_epc_evolution = filter_manipulated(dep_code, period = period)    
    
    df_epc_evolution = variable_diff(df_epc_evolution, variable = variable)
    
    # conso_5_usages_ep_m2_diff 
    # variable_diff = 
    
    if non_zero_variation_only:
        df_epc_evolution = df_epc_evolution[df_epc_evolution[f'{variable}_diff']!=0]
    
    if display_class : # pas très pertinent ?
        if relatif:
            fig,ax = plt.subplots(figsize=(5,5), dpi=300)
            sns.regplot(data=df_epc_evolution, x="epc_cons_diff_rel", y=f'{variable}_diff_rel', hue = 'second_epc', palette=etiquette_colors_dict, hue_order=list('ABCDEFG'), ax=ax) 
            ax.set_ylabel(f'Variation relative de {variable} (%)')
            ax.set_label('Variation relative de conso_5_usages_ep_m2 (%)')
            ax.set_ylim(-150, 150)
            ax.set_xlim(-50, 50)
        
        else :
            fig,ax = plt.subplots(figsize=(5,5), dpi=300)
            sns.regplot(data=df_epc_evolution, x="epc_cons_diff", y=f'{variable}_diff', hue = 'second_epc', palette=etiquette_colors_dict, hue_order=list('ABCDEFG'), ax=ax) 
            ax.set_ylabel(f'Variation de {variable}')
            ax.set_xlabel('Variation de conso_5_usages_ep_m2')
            
    else : 
        if relatif:
            fig,ax = plt.subplots(figsize=(5,5), dpi=300)
            sns.regplot(data=df_epc_evolution, x="epc_cons_diff_rel", y=f'{variable}_diff_rel', fit_reg=True, scatter_kws={'alpha':0.05}, ax=ax)
            ax.set_ylabel(f'Variation relative de {variable} (%)')
            ax.set_xlabel('Variation relative de conso_5_usages_ep_m2 (%)')
            ax.set_title(f'Ensemble des départements\nPériode de {period} jours, N={len(df_epc_evolution)}')

            lim = max(max(np.abs(plt.xlim())), max(np.abs(plt.ylim())))
            ax.set_ylim(-lim, lim)
            ax.set_xlim(-lim, lim) # todo 
            ax.set_ylim(-150, 150) # pour pouvoir voir sur la pente quelle variable varie plus vite
            ax.set_xlim(-150, 150)
            # ax.set_ylim(-40, 40) # pour pouvoir voir sur la pente quelle variable varie plus vite
            # ax.set_xlim(-40, 40)
        
        else :
            fig,ax = plt.subplots(figsize=(5,5), dpi=300)
            sns.regplot(data=df_epc_evolution, x="epc_cons_diff", y=f'{variable}_diff', ax=ax) 
            ax.set_ylabel(f'Variation de {variable}')
            ax.set_xlabel('Variation de conso_5_usages_ep_m2')

    plt.title(f"{departement.name} - {departement.code}\nPériode de {period} jours, N={len(df_epc_evolution)}")
    # plt.title(f"Corrélation entre les variations de {variable} et de consommation annuelle en énergie primaire"+" (kWh.m$^{-2}$)\n"+ f"{departement.name} - {departement.code}, N={len(df_epc_evolution)}")
    # plt.ylim(-200, 200)
    # plt.xlim(-400, 400)
    
    # Définition du chemin de sauvegarde des histogrammes des champs modifiés
    output_folder_ACM = os.path.join('output', '5. analyse_champs_modifies')
    os.makedirs(output_folder_ACM, exist_ok=True)
    existing_files = os.listdir(output_folder_ACM)
    
    # Enregistrement de la figure
    save_name = f'Correlation_{variable}_et_conso_energ_periode_{period}_jours_dep_{dep_code}.png'
    if display_class:
        save_name = save_name.replace('.png','_display_class.png')
    if relatif:
        save_name = save_name.replace('.png','_relatif.png')

    plt.savefig(os.path.join(output_folder_ACM,save_name), bbox_inches='tight')
    
    plt.show()
    plt.close()
 
    
#%% ===========================================================================
# script principal
# =============================================================================



def main():
        
    tic = time.time()
        
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')
    output_folder = os.path.join('output',today)
    os.makedirs(output_folder, exist_ok=True)
    
    national_scale = True
    dep_code = '91'
    departement = Departement(dep_code)
    
    period = 20 # jours d'écart maximal entre deux DPE successifs
    top_n = 10 # nombre de champs affichés sur l'histogramme des champs modifiés
    

    # test type_batiment_dpe=='maison' MAIS “nb_log”=!1 
    if False:
        test_dpe_logement, test_rel_batiment_groupe_dpe_logement, test_batiment_groupe_compile = get_bdnb(dep_code)
    
        test_batiment_groupe_compile = test_batiment_groupe_compile[['batiment_groupe_id','ffo_bat_nb_log']]
        test_batiment_groupe_compile = test_batiment_groupe_compile.compute() 
    
        test_rel_batiment_groupe_dpe_logement = test_rel_batiment_groupe_dpe_logement[['batiment_groupe_id','identifiant_dpe','adresse_brut']]
        test_rel_batiment_groupe_dpe_logement = test_rel_batiment_groupe_dpe_logement.compute()
        #doublons = test_rel_batiment_groupe_dpe_logement[test_rel_batiment_groupe_dpe_logement.duplicated(subset = ["identifiant_dpe"], keep=False)]
        
        # Filtrage des DPE arrêté 2021 méthode 3CL logement
        test_dpe_logement = test_dpe_logement[(test_dpe_logement.type_dpe =='dpe arrêté 2021 3cl logement') & (test_dpe_logement.type_batiment_dpe == 'maison')][['identifiant_dpe','type_batiment_dpe', 'date_etablissement_dpe','classe_bilan_dpe','surface_habitable_logement']]
        # test_dpe_logement = test_dpe_logement.set_index('identifiant_dpe') # pourquoi ?
        test_dpe_logement = test_dpe_logement.compute()
        
          
        test_join_id_dpe = test_batiment_groupe_compile.merge(test_rel_batiment_groupe_dpe_logement, how='inner', on='batiment_groupe_id')
        
        test_filter_individual = test_join_id_dpe.merge(test_dpe_logement, how='inner', on='identifiant_dpe') # on conserve seulement les DPE méthode 3CL 2021 correspondant aux maisons
        df_bug_maison = test_filter_individual[test_filter_individual.ffo_bat_nb_log != 1]

    # Enregistrement de la bdnb filtrée sur les logements individuels pour tous les départements
    if False:
        france = France()
        for dep in tqdm.tqdm(france.departements):
            dep_code = dep.code
            filter_bdnb_individual(dep_code=dep_code, force=False)
    
    
    # graphe de passage heatmap
    if True:        
        plot_heatmap(national_scale=national_scale, dep_code=dep_code, frequency=True, period = period)
        
    # Sankey diagram with Plotly
    if False: 
        plotly_sankey(national_scale=national_scale, period=period, dep_code=dep_code)
        
    # Sankey diagram with pySankey (ne met pas les classes énergétiques dans l'ordre...)
    if False:
        df_epc_evolution = filter_manipulated(dep_code)
        sankey(df_epc_evolution["first_epc"], df_epc_evolution["second_epc"], aspect=20, colorDict = etiquette_colors_dict, fontsize=12)


    # Plot histogramme variable_diff
    if False:
        variable = 'epc_cons'
        ecart_relatif = False
        
        plot_variable_diff(national_scale=national_scale, period=period, dep_code=dep_code, variable=variable, ecart_relatif = ecart_relatif)

    
    # Download DPE details
    if False: 
        download_dpe_json('2591E2079598F')
        # download_dpe_json('2275E2157068C') # 14 brillat savarin
        
    # analyse des champs modifiés
    if False:  
        # filter_bdnb_individual('33',True)
        # filter_bdnb_individual('44',True)
        # filter_bdnb_individual('69',True)
        
        # delete_dpe_copies_paris = delete_dpe_copies('75', 1, 30)
        hist_champs_modifies(dep_code, period = period, top_n = top_n) #filter_dpe='worse_dpe_only')
        dpe_id1 = '2591E2951057W' # dpe au json vide
        
        # dpe qui se ressemblent
        dpe_id1 = '2191E0101828Z'
        dpe_id2 = '2191E0102146F'
        
        # DPE avec $insert dans dpe_diff
        dpe_id1 = '2291E1697414S'
        dpe_id2 = '2291E1746964M'
        #test= diff_dpe_data(dpe_id1, dpe_id2)
        
    # influence des variations d'une variable sur conso_5_usages_ep_m2
    if False:
        variable = 'surface'
        regplot_influence_variable(dep_code=dep_code, variable=variable, period=period, display_class=False, relatif=True, non_zero_variation_only=True)

            
    if False:
        # plot_gain_period(True, '91', 120, 5)
        plot_distrib_dpe_sucessifs(False, )
    
    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')


if __name__ == '__main__':
    main()
