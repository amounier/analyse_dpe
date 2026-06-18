#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:23:12 2026

@author: audrey
"""

# ATTENTION : run "pip install jsondiff" et "pip install selenium" dans la console au préalable !

import time
import requests
import io
import os
import geopandas as gpd
import numpy as np
import pandas as pd
from IPython.display import display
import cartopy.io.img_tiles as cimgt
import cartopy.geodesic as cgeo
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import subprocess
import dask_geopandas 
import re
import warnings
from urllib.request import urlopen, Request
from PIL import Image
from datetime import date
from pyogrio.errors import DataSourceError
from urllib.error import HTTPError

import seaborn as sns
import plotly.io as pio
pio.renderers.default='browser'
import plotly.graph_objects as go
from pySankey.sankey import sankey
import json
import jsondiff as jd
from jsondiff import diff
import pprint
import tqdm

from utils import etiquette_colors_dict
from administrative import Departement, France, draw_departement_map
from download import get_bdnb, draw_local_map, neighbourhood_map, get_batiment_groupe_infos


def filter_bdnb_individual(dep_code, force):
    """
    Filtrage des DPE 2021 3CL associés à des logements individuels (maisons).

    Parameters
    ----------
    dep_code : str
        code du departement.

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
    
    # Définsankeyition du nom du fichier final
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
        # bdnb_dpe_logement = bdnb_dpe_logement.set_index('identifiant_dpe') # pourquoi ?
        bdnb_dpe_logement = bdnb_dpe_logement.compute()
        bdnb_dpe_logement.dropna(inplace = True) # certains DPE n'ont pas de surface associée (2021) # todo : enlever aussi si pas d'adresse ?
        
        bdnb_join_id_dpe = bdnb_batiment_groupe_compile.merge(bdnb_rel_batiment_groupe_dpe_logement, how='inner', on='batiment_groupe_id')
        
        bdnb_filter_individual = bdnb_join_id_dpe.merge(bdnb_dpe_logement, how='inner', on='identifiant_dpe') # on conserve seulement les DPE méthode 3CL 2021 correspondant logements indiv
        doublons = bdnb_filter_individual[bdnb_filter_individual.duplicated(subset = ["identifiant_dpe"], keep=False)] # pour observer les id_dpe en doublons (car correspondants à plusieurs bâtiments à la fois)
        # display(doublons)
        bdnb_filter_individual.drop_duplicates(subset = ["identifiant_dpe"], keep=False, inplace = True) # pour supprimer tous les id_dpe associés à plusieurs bâtiments à la fois
        
        
        bdnb_filter_individual.to_csv(os.path.join(output_folder, save_name), index = False)

    
    else:
        bdnb_filter_individual = pd.read_csv(os.path.join(output_folder, save_name), parse_dates=['date_etablissement_dpe'], date_format='%Y-%m-%d %H:%M:%S')

            
    return bdnb_filter_individual


# %%

def filter_manipulated(dep_code, period = 30): 
# todo : exclure DPE identiques fait le meme jour = doublons ? verifier que meme infos détaillées ou pas
    """
    Identification des bâtiments ayant calculés plusieurs DPEs .
    
    Parameters
    ----------
    dep_code : str
        code du departement.
    period : int
        écart de temps maximal entre deux DPE successifs avant de considérer que des rénovations énergétiques ont pu avoir lieu.

    Returns
    -------
    df_epc_evolution : pandas DataFrame
        DataFrame des paires de DPEs successifs pour chaque bâtiment du département.
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
                    'first_epc_date': group.iloc[i]['date_etablissement_dpe'].strftime("%d %b %Y"), # todo : laisser en format date ?
                    'first_epc_cons': group.iloc[i]['conso_5_usages_ep_m2'],
                    'first_epc': group.iloc[i]['classe_bilan_dpe'],
                    'surface_1' : group.iloc[i]['surface_habitable_logement'],
                    'second_epc_id': group.iloc[i+1]['identifiant_dpe'],
                    'second_epc_date': group.iloc[i+1]['date_etablissement_dpe'].strftime("%d %b %Y"),
                    'second_epc_cons': group.iloc[i+1]['conso_5_usages_ep_m2'],
                    'second_epc': group.iloc[i+1]['classe_bilan_dpe'],
                    'surface_2' : group.iloc[i+1]['surface_habitable_logement']
                }) #'batiment_groupe_id': group.name, # groupby fait des Series (cf ci-dessous)

        return pd.DataFrame(consecutive_pairs)

    df_epc_evolution = df_sorted.groupby('batiment_groupe_id').apply(extract_consecutive_dpe, period=period)
    
    # df_epc_evolution = df_epc_evolution.reset_index(drop=True) # pour nettoyer colonne inutile
    
    df_epc_evolution['conso_diff'] =  df_epc_evolution.second_epc_cons - df_epc_evolution.first_epc_cons
    df_epc_evolution['conso_diff_rel'] =  df_epc_evolution.conso_diff / ((df_epc_evolution.second_epc_cons + df_epc_evolution.first_epc_cons)/2) *100


    df_epc_evolution = variable_diff(dep_code, df_epc_evolution, variable = 'surface',  plot_variable_evolution = False, ecart_relatif = True)

    return df_epc_evolution   
    


def variable_diff(dep_code, df_epc_evolution, variable = 'surface',  plot_variable_evolution = True, ecart_relatif = True): #todo : bizarre d'avoir dep_code ici ?
    """
    Ajoute des colonnes {variable}_diff et {variable}_diff_rel (différence relative à la moyenne des deux DPEs) au DataFrame df_epc_evolution, et peut également plot l'histogramme des variations de cette variable.

    Parameters
    ----------
    dep_code : str
        code du departement.
    df_epc_evolution : pandas DataFrame
        DataFrame initial.
    variable : str, optional
        nom de la variable d'intérêt. The default is 'surface'.
    plot_variable_evolution : boolean, optional
        plot l'écart de {variable} entre DPEs successifs pour l'ensemble du département. The default is True.
    ecart_relatif : boolean, optional
        trace l'histogramme en écart relatif et non absolu. The default is True.

    Returns
    -------
    df_epc_evolution : pandas DataFrame
        tableau avec deux colonnes {variable}_diff et {variable}_diff_rel en plus.
    """
    
    departement = Departement(dep_code)

    df_epc_evolution[f'{variable}_diff'] =  df_epc_evolution[f"{variable}_2"] - df_epc_evolution[f"{variable}_1"]
    df_epc_evolution[f'{variable}_diff_rel'] =  df_epc_evolution[f'{variable}_diff'] / ((df_epc_evolution[f"{variable}_1"]+df_epc_evolution[f"{variable}_2"])/2) *100  # écart relatif par rapport à la moyenne des variables entre les deux DPEs


    if plot_variable_evolution:
        fig, ax = plt.subplots(figsize=(5,5), dpi=300)
        
        if ecart_relatif :
            df_epc_evolution.hist(column=f'{variable}_diff_rel', ax=ax, bins=300, color='k')
            
            ax.set_title(f"Ecart relatif de {variable} entre deux DPE successifs\n({departement.name} - {departement.code}, N={len(df_epc_evolution)})") # changer nom figure ?
            ax.set_ylabel("Nombre d'observations")
            ax.set_xlabel(f"Ecart de {variable} entre DPE successifs, en %")
            
            ax.set_xlim([-100,100])
            
        else:
            df_epc_evolution.hist(column=f'{variable}_diff', ax=ax, bins=np.linspace(), color='k') # todo : mettre liste de valeur avec sequence, eventuellement utiliser seaborn, pour centre l'histogramme
            
            ax.set_title(f"Evolution de {variable} entre deux DPE successifs\n({departement.name} - {departement.code}, N={len(df_epc_evolution)})")
            ax.set_ylabel("Nombre d'observations")
            ax.set_xlabel(f"Différence de {variable} entre DPE successifs")
            
            ax.set_xlim([-100,100])

        # Enregistrement de la figure
        output_folder_hist_variations = os.path.join('output', 'hist_variations_entre_DPE_successifs')
        os.makedirs(output_folder_hist_variations, exist_ok=True)
        existing_files = os.listdir(output_folder_hist_variations)
        
        if ecart_relatif :
            save_name = f'Ecart_relatif_surface_entre_DPE_successifs_dep{dep_code}.png'
        else:
            save_name = f'Ecart_surface_entre_DPE_successifs_dep{dep_code}.png'
        plt.savefig(os.path.join(output_folder_hist_variations,save_name), bbox_inches='tight')

        
        surface_manip_count_1 = len(df_epc_evolution[df_epc_evolution.surface_diff != 0])
        surface_manip_count_1_percent = surface_manip_count_1 / len(df_epc_evolution) *100
        print(f'Nombre de modifications de surface non nulles pour le département {dep_code} :', surface_manip_count_1, f'parmi N={len(df_epc_evolution)} ({surface_manip_count_1_percent:.1f} %)')
          
        surface_manip_count_2 = len(df_epc_evolution) - len(df_epc_evolution[(-1 < df_epc_evolution.surface_diff) & (df_epc_evolution.surface_diff < 1)])
        surface_manip_count_2_percent = surface_manip_count_2 / len(df_epc_evolution) *100
        print(f'Nombre de modifications de surface supérieures à +-1 m2 pour le département {dep_code} :', surface_manip_count_2, f'parmi N={len(df_epc_evolution)} ({surface_manip_count_2_percent:.1f} %)')

    
    return df_epc_evolution


def plot_gain_period(dep_code, period):
    
    df_epc_evolution = filter_manipulated(dep_code, period)
    df_epc_evolution['ecart_date'] = pd.to_datetime(df_epc_evolution.second_epc_date) - pd.to_datetime(df_epc_evolution.first_epc_date)
    df_epc_evolution['ecart_date'] = df_epc_evolution['ecart_date'].dt.days
    # Calcul du gain moyen d'étiquette sur l'ensemble des paires de DPEs successifs
    epc_order = {'A':7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1} # on attribue valeur à chaque classe
    df_epc_evolution['first_val'] = df_epc_evolution['first_epc'].map(epc_order) 
    df_epc_evolution['second_val'] = df_epc_evolution['second_epc'].map(epc_order)
    df_epc_evolution['gain_etiquette'] = df_epc_evolution['second_val'] - df_epc_evolution['first_val']


    # Tracé histogramme des ecart_date
    # bins_sequence = list(range(0,period+1)) 
    # fig,ax = plt.subplots(figsize=(5,5), dpi=300) 
    # df_epc_evolution.hist(column='ecart_date', ax=ax, bins=bins_sequence, color='k')
    # df_epc_evolution.hist(column='ecart_date', ax=ax, bins=20, color='k') # autre version


    df_gain_moyen = df_epc_evolution.groupby(['ecart_date'])['gain_etiquette'].mean()

    # Tracé du gain moyen d'étiquette sur périodes de temps
    period_list = list(range(0,120,5))
    gain_moyen_etiquette = []
    #for period in tqdm.tqdm(period_list[:-1]):
    for i in range(0,23):
        borne_inf = period_list[i]
        borne_sup= period_list[i+1]
        gain_moyen_sur_cette_periode = df_epc_evolution[(df_epc_evolution.ecart_date > borne_inf) & (df_epc_evolution.ecart_date < borne_sup)]['gain_etiquette'].mean()
        gain_moyen_etiquette.append(gain_moyen_sur_cette_periode)


    fig,ax = plt.subplots(figsize=(5,5), dpi=300)  
    ax.plot(gain_moyen_etiquette, label = 'gain_moyen_etiquette') # pb dimension liste et nptq
    plt.show()

    
    # sns.regplot(data=df_epc_evolution, x='ecart_date', y='gain_etiquette') 


    return


def analyse_gain_etiquette(dep_code, period):
    
    df_epc_evolution = filter_manipulated(dep_code, period = period)[['first_epc','second_epc']] # version rapide non nettoyée
    N = len(df_epc_evolution)

    # Calcul de la part des DPE stables
    stable_mask = df_epc_evolution['first_epc'] == df_epc_evolution['second_epc']
    part_dpe_stables = stable_mask.sum() / N
    
    # Calcul du gain moyen d'étiquette sur l'ensemble des paires de DPEs successifs
    epc_order = {'A':7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1} # on attribue valeur à chaque classe
    df_epc_evolution['first_val'] = df_epc_evolution['first_epc'].map(epc_order) 
    df_epc_evolution['second_val'] = df_epc_evolution['second_epc'].map(epc_order)
    df_epc_evolution['gain_etiquette'] = df_epc_evolution['second_val'] - df_epc_evolution['first_val']
    gain_moyen_etiquette = df_epc_evolution['gain_etiquette'].mean()

    # Calcul du gain moyen d'étiquette parmi les DPEs modifiés uniquement
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


def dicts_dep_gain_moyen_etiquette(period):
    
    france = France()
    
    dict_part_dpe_stables  = {d:[] for d in france.departements} 
    dict_gain_moyen_etiquette  = {d:[] for d in france.departements} 
    dict_gain_moyen_etiquette_parmi_modif  = {d:[] for d in france.departements} 

    for dep in tqdm.tqdm(france.departements) :
        dep_code = dep.code
        print(dep) 
        part_dpe_stables, gain_moyen_etiquette, gain_moyen_etiquette_parmi_modif = analyse_gain_etiquette(dep_code, period=period)
        dict_part_dpe_stables[dep] = part_dpe_stables
        dict_gain_moyen_etiquette[dep] = gain_moyen_etiquette
        dict_gain_moyen_etiquette_parmi_modif[dep] = gain_moyen_etiquette_parmi_modif

    return dict_part_dpe_stables, dict_gain_moyen_etiquette, dict_gain_moyen_etiquette_parmi_modif


# %%


def plot_heatmap(dep_code, frequency, period = 30):
    """
    Formate un DataFrame de comparaison de DPE successifs en un DataFrame 2D comptant les évolutions entre classes.

    Parameters
    ----------
    df_epc_evolution : pandas DataFrame
        DataFrame de comparaison de DPE successifs (issu de filter_manipulated)
    frequency : boolean
        if True, trace la heatmap en fréquence et non en absolu

    Returns
    -------
    df_heatmap : pandas DataFrame
        Matrice des transitions entre classes de DPE successifs.
        Colonnes = 1er DPE, Lignes = 2e DPE.
    """
    
    departement = Departement(dep_code)
    
    # Version rapide non nettoyée
    df_epc_evolution = filter_manipulated(dep_code, period = period)
    
    # Version nettoyée des DPEs en double : (beaucoup plus long car il faut télécharger tous les json) mais ne change pas grand chose --> a eviter
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
    annot = np.where(annot != 0, annot, "") # todo: modifier pour garder que cases > 0.1 ?


    # Tracé de la figure    
    fig,ax = plt.subplots(figsize=(5,5), dpi=300)

    cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
    posn = ax.get_position()
    cbar_ax.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
    
    if frequency:
        ax = sns.heatmap(df_heatmap, ax=ax, vmin=0, vmax=25, annot=annot, fmt="", cmap='bone_r', cbar=True, cbar_ax=cbar_ax, cbar_kws={'label':'Percentage (%)'})
    else:
        ax = sns.heatmap(df_heatmap, ax=ax, annot=annot, fmt="", cmap='bone_r', cbar_ax=cbar_ax,cbar=True,cbar_kws={'label':"Nombre d'observations"})
    
    ax.set_title(f'Modification des DPE sur une période de {period} jours\n{departement.name} - {departement.code}, N={len(df_epc_evolution)}')
    for spine in ax.spines.values():
        spine.set_visible(True)
    for spine in cbar_ax.spines.values():
        spine.set_visible(True)
    ax.set_ylabel('Second EPC')
    ax.set_xlabel('First EPC')
    
    
    # Définition du chemin de sauvegarde des heatmap
    output_folder_heatmap = os.path.join('output', 'heatmap')
    os.makedirs(output_folder_heatmap, exist_ok=True)
    existing_files = os.listdir(output_folder_heatmap)
    
    # Enregistrement de la figure
    save_name = f'DPE_manipulation_classes_{dep_code}_sur_{period}_jours.png'
    if frequency:
        save_name = save_name.replace('.png','_frequency.png')

    plt.savefig(os.path.join(output_folder_heatmap,save_name), bbox_inches='tight')
    
    
    plt.show()


    return 



def plot_pysankey(df_epc_evolution):
    
    sankey(df_epc_evolution["first_epc"], df_epc_evolution["second_epc"], aspect=20, colorDict = etiquette_colors_dict, fontsize=12)

    return 



def plotly_sankey(dep_code, period):
    
    departement = Departement(dep_code)
    output_folder_sankey = os.path.join('output', 'sankey diagram')
    os.makedirs(output_folder_sankey, exist_ok=True)
    
    df_epc_evolution = filter_manipulated(dep_code, plot_surface_evolution = False, period = period)
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
    
    # Personnaliser la disposition pour séparer les deux groupes de nœuds
    fig.update_layout(
        title_text=f"Transitions de classes entre DPE successifs ({departement.name} - {departement.code}, N={len(df_epc_evolution)}, écart max. entre DPE = {period} jours)",
        font_size=30,
        title_font_size=30,
        #font_color = 'black', 
        font_shadow = "auto", # 'None' si pas d'ombre
        #font_family='Arial Black' # todo: que mettre ?
        # todo : mettre label a l'exterieur
    )
    
    
    # todo : enregistrer figure (pb avec kaleido)
    # save_name = f"Transitions de classes entre DPE successifs ({departement.name} - {departement.code}, N={len(df_epc_evolution)}, period = {period} jours).png"
    # fig.write_image(os.path.join(output_folder_sankey,save_name))
    # pio.savefig(os.path.join(output_folder_sankey,save_name), bbox_inches='tight')

    
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
        Ensemble des DPEs présents dans la table df_epc_evolution.
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
    Création d'un DataFrame pour comparer les variables modifiées entre deux DPEs successifs. Ne tient pas compte des variables finales issues de calculs (conso_5_usages etc)

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
    
    # Jointure des détails des DPEs successifs
    df_dpe1 = pd.DataFrame().from_dict(json_dpe1['results'][0], orient='index', columns =['First DPE'])
    df_dpe2 = pd.DataFrame().from_dict(json_dpe2['results'][0], orient='index', columns =['Second DPE'])

    comparison_df = df_dpe1.join(df_dpe2, how='outer') # on conserve tous les champs, y compris ceux qui ne sont renseignés que dans un seul des deux df
    comparison_df = df_changing_variables.join(comparison_df)
    
    print(comparison_df)

    return comparison_df



def delete_dpe_copies(dep_code, period): # todo : prendre plutot df_epc_evolution en argument ? changer le nom
    """
    Nettoie df_epc_evolution en enlevant les DPEs successifs qui sont en réalité identiques.

    Parameters
    ----------
    dep_code : str
        code du departement.

    Returns
    -------
    df_epc_evolution : pandas DataFrame
        DataFrame des paires de DPEs successifs avec ajout d'une colonne "dpe_diff" listant les champs modifiés.
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
                liste_dpe_diff.append(liste_dpe_diff_ligne) # liste des champs modifiés entre les deux DPEs

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
    
    # Filtrage des DPEs qui sont passés dans une meilleure classe ou inversement
    if filter_dpe == 'better_dpe_only':
        df_epc_evolution = df_epc_evolution[df_epc_evolution.second_epc < df_epc_evolution.first_epc]
    elif filter_dpe == 'worse_dpe_only':
        df_epc_evolution = df_epc_evolution[df_epc_evolution.second_epc > df_epc_evolution.first_epc]
    
    # Dépliage de la colonne de listes en autant de lignes qu'il y a d'éléments par liste
    df_exploded = df_epc_evolution.explode('dpe_diff')
    # Décompte des occurrences de chaque champ2491E0874510Q
    counts = df_exploded['dpe_diff'].value_counts(ascending = True)
    counts.drop(index = set_admin_and_geog, inplace = True, errors='ignore')
    
    counts_norm = counts / len(df_epc_evolution) *100 # en % du nb de paires de DPE successifs 
       
    
    # Tracé histogramme (bar chart) des {top_n} champs les plus fréquemment modifiés
    # Titre figure : Champs les plus modifiés entre deux DPE successifs de moins de {period} jours\n{departement.name} - {departement.code}, N={len(df_epc_evolution)}
    fig,ax = plt.subplots(figsize=(10, 15))                  
    counts_norm.head(top_n).plot(kind='barh')
    if filter_dpe == 'better_dpe_only': 
        fig.suptitle(f'DPE successifs améliorés en moins de {period} jours ({departement.name} - {departement.code}, N={len(df_epc_evolution)})\n ')
    elif filter_dpe == 'worse_dpe_only': # todo: changer titre car pas convaincue
        fig.suptitle(f'DPE successifs empirés en moins de {period} jours ({departement.name} - {departement.code}, N={len(df_epc_evolution)})\n ')
    else :   
        fig.suptitle(f'DPE successifs de moins de {period} jours ({departement.name} - {departement.code}, N={len(df_epc_evolution)})\n ') #, fontsize=10)
    ax.set_xlabel("Nombre d'occurrences (%)")
    ax.set_ylabel(None)
    ax.set_xlim(0, max(counts_norm)+5)
    for container in ax.containers:
        ax.bar_label(container, fmt= "%.1f", padding = 1)
    plt.tight_layout()
    
    
    # Définition du chemin de sauvegarde des histogrammes des champs modifiés
    output_folder_ACM = os.path.join('output', 'analyse_champs_modifies')
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
    



def regplot_influence_variable(dep_code, variable, period, display_class, relatif):
    # todo : regression sans prendre en compte variation de surface nulle ?
    
    departement = Departement(dep_code)
    df_epc_evolution = filter_manipulated(dep_code, period = period)    
    variable_diff(dep_code, df_epc_evolution, variable = variable,  plot_variable_evolution = False)
    # conso_5_usages_ep_m2_diff 
    # variable_diff = 
    
    if display_class :
        if relatif:
            df_epc_evolution = df_epc_evolution[df_epc_evolution[f'{variable}_diff_rel']!=0]

            sns.lmplot(data=df_epc_evolution, x="conso_diff_rel", y=f'{variable}_diff_rel', hue = 'second_epc', palette=etiquette_colors_dict, hue_order=list('ABCDEFG')) 
            plt.ylabel(f'Variation relative de {variable} (%)')
            plt.xlabel('Variation relative de conso_5_usages_ep_m2 (%)')
            plt.ylim(-150, 150)
            plt.xlim(-50, 50)
        
        else :
            sns.lmplot(data=df_epc_evolution, x="conso_diff", y=f'{variable}_diff', hue = 'second_epc', palette=etiquette_colors_dict, hue_order=list('ABCDEFG')) 
            plt.ylabel(f'Variation de {variable}')
            plt.xlabel('Variation de conso_5_usages_ep_m2')
            
    else : 
        if relatif:
            df_epc_evolution = df_epc_evolution[df_epc_evolution[f'{variable}_diff_rel']!=0]
 
            sns.lmplot(data=df_epc_evolution, x="conso_diff_rel", y=f'{variable}_diff_rel', fit_reg=False)
            plt.ylabel(f'Variation relative de {variable} (%)')
            plt.xlabel('Variation relative de conso_5_usages_ep_m2 (%)')
            #plt.axis('equal')
            lim = max(max(np.abs(plt.xlim())), max(np.abs(plt.ylim())))
            # plt.ylim(-lim, lim)
            # plt.xlim(-lim, lim) #todo 
            plt.ylim(-150, 150)
            plt.xlim(-150, 150)
        
        else :
            sns.lmplot(data=df_epc_evolution, x="conso_diff", y=f'{variable}_diff') 
            plt.ylabel(f'Variation de {variable}')
            plt.xlabel('Variation de conso_5_usages_ep_m2')
    plt.title(f"Corrélation entre les variations de {variable} et de consommation annuelle en énergie primaire"+" (kWh.m$^{-2}$)\n"+ f"{departement.name} - {departement.code}, N={len(df_epc_evolution)}")
    # plt.ylim(-200, 200)
    # plt.xlim(-400, 400)
    
    # Définition du chemin de sauvegarde des histogrammes des champs modifiés
    output_folder_ACM = os.path.join('output', 'analyse_champs_modifies')
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
    
    
    dep_code = '91'
    departement = Departement(dep_code)
    
    period = 30 # jours d'écart maximal entre deux DPE successifs
    top_n = None # nombre de champs affichés sur l'histogramme des champs modifiés
    
    
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


    
    # graphe de passage heatmap
    if False:        
        plot_heatmap(dep_code, frequency=True, period = 40)
        
        
    # Sankey diagram with Plotly
    if False: 
        plotly_sankey(dep_code, 1, 30)
        
    
    # Sankey diagram with pySankey
    if False:

        bdnb_df = filter_bdnb_individual(dep_code) # prend du temps je pense
        
        df_epc_evolution = filter_manipulated(bdnb_df)
        
        fig = sankey(df_epc_evolution["first_epc"], df_epc_evolution["second_epc"], aspect=20, colorDict = etiquette_colors_dict, fontsize=12)

        # Get current figure
        # fig = plt.gcf()
        
# =============================================================================
#         # Set size in inches
#         fig.set_size_inches(6, 6)
#         
#         # Set the color of the background to white
#         fig.set_facecolor("w")
# =============================================================================
        
        # Save the figure
        save_name = 'sankey_diagram_evolution_dpe_successifs_{dep_code}'
        fig.savefig(save_name, bbox_inches="tight", dpi=150)

    
    # Download DPE details
    if False: 
        download_dpe_json('2591E2079598F')
        # download_dpe_json('2275E2157068C') # 14 brillat savarin
        
        
    # analyse des champs modifiés
    if True:  
        # filter_bdnb_individual('33',True)
        # filter_bdnb_individual('44',True)
        # filter_bdnb_individual('69',True)
        
        # delete_dpe_copies_paris = delete_dpe_copies('75', 1, 30)
        hist_champs_modifies(dep_code, period = period, top_n = top_n) #filter_dpe='worse_dpe_only')
        dpe_id1 = '2591E2951057W' # dpe au json vide
        
        # dpe qui se ressemblent
        dpe_id1 = '2191E0101828Z'
        dpe_id2 = '2191E0102146F'
        
        # dpes avec $insert dans dpe_diff
        dpe_id1 = '2291E1697414S'
        dpe_id2 = '2291E1746964M'
        #test= diff_dpe_data(dpe_id1, dpe_id2)
        
    # influence des variations d'une variable sur conso_5_usages_ep_m2
    if False:
        variable = 'surface'
        regplot_influence_variable(dep_code, variable, period = period, display_class=False, relatif=True)
    
    if False:
        bdnb_filter_individual = filter_bdnb_individual('91',30)
        
    
    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')


if __name__ == '__main__':
    main()
