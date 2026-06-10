#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:23:12 2026

@author: audrey
"""

# ATTENTION : run "pip install jsondiff" dans la console au préalable !

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
from jsondiff import diff
import pprint

from utils import etiquette_colors_dict
from administrative import Departement, France
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
        bdnb_dpe_logement = bdnb_dpe_logement[(bdnb_dpe_logement.type_dpe =='dpe arrêté 2021 3cl logement') & (bdnb_dpe_logement.type_batiment_dpe == 'maison')][['identifiant_dpe','type_batiment_dpe', 'date_etablissement_dpe','classe_bilan_dpe','surface_habitable_logement']]
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

def filter_manipulated(dep_code, plot_surface_evolution = True, surface_gap = 1, period = 30, ecart_relatif = True): # todo: rajouter condition sur surface ?
# todo : exclure DPE identiques fait le meme jour = doublons ? verifier que meme infos détaillées ou pas
    """
    Identification des bâtiments ayant calculés plusieurs DPEs .
    
    Parameters
    ----------
    dep_code : str
        code du departement.
    plot_surface_evolution : boolean, optional
        plot l'évolution des déclarations de surfaces du logement entre deux DPEs successifs.
    surface_gap : int
        écart de surface toléré avant de considérer que les deux logements sont différents.
    period : int
        écart de temps maximal entre deux DPE successifs avant de considérer que des rénovations énergétiques ont pu avoir lieu.

    Returns
    -------
    df_epc_evolution : pandas DataFrame
        Comparaison entre deux DPEs successifs d'un même bâtiment   
        Colonnes :
            - batiment_groupe_id (index) : identifiant du batiment
            - first_epc_id : identifiant du premier DPE calculé
            - first_epc_surf : surface renseignée lors du calcul du 1er DPE
            - first_epc : classe du premier DPE calculé (compris entre ‘A’ et ‘G’)
            - second_epc_id : identifiant du deuxième DPE calculé
            - second_epc_surf : surface renseignée lors du calcul du 2e DPE
            - second_epc : classe du deuxième DPE calculé (compris entre ‘A’ et ‘G’).
    """
    
    departement = Departement(dep_code)
    bdnb_df = filter_bdnb_individual(dep_code, force=False) # prend du temps si non stocké en .csv
    
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
            if date_diff <= period :
                consecutive_pairs.append({
                    'adresse_brut': group.iloc[i]['adresse_brut'],
                    'first_epc_id': group.iloc[i]['identifiant_dpe'],
                    'first_epc_date': group.iloc[i]['date_etablissement_dpe'].strftime("%d %b %Y"),
                    'first_epc': group.iloc[i]['classe_bilan_dpe'],
                    'first_epc_surf' : group.iloc[i]['surface_habitable_logement'],
                    'second_epc_id': group.iloc[i+1]['identifiant_dpe'],
                    'second_epc_date': group.iloc[i+1]['date_etablissement_dpe'].strftime("%d %b %Y"),
                    'second_epc': group.iloc[i+1]['classe_bilan_dpe'],
                    'second_epc_surf' : group.iloc[i+1]['surface_habitable_logement']
                }) #'batiment_groupe_id': group.name, # groupby fait des Series (cf ci-dessous)

        return pd.DataFrame(consecutive_pairs)

    df_epc_evolution = df_sorted.groupby('batiment_groupe_id').apply(extract_consecutive_dpe, period=period)
    
    # df_epc_evolution = df_epc_evolution.reset_index(drop=True) # pour nettoyer colonne inutile
    
    df_epc_evolution['surface_diff'] =  df_epc_evolution.second_epc_surf - df_epc_evolution.first_epc_surf
    df_epc_evolution['surface_diff_rel'] =  df_epc_evolution.surface_diff / df_epc_evolution.first_epc_surf *100  # écart relatif par rapport à la première surface déclarée


    if plot_surface_evolution: 
        fig, ax = plt.subplots(figsize=(5,5), dpi=300)
        
        if ecart_relatif :
            df_epc_evolution.hist(column='surface_diff_rel', ax=ax, bins=300, color='k')
            
            ax.set_title(f"Ecart relatif des surfaces déclarées entre deux DPE successifs\n({departement.name} - {departement.code}, N={len(df_epc_evolution)})") # changer nom figure ?
            ax.set_ylabel("Nombre d'observations")
            ax.set_xlabel("Ecart à la 1$^{ère}$ surface, en %")
            
            ax.set_xlim([-100,100])
            
        else:
            df_epc_evolution.hist(column='surface_diff', ax=ax, bins=300, color='k')
            
            ax.set_title(f"Evolution des surfaces déclarées entre deux DPE successifs\n({departement.name} - {departement.code}, N={len(df_epc_evolution)})")
            ax.set_ylabel("Nombre d'observations")
            ax.set_xlabel("Différence entre les deux surfaces")
            
            ax.set_xlim([-100,100])

        
        # Enregistrement de la figure
        output_folder_surface = os.path.join('output', 'hist_ecart_surface_entre_DPE_successifs')
        os.makedirs(output_folder_surface, exist_ok=True)
        existing_files = os.listdir(output_folder_surface)
        
        if ecart_relatif :
            save_name = f'Ecart_relatif_surface_entre_DPE_successifs_dep{dep_code}.png'
        else:
            save_name = f'Ecart_surface_entre_DPE_successifs_dep{dep_code}.png'
        plt.savefig(os.path.join(output_folder_surface,save_name), bbox_inches='tight')
        
        surface_manip_count_1 = len(df_epc_evolution[df_epc_evolution.surface_diff != 0])
        surface_manip_count_1_percent = surface_manip_count_1 / len(df_epc_evolution) *100
        print(f'Nombre de modifications de surface non nulles pour le département {dep_code} :', surface_manip_count_1, f'parmi N={len(df_epc_evolution)} ({surface_manip_count_1_percent:.1f} %)')
                
        surface_manip_count_2 = len(df_epc_evolution) - len(df_epc_evolution[(-surface_gap < df_epc_evolution.surface_diff) & (df_epc_evolution.surface_diff < surface_gap)])
        surface_manip_count_2_percent = surface_manip_count_2 / len(df_epc_evolution) *100
        print(f'Nombre de modifications de surface supérieures à +-{surface_gap} m2 pour le département {dep_code} :', surface_manip_count_2, f'parmi N={len(df_epc_evolution)} ({surface_manip_count_2_percent:.1f} %)')

       
    
    return df_epc_evolution   
    

# %%


def plot_heatmap(dep_code, frequency, surface_gap = 1, period = 30):
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

    df_epc_evolution = filter_manipulated(dep_code, plot_surface_evolution = False, surface_gap = surface_gap, period = period)
    
    
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
    
    ax.set_title(f'{departement.name} - {departement.code}, N={len(df_epc_evolution)}')
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
    save_name = f'DPE_manipulation_classes_{dep_code}.png'
    if frequency:
        save_name = save_name.replace('.png','_frequency.png')

    plt.savefig(os.path.join(output_folder_heatmap,save_name), bbox_inches='tight')
    
    
    plt.show()


    return 



def plot_pysankey(df_epc_evolution):
    
    sankey(df_epc_evolution["first_epc"], df_epc_evolution["second_epc"], aspect=20, colorDict = etiquette_colors_dict, fontsize=12)

    return 



def plotly_sankey(dep_code, surface_gap, period):
    
    departement = Departement(dep_code)
    output_folder_sankey = os.path.join('output', 'sankey diagram')
    os.makedirs(output_folder_sankey, exist_ok=True)
    
    df_epc_evolution = filter_manipulated(dep_code, plot_surface_evolution = False, surface_gap = surface_gap, period = period)

        
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



def download_dpe_details(dpe_id, force=False):
    """
    Téléchargement des fichiers de sorties des DPE (au format XML)

    Parameters
    ----------
    dpe_id : str
        identifiant du dpe.
    force : boolean, optional
        DESCRIPTION. The default is False.

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
            
            
    return
            
    
    


def diff_dpe_data(dpe_id1,dpe_id2):
    """
    Retourne les différences entre deux DPE successifs (json).

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
        json_dpe1 = json.load(f)
    with open(os.path.join('data','DPE','JSON','{}.json'.format(dpe_id2)), 'r') as f:
        json_dpe2 = json.load(f)
        
    # Compute the differences using jsondiff
    dpe_diff = diff(a=json_dpe1, b=json_dpe2)
    
    
    
# =============================================================================
# BONUS pretty print,     Based on https://github.com/matteobarzaghi/jsondiff

#     def convert_keys(obj):
#         """Recursively convert all dict keys to strings."""
#         if isinstance(obj, dict):
#             return {str(k): convert_keys(v) for k, v in obj.items()}
#         elif isinstance(obj, list):
#             return [convert_keys(i) for i in obj]
#         else:
#             return obj
# 
#     
#     # Pretty-print the diff to the console using pprint
#     pp = pprint.PrettyPrinter(indent=2)
#     pp.pprint(dpe_diff)
#     
#     # Convert keys to strings for JSON serialization
#     differences_converted = convert_keys(dpe_diff)
#     
#     # Save the pretty-printed diff to a file
#     pretty_diff = json.dumps(differences_converted, indent=2)
#     with open("diff_output.txt", "w", encoding="utf-8") as outfile:
#         outfile.write(pretty_diff)
# =============================================================================
        
    
    
    return json_dpe1, json_dpe2, dpe_diff



def compare_dpe_data(dpe_id1,dpe_id2):
    """
    Création d'un DataFrame pour comparer les variables modifiées entre deux DPE successifs. 

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
        print('Les deux DPE sont exactement identiques.')
        return 

    # Initialisation DataFrame des variables modifiées
    list_changing_variables = [k for k in dpe_diff['results'][0].keys()] # todo : faire un code plus beau sans dictionnaire de dictionnaire ?
    df_changing_variables = pd.DataFrame(index = list_changing_variables)
    
    # Jointure des détails des DPEs successifs
    df_dpe1 = pd.DataFrame().from_dict(json_dpe1['results'][0], orient='index', columns =['First DPE'])
    df_dpe2 = pd.DataFrame().from_dict(json_dpe2['results'][0], orient='index', columns =['Second DPE'])

    comparison_df = df_dpe1.join(df_dpe2)
    comparison_df = df_changing_variables.join(comparison_df)
    
    print(comparison_df)

    return 




def delete_dpe_copies(dep_code):
    """
    SUMMARY.

    Parameters
    ----------
    dep_code : str
        code du departement.

    Returns
    -------
    df_epc_evolution : pandas DataFrame
        DataFrame des paires de DPEs successifs avec ajout d'une colonne "dpe_diff" listant les champs modifiés.
    """
    
    # Récupération des DPE successifs effectués LE MEME JOUR (period = 0)
    df_epc_evolution = filter_manipulated(dep_code, plot_surface_evolution = False, surface_gap = 1, period = 0, ecart_relatif = True)
    
    # Téléchargement de tous les fichiers json nécessaires au calcul dpe_diff
    ensemble_dpe = set(pd.concat([df_epc_evolution["first_epc_id"], df_epc_evolution["second_epc_id"]]))    
    for dpe_id in ensemble_dpe : 
        download_dpe_details(dpe_id)
        time.sleep(1)   # 0.3s par dpe_id environ --> 200 req/min et 600 requêtes/min max
        # pas sûre que ça regle le pb
    
    # Création colonne "dpe_diff" listant les champs modifiés entre les deux DPE successifs
    liste_dpe_diff = []
    for index, row in df_epc_evolution.iterrows() :
        dpe_id1 = row["first_epc_id"]
        dpe_id2 = row["second_epc_id"]
        _, _, dpe_diff = diff_dpe_data(dpe_id1, dpe_id2)
        
        if dpe_diff['results'] == [] : 
            df_epc_evolution.drop(labels=index, inplace = True) # on supprime les DPE pour lesquels le json est vide
        else:
            liste_dpe_diff_ligne = [k for k in dpe_diff['results'][0].keys()]
            liste_dpe_diff.append(liste_dpe_diff_ligne) # liste des champs modifiés entre les deux DPEs

    df_epc_evolution["dpe_diff"] = liste_dpe_diff

    
    return df_epc_evolution

        
    
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
    
    
    # test "type_batiment_dpe” = maison MAIS “nb_log”=!1 
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
        plot_heatmap(dep_code, frequency=True, surface_gap = 1, period = 30)
        
        
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

    
    # Dowload DPE details
    if True: 
        download_dpe_details('2591E2079598F')
        # download_dpe_details('2275E2157068C') # 14 brillat savarin
        

    
    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')


if __name__ == '__main__':
    main()
