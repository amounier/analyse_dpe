#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:05:21 2026

@author: audrey
"""

import time
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sns
import statsmodels.api as sm

from administrative import  list_dep_code, Departement
from distribution import calcul_bunching_france, cut_france_bunching
from manipulation_dpe import dicts_dep_gain_moyen_etiquette



def tension_immob_dep():
    """
    Détermination des proportions de logements dans chacune des 5 zones du zonage ABC pour chaque département.
    Rappel : 
    Les 5 zones sont désignées par ordre de déséquilibre décroissant :
        Zone A : grandes agglomérations où la demande est élevée (Lyon, Lille, Marseille, Montpellier, etc.).
        Zone A bis : Paris et sa proche banlieue avec une demande locative extrêmement forte.
        Zone B1 : villes moyennes dynamiques et zones frontalières ou littorales.
        Zone B2 : petites villes ou communes où la tension est modérée.
        Zone C : zones détendues où l’offre dépasse la demande.

    Parameters
    ----------
    path_tension : str
        chemin vers le fichier Excel des tensions locatives par communes.
    path_logements : str
        chemin vers le fichier Excel du nombre de logements par communes.

    Returns
    -------
    df_tension_immob_dep : pandas DataFrame
        Part des logements du département dans chacune des 5 zones 
        Colonnes :  dep_code (index)  |  total_logements  |  part_A = part de logements du département en zone A  |  part_Abis  |  part_B1  |  part_B2  |  part_C  
    """
    # chemin vers le fichier Excel des tensions locatives par communes
    path_tension = os.path.join('data','INSEE','Liste ensemble des communes - Zonage ABC 5 septembre 2025.xlsx')
    
    # chemin vers le fichier Excel du nombre de logements par communes
    path_logements =  os.path.join('data','INSEE','logement-2022.xlsx')
    
    # Import du zonage de la tension locative par commune
    df_zonage_ABC = pd.read_excel(path_tension, usecols=['CODGEO', 'DEP',  'Zonage en vigueur depuis le 5 septembre 2025'])
    df_zonage_ABC = df_zonage_ABC[~df_zonage_ABC['DEP'].isin(['971', '972', '973', '974', '975', '976'])] 
     
    # Import du nombre de logements par communes
    df_logements = pd.read_excel(path_logements, names=['CODGEO', 'LIBGEO', 'LOGEMENTS_2022'], na_values = 'N/A - résultat non disponible', skiprows=3)
    
    # Jointure des deux dataframes
    df_communes = pd.merge(df_zonage_ABC, df_logements, on='CODGEO') #, 'LIBGEO'])
    df_communes.set_index('CODGEO', inplace=True)
    df_communes = df_communes[~df_communes['DEP'].isin(['971', '972', '973', '974', '975', '976'])] # on étudie seulement la France hexagonale


    # Création du df des tensions immobilières par département
    df_tension_immob_dep = pd.DataFrame(index=list_dep_code) 
    df_tension_immob_dep['departement'] = [Departement(f'{n}') for n in df_tension_immob_dep.index]
    
    # Colonne nombre total de logements par département
    df_tension_immob_dep['total_logements'] = df_communes.groupby('DEP')['LOGEMENTS_2022'].sum()
    
    # Colonnes nombre de logements dans chaque zone par département
    logements_par_zone = df_communes.groupby(['DEP', 'Zonage en vigueur depuis le 5 septembre 2025'])['LOGEMENTS_2022'].sum()
    logements_par_zone = logements_par_zone.unstack(fill_value=0)  # unstack transforme les zones ABC en colonnes, et fill_value remplace les valeurs manquantes par 0
    # todo : l'ajouter aussi au dataframe final ?
    
    # Colonnes parts des logements dans chaque zone par département
    parts_par_zone = logements_par_zone.div(df_tension_immob_dep['total_logements'], axis=0)
    parts_par_zone.columns = [f'part_{col}' for col in parts_par_zone.columns] # on renomme les colonnes
        
    df_tension_immob_dep = df_tension_immob_dep.join(parts_par_zone)
    
    return df_tension_immob_dep
    
    

def get_nb_diagnostiqueur_dep():
    # rq : la corse n'a pas de diagnostiqueurs !
    data = pd.read_csv(os.path.join('data','MTE','annuaire-diagnostiqueurs-immobiliers.csv'),sep=';')
    data = data[data['Type de certificat'].str.contains('DPE')]
    data['CP'] = [f"{e:05d}" for e in data.CP]
    data.drop_duplicates(subset = ["N° de certificat", "Organisme", "CP"], inplace = True)
    data = data[~data.CP.str.startswith('97')] # uniquement territoire hexagonal
    data = data[~data.CP.str.startswith('20')] # hors corse (2A et 2B aggrégé)
    data['dep_code'] = [Departement(e[:2]).code for e in data.CP]
    
    data_count = data.groupby('dep_code')[['Organisme']].count()
    data_count = data_count.rename(columns={'Organisme':'nb_diagnostiqueurs_dep'})
    return data_count
    

def get_nb_rp_loc():
    # Nombre de résidences principales occupées par locataires
    
    df_nb_rp_loc = pd.read_excel(os.path.join('data','INSEE','base-cc-logement-2022.xlsx'), usecols=['CODGEO', 'P22_RP_LOC'], skiprows=5)  # names = ['CODGEO', 'nb_rp_loc']
    df_nb_rp_loc.set_index('CODGEO', inplace=True)
    df_nb_rp_loc = df_nb_rp_loc[~df_nb_rp_loc.index.str.startswith('97')] # uniquement territoire hexagonal
    df_nb_rp_loc = df_nb_rp_loc[~df_nb_rp_loc.index.str.startswith('20')] # hors corse (2A et 2B aggrégé)
    df_nb_rp_loc['dep_code'] = [Departement(e[:2]).code for e in df_nb_rp_loc.index]
    df_nb_rp_loc = df_nb_rp_loc.groupby('dep_code')[['P22_RP_LOC']].sum()

    
    return df_nb_rp_loc
    
#%% ===========================================================================
# script principal
# =============================================================================



def main():
    
    tic = time.time()
    
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')
    output_folder = os.path.join('output',today)
    os.makedirs(output_folder, exist_ok=True)
    
    if True:
       
        pd.options.display.max_columns = None
        df_tension_immob_dep = tension_immob_dep()
        # print(df_tension_immob_dep)
        
# =============================================================================
#         # to do : Visualisation des parts par département
#         zone = 'A'
#         france = France()
#         dict_dep_zone = {d:0. for d in france.departements} 
#         for dep in france.departements :
#             dep_code = dep.code
#             print(dep)
#             dict_dep_zone[dep] = df_tension_immob_dep[f'part_{zone}']
# =============================================================================
        

    if True :
        
        # Fixation des paramètres de mesure du bunching
        old_built_filter = True

        # method='diff_simple'
        # method = 'diff_beta_cente_abs'
        method='diff_moyenne'
        
        itv_bunching=10
        window_size=50
            
        seuils = ['D/E', 'E/F', 'F/G']
        
        france_bunching = calcul_bunching_france(output_folder, method, itv_bunching, window_size, old_built_filter, max_xlim = 600, verbose=False, force=True)
        france_bunching_cut, seuils_sans_slash = cut_france_bunching(france_bunching, seuils)
        france_bunching[f'Somme_seuils_{seuils_sans_slash}_method_{method}'] = france_bunching_cut.sum(axis=1) # on somme sur les lignes des colonnes conservées    
        
        bunching = france_bunching[[f'Somme_seuils_{seuils_sans_slash}_method_{method}']] #todo : ne sert a rien de rajouter ligne a france_binching ? on pourrait direct dire bunching = = france_bunching_cut.sum(axis=1) ?
        
        # Vecteur des paramètres
        variables = df_tension_immob_dep # todo : dire direct variables=tension_immob_dep() ?
        variables = sm.add_constant(variables)
        
        # variables_list = ['part_A','part_Abis','part_B1','part_B2','part_C']
        # variables_list = ['part_A','part_C','zcl_Tref','total_logements','nb_diagnostiqueurs_dep']
        variables_list = ['part_C', 'zcl_H3','total_logements', 'P22_RP_LOC']
        variables_list = ['gain_moyen_etiquette','part_dpe_stables']
        
        
        bunching = bunching.join(variables)
        # bunching = bunching.join(get_nb_diagnostiqueur_dep())
        bunching = bunching.join(get_nb_rp_loc())
        
        # ajout de la zone climatique 
        bunching["zcl"] = [Departement(e).climat for e in bunching.index]
        bunching["zcl_Tref"] = [Departement(e).climat_Tref for e in bunching.index]
        bunching['zcl_H3'] = (bunching.zcl == 'H3').map(int)
        
        # ajout du log du nombre de logements
        bunching['log_total_logements'] = np.log(bunching.total_logements)
        
        # ajout du gain moyen d'étiquette sur une période # todo : l'enlever, rien a faire dans econometrie
        period = 20
        dict_part_dpe_stables, dict_gain_moyen_etiquette, dict_gain_moyen_etiquette_parmi_modif = dicts_dep_gain_moyen_etiquette(period)
        bunching['part_dpe_stables'] = dict_part_dpe_stables.values()
        bunching['gain_moyen_etiquette'] = dict_gain_moyen_etiquette.values() # todo : comment etre sure que les departements sont bien alignés ?
        
        
        bunching_test = bunching[[f'Somme_seuils_{seuils_sans_slash}_method_{method}','const']+variables_list].dropna() # on ne garde que les variables (=colonnes) qui nous intéressent
        
        model = sm.OLS(bunching_test[f'Somme_seuils_{seuils_sans_slash}_method_{method}'], bunching_test[['const']+variables_list])
        results = model.fit()
        results.params
        print(results.summary())
        
        
        p = sns.pairplot(data= bunching_test[[f'Somme_seuils_{seuils_sans_slash}_method_{method}']+variables_list])
        p.fig.suptitle(f"Corrélation entre le bunching méthode {method} et d'autres variables, old_built_filter = {old_built_filter}")
        p.fig.subplots_adjust(top=0.96)
        
        if True :
            if old_built_filter:
                save_path = os.path.join(output_folder,f'Pairplot_correlation_bunching_methode_{method}_{variables_list}_old_built.png') 
            else:
                save_path = os.path.join(output_folder,f'Pairplot_correlation_bunching_methode_{method}_{variables_list}.png') 
            plt.savefig(save_path, bbox_inches='tight')


    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')
    
    

if __name__ == '__main__':
    main()
