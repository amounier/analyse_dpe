#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 23:32:23 2026

@author: amounier
"""

import time
import requests
import io
import os
import json
import geopandas as gpd
import numpy as np
import pandas as pd
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

from administrative import Departement, France


def download_bdnb(dep_code, force=False):
    """
    Téléchargement automatique d'un fichier départemental de la BDNB

    Parameters
    ----------
    dep_code : str
        code du département (conforme avec la nomenclature de la BDNB).
    force : boolean, optional
        Force le retéléchargement même si présent sur disque. The default is False.

    Returns
    -------
    None.

    """
    # Définition du chemin de sauvegarde
    save_path = os.path.join('data','BDNB')
    os.makedirs(save_path, exist_ok=True)
    existing_files = os.listdir(save_path)

    dep = dep_code.lower() # pour aller avec l'url ou la Corse est en minuscule 2a 2b
    
    # Définition du nom du fichier final
    file = f'open_data_millesime_2025-07-a_dep{dep}_gpkg'
    
    # Pour télécharger la version 2026 :
    # file = f'open_data_millesime_2026-02-a_dep{dep}_gpkg'

    file_path = os.path.join(save_path,file)
    
    # Téléchargement seulement si nécessaire
    if file not in existing_files or force:
        # url de téléchargement
        url = f'https://open-data.s3.fr-par.scw.cloud/bdnb_millesime_2025-07-a/millesime_2025-07-a_dep{dep}/open_data_millesime_2025-07-a_dep{dep}_gpkg.zip'
        
        # Pour télécharger la version 2026 :
        # url = f'https://open-data.s3.fr-par.scw.cloud/bdnb_millesime_2026-02-a/millesime_2026-02-a_dep{dep}/open_data_millesime_2026-02-a_dep{dep}_gpkg.zip'
        
        subprocess.run(f"wget -P {save_path} {url}",shell=True)
        subprocess.run(f"unzip {file_path+'.zip'} -d {file_path}",shell=True)
        
        print('\n zip files deleted')
        subprocess.run(f"rm {os.path.join(save_path,'*.zip')}",shell=True)
        return 
    return


def get_bdnb(dep_code='75', chunksize=5e4):
    """
    Ouvre de manière non compilée les données de la BDNB d'un département, selon 3 tables:
        - dpe_logement
        - rel_batiment_groupe_dpe_logement
        - batiment_groupe_compile

    Parameters
    ----------
    dep : str, optional
        code du département. The default is '75'.
    chunksize : float, optional
        taille des chunk dask. The default is 5e4.

    Returns
    -------
    bdnb_dpe_logement : pandas DataFrame
        Base des DPE.
    bdnb_rel_batiment_groupe_dpe_logement : pandas DataFrame
        lien entre les bases.
    bdnb_batiment_groupe_compile : pandas DataFrame
        base des batiments (groupe).

    """
    def fix_invalid_seconds(val):
        if isinstance(val, str):
            val = re.sub(r'T(\d{2}):(\d{2}):60', r'T\1:\2:59', val)
            return pd.Timestamp(val)
        return val
    
    dep = dep_code.lower()
    file = os.path.join('data','BDNB',f'open_data_millesime_2025-07-a_dep{dep}_gpkg','gpkg','bdnb.gpkg')
    
    # file = os.path.join('data','BDNB',f'open_data_millesime_2026-02-a_dep{dep}_gpkg','gpkg','bdnb.gpkg')

    
    try:
        bdnb_dpe_logement = dask_geopandas.read_file(file, chunksize=chunksize, layer='dpe_logement')
        bdnb_rel_batiment_groupe_dpe_logement = dask_geopandas.read_file(file, chunksize=chunksize, layer='rel_batiment_groupe_dpe_logement')
        bdnb_batiment_groupe_compile = dask_geopandas.read_file(file, chunksize=chunksize, layer='batiment_groupe_compile')
        
        bdnb_dpe_logement['date_reception_dpe'] = bdnb_dpe_logement['date_reception_dpe'].apply(fix_invalid_seconds,meta=('date_reception_dpe', 'datetime64[ns]'))
        bdnb_dpe_logement['date_etablissement_dpe'] = bdnb_dpe_logement['date_etablissement_dpe'].apply(fix_invalid_seconds,meta=('date_etablissement_dpe', 'datetime64[ns]'))
        return bdnb_dpe_logement, bdnb_rel_batiment_groupe_dpe_logement, bdnb_batiment_groupe_compile
    
    except DataSourceError:
        departement = Departement(dep)
        print(f'\n {departement} indisponible, téléchargement des données...')
        download_bdnb(dep_code=dep)
        return get_bdnb(dep_code=dep)


def draw_local_map(geometry,batiment_groupe_id,style='map',figsize=12, radius=370, grey_background=True, save_path=None, include_OSM_copyright=True):
    """
    based on https://www.theurbanist.com.au/2021/03/plotting-openstreetmap-images-with-cartopy/

    """
    
    def image_spoof(self, tile):
        """Reformat for cartopy"""
        url = self._image_url(tile)                
        req = Request(url)                         
        req.add_header('User-agent','Anaconda 3')  
        fh = urlopen(req) 
        im_data = io.BytesIO(fh.read())            
        fh.close()                                 
        img = Image.open(im_data)  
        if grey_background:
            img = img.convert("L")             
        img = img.convert(self.desired_tile_form)  
        return img, self.tileextent(tile), 'lower' 
    
    # reformat web request for street map spoofing
    cimgt.OSM.get_image = image_spoof 
    img = cimgt.OSM()
    
    fig = plt.figure(figsize=(figsize,figsize)) 
    
    # project using coordinate reference system (CRS) of street map
    ax = plt.axes(projection=img.crs) 
    data_crs = ccrs.PlateCarree()
    
    # compute OSM scale
    scale = int(100/np.log(radius))
    scale = (scale<20) and scale or 19
    
    # compute extent of map
    lon,lat = geometry.centroid.x, geometry.centroid.y
    dist = radius*1.1
    dist_cnr = np.sqrt(2*dist**2)
    top_left = cgeo.Geodesic().direct(points=(lon,lat),azimuths=-45,distances=dist_cnr)[:,0:2][0]
    bot_right = cgeo.Geodesic().direct(points=(lon,lat),azimuths=135,distances=dist_cnr)[:,0:2][0]
    extent = [float(f) for f in [top_left[0], bot_right[0], bot_right[1], top_left[1]]]
    ax.set_extent(extent, crs=ccrs.PlateCarree()) 
    
    # add OSM with zoom specification
    ax.add_image(img, int(scale)) 
    
    # add building on map
    ax.add_geometries(geometry, crs=data_crs, color='tab:blue')
    ax.set_title(batiment_groupe_id,fontdict={'fontsize':15})
    
    # add OSM copyright
    if include_OSM_copyright:
        ax.text(0.5, -0.035, '\xa9 OpenStreetMap contributors', fontsize='x-large', horizontalalignment='center',verticalalignment='bottom', transform=ax.transAxes)
        
    # sauvegarde de l'image
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        
    return fig,ax


def neighbourhood_map(batiment_groupe_id, path, save=True):
    """
    carte des alentours d'un bâtiment de la BDNB

    Parameters
    ----------
    batiment_groupe_id : TYPE
        DESCRIPTION.
    save : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    # requête à la BDNB
    r = requests.get('https://api.bdnb.io/v1/bdnb/donnees/batiment_groupe_complet/adresse',
                     params={'batiment_groupe_id': 'eq.'+batiment_groupe_id,#},
                             'select': 'batiment_groupe_id, s_geom_groupe, contient_fictive_geom_groupe, geom_groupe'},
                     headers = {"Accept": "application/geo+json"},)

    # lecture des données d'API
    gdf = gpd.read_file(io.StringIO(r.text))
    gdf = gdf[['geometry']]
    gdf = gdf.set_crs(epsg=2154, allow_override=True)
    gdf = gdf[gdf.columns[~gdf.isnull().all()]]
    
    # reprojection en longitude latitude
    gdf = gdf.to_crs(epsg=4326) 
    
    # sauvegarde de la carte
    if save:
        save_path = os.path.join(path,'{}_map.png'.format(batiment_groupe_id))
    else:
        save_path = None
    fig,ax = draw_local_map(gdf.iloc[0].geometry, save_path=save_path, batiment_groupe_id=batiment_groupe_id)
    plt.show()
    plt.close()
    return


def get_batiment_groupe_infos(batiment_groupe_id,variables=None):
    """
    requete à l'API de la BDNB pour récupérer les informations d'un bâtiment
    (par l'usage de l'identifiant batiment_groupe_id)

    Parameters
    ----------
    batiment_groupe_id : str
        DESCRIPTION.
    variables : list, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """
    # requête à la BDNB
    r = requests.get('https://api.bdnb.io/v1/bdnb/donnees/batiment_groupe_complet/adresse',
                     params={'batiment_groupe_id': 'eq.'+batiment_groupe_id},
                     headers = {"Accept": "application/geo+json"},)

    data = json.loads(r.text)
    
    if isinstance(variables, list):
        res = dict()
        for key in variables:
            res[key] = data.get('features')[0].get('properties').get(key)
    elif isinstance(variables, str):
        res = dict()
        res[variables] = data.get('features')[0].get('properties').get(variables)
    else:
        res = data
    return res



#%% ===========================================================================
# script principal
# =============================================================================

def main():
    tic = time.time()
    
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')
    output_folder = os.path.join('output',today)
    os.makedirs(output_folder, exist_ok=True)
    
    #%% Téléchargement des données BDNB d'un département
    if False:
        dep = Departement(91)
        print(dep)
        
        download_bdnb(dep.code)
        
    #%% Téléchargement des données BDNB de tous les départements (France hexagonale)    
    if False:
        france = France()
        for dep in france.departements :
            print(dep)
            download_bdnb(dep.code)

    
    #%% Test d'une carte locale
    if True:
        #bat_groupe_id = 'bdnb-bg-FHEF-WAAZ-S5XC' # suspicion manipulation dpe (Paris)
        # bat_groupe_id = 'bdnb-bg-V8HA-KABB-ZCSX' # immeuble
        # bat_groupe_id = 'bdnb-bg-8M4H-6M5W-M3JE' # 14 rue Brillat Savarin
        # bat_groupe_id = 'bdnb-bg-BUZK-W1C9-14P3' # 704 logements
        # bat_groupe_id = 'bdnb-bg-129J-JTEH-Z4XN' # bat sans adresse
        # bat_groupe_id = 'bdnb-bg-37BE-BE89-5NR5'
        # bat_groupe_id = 'bdnb-bg-18GG-5GGC-X6NX'
        bat_groupe_id = 'bdnb-bg-RGSM-7GV4-4QBK'

        
        neighbourhood_map(bat_groupe_id, output_folder)
        infos = get_batiment_groupe_infos(bat_groupe_id)
        print(infos)
    
    tac = time.time()
    print(f'Done in {tac-tic:.2f}s.')
    
if __name__ == '__main__':
    main()