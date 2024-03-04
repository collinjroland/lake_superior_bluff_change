# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 20:00:20 2023

@author: quatlab

# Title: JABLTX Analaysis Lake Superior
# Author: Collin Roland
# Date Created: 20231230
# Summary: Goal is to systematically analyze volume change for all JABLTX pointclouds
# Date Last Modified: 20240224
# To do: cleanup and documentation, 
"""

# %% Import packages

import alphashape
import contextily as cx
import folium
import geopandas as gpd
import glob
import json
import latex
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
import pdal
import pathlib
from pathlib import Path
import pyproj
import rasterio as rio
import rasterio
from rasterio import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio import errors
from rasterstats import zonal_stats
import re
import rioxarray
import scipy.io
import shapely
import shapelysmooth
from shapely import Point, LineString, MultiLineString, MultiPoint, distance, intersects, buffer, prepare
from shapely.ops import split, snap
from whitebox_tools import WhiteboxTools

# plt.rcParams.update({
#     'figure.constrained_layout.use': True,
#     'font.size': 12,
#     'axes.edgecolor': 'black',
#     'xtick.color':    'black',
#     'ytick.color':    'black',
#     'axes.labelcolor':'black',
#     'axes.spines.right':True,
#     'axes.spines.top':  True,
#     'xtick.direction': 'in',
#     'ytick.direction': 'in',
#     'xtick.major.size': 6,
#     'xtick.minor.size': 4,
#     'ytick.major.size': 6,
#     'ytick.minor.size': 4,
#     'xtick.major.pad': 15,
#     'xtick.minor.pad': 15,
#     'ytick.major.pad': 15,
#     'ytick.minor.pad': 15,
#     })

wbt = WhiteboxTools()
wbt.set_whitebox_dir(Path(r'C:\Users\quatlab\Documents\WBT'))
%matplotlib qt5
# %% Set home directory

homestr = r'D:\CJR'
home = Path(r'D:\CJR')

# %% Self-defined functions

def read_file(file):
    """Read in a raster file
    
    Parameters
    -----
    file: (string) path to input file to read
    """
    return(rasterio.open(file))

def reproj_match(infile, match):
    """Reproject a file to match the shape and projection of existing raster. 
    Uses bilinear interpolation for resampling.
    
    Parameters
    ----------
    infile : (string) path to input file to reproject
    match : (string) path to raster with desired shape and projection 
    outfile : (string) path to output file tif
    """
    # open input
    with rasterio.open(infile) as src:
        src_transform = src.transform
        
        # open input to match
        with rasterio.open(match) as match:
            dst_crs = match.crs
            
            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,     # input CRS
                dst_crs,     # output CRS
                match.width,   # input width
                match.height,  # input height 
                *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
            )
            # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "compress":"LZW",
                           "dtype":"float32",
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": match.nodata})
        memfile = MemoryFile()
        # with MemoryFile() as memfile:
        with memfile.open(**dst_kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear)
        try:
            data = memfile.open()
            error_state = False
            return data, error_state
        except rasterio.errors.RasterioIOError as err:
            error_state = True
            data= []
            return data,error_state
            pass
            #with memfile.open() as dataset:  # Reopen as DatasetReader
                #return dataset
                
def read_paths(path,extension):
    """Read the paths of all files in a directory (including subdirectories)
    with a specified extension
    
    Parameters
    -----
    file: (string) path to input file to read
    extension: (string) file extension of interest
    """
    AllPaths = []
    FileNames =[]
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                FileNames.append(file)
                filepath = subdir+os.sep+file
                AllPaths.append(filepath)
    return(AllPaths,FileNames)

def get_cell_size(str_grid_path):
    with rasterio.open(str(str_grid_path)) as ds_grid:
        cs_x, cs_y = ds_grid.res
    return cs_x

def define_grid_projection(str_source_grid, dst_crs, dst_file):
    print('Defining grid projection:')
    with rasterio.open(str_source_grid, 'r') as src:
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
        })
        arr_src = src.read(1)
        with rasterio.open(dst_file, 'w', **kwargs) as dst:
            dst.write(arr_src, indexes=1)

def reproject_grid_layer(str_source_grid, dst_crs, dst_file, resolution, logger):
    # reproject raster plus resample if needed
    # Resolution is a pixel value as a tuple
    try:
        st = timer()
        with rasterio.open(str_source_grid) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=resolution)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            with rasterio.open(dst_file, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear)
        end = round((timer() - st)/60.0, 2)
        logger.info(f'Reprojected DEM. Time elapsed: {end} mins')
        return dst_file
    except:
        logger.critical(f'{str_source_grid}: failed to reproject.')
        sys.exit(1)

def reproject_vector_layer(in_shp, str_target_proj4, logger):
    print(f'Reprojecting vector layer: {in_shp}')
    proj_shp = in_shp.parent / f'{in_shp.stem}_proj.shp'
    if proj_shp.is_file():
        logger.info(f'{proj_shp} reprojected file already exists\n')
        return str(proj_shp)
    else:
        gdf = gpd.read_file(str(in_shp))
        # fix float64 to int64
        float64_2_int64 = ['NHDPlusID', 'Shape_Area', 'DSContArea', 'USContArea']
        for col in float64_2_int64:
            try:
                gdf[col] = gdf[col].astype(np.int64)
            except KeyError:
                pass
        gdf_proj = gdf.to_crs(str_target_proj4)
        gdf_proj.to_file(str(proj_shp))
        logger.info(f'{proj_shp} successfully reprojected\n')
        return str(proj_shp)

def clip_features_using_grid(
        str_lines_path, output_filename, str_dem_path, in_crs, logger, mask_shp):
    # clip features using HUC mask, if the mask doesn't exist polygonize DEM
    mask_shp = Path(mask_shp)
    if mask_shp.is_file():
        st = timer()
        # whitebox clip
        WBT.clip(str_lines_path, mask_shp, output_filename)
        end = round((timer() - st)/60.0, 2)
        logger.info(f'Streams clipped by {mask_shp}. Time Elapsed: {end} mins')
    else:
        st = timer()
        logger.warning(f'''
        {mask_shp} does not file exists. Creating new mask from DEM.
        This step can be error prone please review the output.
        ''')
        # Polygonize the raster DEM with rasterio:
        with rasterio.open(str(str_dem_path)) as ds_dem:
            arr_dem = ds_dem.read(1)
        arr_dem[arr_dem > 0] = 100
        mask = arr_dem == 100
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(arr_dem, mask=mask, transform=ds_dem.transform))
            )
        poly = list(results)
        poly_df = gpd.GeoDataFrame.from_features(poly)
        poly_df.crs = in_crs
        # poly_df = poly_df[poly_df.raster_val == 100.0]
        # tmp_shp = os.path.dirname(str_dem_path) + "/mask.shp"  # tmp huc mask
        poly_df.to_file(str(mask_shp))
        # whitebox clip
        WBT.clip(str_lines_path, str(mask_shp), output_filename)
        end = round((timer() - st)/60.0, 2)
        logger.info(f'Streams clipped by {mask_shp}. Time Elapsed: {end} mins')

def run_pdal(json_path,bounds,outfile):
    # 
    with open(json_path) as json_file:
        the_json = json.load(json_file)
    #the_json[0]['filename'] = filename_laz
    #the_json[-9]['groups'] = groups
    the_json[-4]['bounds'] = bounds
    the_json[-1]['filename'] = outfile
    pipeline = pdal.Pipeline(json.dumps(the_json))
    try:
        pipeline.execute()
    except RuntimeError as e:
        print(e)
        
def open_memory_tif(arr, meta):
    from rasterio.io import MemoryFile
    #     with rasterio.Env(GDAL_CACHEMAX=256, GDAL_NUM_THREADS='ALL_CPUS'):
    with MemoryFile() as memfile:
        with memfile.open(**meta) as dataset:
            dataset.write(arr, indexes=1)
        return memfile.open()
    
def raster_clip(infile, clip_geom, crs_in):
    #Debugging
    # infile= value
    # crs_in = clip_geom.crs
    # clip_geom = clip_geom
    
    infile=infile
    clip_geom = clip_geom.to_crs('EPSG:32615')
    clip_geom = clip_geom.reset_index()
    dem = read_file(infile)
    try:
        out_image, out_transform = mask(dem,[clip_geom.geometry[0]], nodata=dem.meta['nodata'],crop=True)
        error_state = False
        out_meta = dem.meta
        out_meta.update({"crs": dem.crs,
                           "compress":"LZW",
                           "dtype":"float32",
                           "transform": out_transform,
                           "width": out_image.shape[2],
                           "height": out_image.shape[1]})
        clip_kwargs = out_meta
        memfile_clip = MemoryFile()
        with memfile_clip.open(**clip_kwargs) as dst:
            dst.write(out_image)
        return memfile_clip,error_state
    except ValueError as err:
        memfile_clip=[]
        error_state=True
        return memfile_clip,error_state
        pass
    
def raster_clip_from_dataset(dem, clip_geom, crs_in):
    #Debugging
    # infile= value
    # crs_in = clip_geom.crs
    # clip_geom = clip_geom
    clip_geom = clip_geom.to_crs('EPSG:32615')
    clip_geom = clip_geom.reset_index()
    try:
        out_image, out_transform = mask(dem,[clip_geom.geometry[0]], nodata=dem.meta['nodata'],crop=True)
        error_state = False
        out_meta = dem.meta
        out_meta.update({"crs": dem.crs,
                           "compress":"LZW",
                           "dtype":"float32",
                           "transform": out_transform,
                           "width": out_image.shape[2],
                           "height": out_image.shape[1]})
        clip_kwargs = out_meta
        memfile_clip = MemoryFile()
        with memfile_clip.open(**clip_kwargs) as dst:
            dst.write(out_image)
        return memfile_clip,error_state
    except ValueError as err:
        memfile_clip=[]
        error_state=True
        return memfile_clip,error_state
        pass
    
def compute_volume_change_stats(array, dod_tif, lod):
    """Compute volume change summary stats for a clipped DEM of difference.
    Parameters
    -----
    array: numpy array of DEM of difference
    dod_tif: rasterio read of DOD tif
    lod: float number representing change detection limit
    """
    # Debugging
    # array = dod_clip
    # dod_tif = dod_merge
    # lod = 0.20
    colnames = ['lod', 'area', 'area_neg', 'area_pos', 'frac_neg', 'frac_pos',
                'neg_sum', 'neg_sum_high', 'neg_sum_low', 'pos_sum', 'pos_sum_high',
                'pos_sum_low', 'net_sum', 'net_sum_high', 'net_sum_low', 'ns_md2',
                'ns_l_md2', 'ns_h_md2', 'min', 'max', 'median', 'mean', 'stddev']
    stats_df = pd.DataFrame(columns=colnames)
    stats_df = pd.DataFrame(columns=colnames)
    cell_size = dod_tif.meta['transform'][0] * dod_tif.meta['transform'][0]
    lod_condition_low = ((array < (-1. * lod)) & (array > -100.))
    lod_condition_high = ((array > (lod)) & (array < 100.))
    lod_condition_combined = np.logical_or(lod_condition_low, lod_condition_high)
    array_filt = array[lod_condition_combined]
    array_filt_neg = array[lod_condition_low]
    array_filt_pos = array[lod_condition_high]
    array_filt_neg_mask = np.ma.masked_where(lod_condition_low == 0, array)
    array_filt_pos_mask = np.ma.masked_where(lod_condition_high == 0, array)
    array_filt_mask = np.ma.masked_where(lod_condition_combined == 0, array)
    array_filt_mask_fill = array_filt_mask.filled(fill_value=dod_tif.meta['nodata'])
    stats_df.at[0, 'lod'] = lod
    stats_df.at[0, 'area'] = ((np.count_nonzero(array > -100.)) * cell_size)
    stats_df.at[0, 'area_neg'] = ((np.count_nonzero(array_filt_neg)) * cell_size)
    stats_df.at[0, 'area_pos'] = ((np.count_nonzero(array_filt_pos)) * cell_size)
    stats_df.at[0, 'frac_neg'] = ((stats_df['area_neg'] / stats_df['area'])[0])
    stats_df.at[0, 'frac_pos'] = ((stats_df['area_pos'] / stats_df['area'])[0])
    stats_df.at[0, 'neg_sum'] = ((np.ma.sum(array_filt_neg_mask)).astype(float) * cell_size)
    stats_df.at[0, 'neg_sum_low'] = ((np.ma.sum(array_filt_neg_mask + lod)).astype(float) * cell_size)
    stats_df.at[0, 'neg_sum_high'] = ((np.ma.sum(array_filt_neg_mask - lod)).astype(float) * cell_size)
    stats_df.at[0, 'pos_sum'] = ((np.ma.sum(array_filt_pos_mask)).astype(float) * cell_size)
    stats_df.at[0, 'pos_sum_high'] = ((np.ma.sum(array_filt_pos_mask + lod)).astype(float) * cell_size)
    stats_df.at[0, 'pos_sum_low'] = ((np.ma.sum(array_filt_pos_mask - lod)).astype(float) * cell_size)
    stats_df.at[0, 'net_sum'] = ((np.ma.sum(array_filt_mask)).astype(float) * cell_size)
    stats_df.at[0, 'net_sum_high'] = ((np.ma.sum(array_filt_mask + lod)).astype(float) * cell_size)
    stats_df.at[0, 'net_sum_low'] = ((np.ma.sum(array_filt_mask - lod)).astype(float) * cell_size)
    for count, value in enumerate(stats_df.values[0]):
        if np.ma.is_masked(value):
            stats_df.values[0][count] = stats_df.values[0][count].data.item()
    stats_df.at[0, 'ns_md2'] = (stats_df['pos_sum'] + stats_df['neg_sum']).values[0]
    stats_df.at[0, 'ns_l_md2'] = np.min([np.abs((stats_df['neg_sum_low'] + stats_df['pos_sum_high']).values[0]),
                                        np.abs((stats_df['neg_sum_high'] + stats_df['pos_sum_low']).values[0])])
    stats_df.at[0, 'ns_h_md2'] = np.max([np.abs((stats_df['neg_sum_low'] + stats_df['pos_sum_high']).values[0]),
                                        np.abs((stats_df['neg_sum_high'] + stats_df['pos_sum_low']).values[0])])
    stats_df.at[0, 'min'] = np.ma.min(array_filt_mask)
    stats_df.at[0, 'max'] = np.ma.max(array_filt_mask)
    stats_df.at[0, 'median'] = np.ma.median(array_filt_mask)
    stats_df.at[0, 'mean'] = np.ma.mean(array_filt_mask)
    stats_df.at[0, 'stddev'] = np.ma.std(array_filt_mask)
    for count, value in enumerate(stats_df.values[0]):
        if np.ma.is_masked(value):
            stats_df.values[0][count] = stats_df.values[0][count].data.item()
    return(stats_df)

def point_transect_distance(transect, points, raster):
    points_filt = points[points.geometry.intersects(transect)]
    points_filt = points_filt.reset_index()
    if len(points_filt) > 0:
        points_dist = transect.project(points_filt.geometry).values[0]
        points_elev = [sample[0] for sample in raster.sample(points_filt.geometry[0].coords)][0]
    else:
        points_dist = np.nan
        points_elev = np.nan
    return [points_dist, points_elev]
# %% Self-defined functions to generate transects

def densify_geometry(line_geometry, step, crs=None):
    # crs: epsg code of a coordinate reference system you want your line to be georeferenced with
    # step: add a vertice every step in whatever unit your coordinate reference system use.
    length_m=line_geometry.length # get the length
    xy=[] # to store new tuples of coordinates
    for distance_along_old_line in np.arange(0,int(length_m),step): 
        point = line_geometry.interpolate(distance_along_old_line) # interpolate a point every step along the old line
        xp,yp = point.x, point.y # extract the coordinates
        xy.append((xp,yp)) # and store them in xy list
    new_line=LineString(xy) # Here, we finally create a new line with densified points.  
    if crs != None:  #  If you want to georeference your new geometry, uses crs to do the job.
        new_line_geo=gpd.geoseries.GeoSeries(new_line,crs=crs) 
        return new_line_geo
    else:
        return new_line
    
def gen_xsec(point, angle, poslength, neglength, step, merged_dem, crs=None):
     '''
     point - Tuple (x, y)
     angle - Angle you want your end point at in degrees.
     length - Length of the line you want to plot.
    
     Will plot the line on a 10 x 10 plot.
     '''
     #
     step = step
     crs=crs
     xsec_angle = angle
     # unpack the first point
     x = point.xy[0][0]
     y = point.xy[1][0]
     
     # find the end point
     endy = y + poslength * math.cos(math.radians(xsec_angle))
     endx = x + poslength * math.sin(math.radians(xsec_angle))
     end_point = Point(endx, endy) 
     
     # find the start point
     starty = y + neglength * math.cos(math.radians(xsec_angle+180))
     startx = x + neglength * math.sin(math.radians(xsec_angle+180))
     start_point = Point(startx,starty)
     
     # Figure out which direction is landward
     start_elev = [sample[0] for sample in merged_dem.sample(start_point.coords)]
     # print("Start elev=",start_elev)
     if start_elev == np.nan:
         start_elev = -9999.0
     end_elev = [sample[0] for sample in merged_dem.sample(end_point.coords)]
     # print("End elev=",end_elev)
     if start_elev>=end_elev:
         xsec_angle_mod = xsec_angle+180
         
         # find the end point
         endy = y + poslength * math.cos(math.radians(xsec_angle_mod))
         endx = x + poslength * math.sin(math.radians(xsec_angle_mod))
         end_point = Point(endx, endy) 
         
         # find the start point
         starty = y + neglength * math.cos(math.radians(xsec_angle_mod+180))
         startx = x + neglength * math.sin(math.radians(xsec_angle_mod+180))
         start_point = Point(startx,starty)
     
     # generate a line from points
     xsec_line = LineString([start_point, end_point])
     
     # densify line to specified resolution
     xsec_line = densify_geometry(xsec_line, step, crs=crs)
     return xsec_line, start_point, end_point

def gen_xsec_wrap(shoreline_clip, outdir, tile_name, poslength, neglength, xsec_spacing, simp_tolerance, step, merged_dem, crs):
     '''
     line - geoseries with a linestring geometry
     point - Tuple (x, y)
     angle - Angle you want your end point at in degrees.
     length - Length of the line you want to plot.
    
     Will plot the line on a 10 x 10 plot.
     '''
     # Debugging
        # xsec_spacing=5
        # tile_name = os.path.splitext(tile2009_sing.Name)[0]
        # shoreline_sing = shoreline_sing
        # poslength = 150.0
        # neglength = 50.0
        # spacing = xsec_spacing
        # simp_tolerance = 20 
        # shoreline_clip = shoreline_clip
     # Smooth baseline
     for count_1 in range(0,len(shoreline_clip)):
         shoreline_sing_1 = gpd.GeoSeries(data=shoreline_clip.iloc[count_1], crs=crs)
         if shoreline_sing_1.geometry.geom_type[0]=='MultiLineString':
             shoreline_sing_2 = shoreline_sing_1.geometry.explode(index_parts=True)
         else:
             shoreline_sing_2 = shoreline_sing_1
         for count_4 in range(0,len(shoreline_sing_2)):
             shoreline_sing_3 = gpd.GeoSeries(data=shoreline_sing_2.iloc[count_4], crs=crs)
             if (shoreline_sing_3.geometry.length[0]>=(2*spacing)):
                 simplify_baseline = shoreline_sing_3.simplify(simp_tolerance)
                 simplify_baseline = simplify_baseline.reset_index()
                 smooth_baseline = shapelysmooth.catmull_rom_smooth(simplify_baseline.geometry[0],alpha=0.9)
                 # fig,ax = plt.subplots(1,1)
                 # shoreline_sing.plot(ax=ax)
                 # plt.plot(smooth_baseline.coords.xy[0],smooth_baseline.coords.xy[1],color='red')  
                 # Create cross-section points
                 num_points = []
                 num_points = int(smooth_baseline.length/spacing)
                 lenSpace = np.linspace(spacing,smooth_baseline.length,num_points)
                 tempPointList_xsec = []
                 tempLineList_xsec = []
                 tempStartPoints_xsec = []
                 tempEndPoints_xsec = []
                 for space in lenSpace:
                     tempPoint_xsec = (smooth_baseline.interpolate(space))#.tolist()[0]
                     tempPointList_xsec.append(tempPoint_xsec) 
                 # Calculate bearing at cross-section points
                 transformer = pyproj.Transformer.from_crs(6344,6318)
                 geodesic = pyproj.Geod(ellps='WGS84')
                 temp_angle = []
                 for count_2,value in enumerate(tempPointList_xsec):
                     if count_2>0:
                         lat1,long1 = transformer.transform(tempPointList_xsec[count_2-1].x,tempPointList_xsec[count_2-1].y)
                         lat2,long2 = transformer.transform(tempPointList_xsec[count_2].x,tempPointList_xsec[count_2].y)
                         fwd_azimuth,back_azimuth,distance = geodesic.inv(long1,lat1,long2,lat2)
                         orth_angle = back_azimuth-90
                         temp_angle.append(orth_angle)
                     if count_2==0:
                         count_2=1
                         lat1,long1 = transformer.transform(tempPointList_xsec[count_2-1].x,tempPointList_xsec[count_2-1].y)
                         lat2,long2 = transformer.transform(tempPointList_xsec[count_2].x,tempPointList_xsec[count_2].y)
                         fwd_azimuth,back_azimuth,distance = geodesic.inv(long1,lat1,long2,lat2)
                         orth_angle = back_azimuth-90
                         temp_angle.append(orth_angle)
                 # Generate cross-section lines
                 xsec_lines = []
                 xsec_starts = []
                 xsec_ends = []
                 for count_3,value in enumerate(temp_angle):
                     xsec_line,start_point,end_point = gen_xsec(tempPointList_xsec[count_3],value,poslength,neglength, step, merged_dem, crs=crs)
                     xsec_lines.append(xsec_line.iloc[0])
                     xsec_starts.append(start_point)
                     xsec_ends.append(end_point)  
                 # Convert to GDF
                 xsec_lines = gpd.GeoSeries(xsec_lines,crs=crs)
                 xsec_starts = gpd.GeoSeries(xsec_starts,crs=crs)
                 xsec_ends = gpd.GeoSeries(xsec_ends,crs=crs)
                 xsec_shoreline = gpd.GeoSeries(smooth_baseline,crs=crs)   
                 # Write outputs
                 tmp_start = os.path.join(outdir,'StartPoints',(tile_name+'_start_points_'+str(count_1)+'_'+str(count_4)+'.shp'))
                 tmp_end = os.path.join(outdir,'EndPoints',(tile_name+'_end_points_'+str(count_1)+'_'+str(count_4)+'.shp'))
                 tmp_lines = os.path.join(outdir,'Transects',(tile_name+'_transects_'+str(count_1)+'_'+str(count_4)+'.shp'))
                 tmp_shore = os.path.join(outdir,'Shorelines',(tile_name+'_shoreline_'+str(count_1)+'_'+str(count_4)+'.shp'))
                 xsec_lines.to_file(tmp_lines)
                 xsec_starts.to_file(tmp_start)
                 xsec_ends.to_file(tmp_end)
                 xsec_shoreline.to_file(tmp_shore)
                 plt.close()


def fix_index(gdf):
    gdf["row_id"] = gdf.index
    gdf.reset_index(drop=True, inplace=True)
    gdf.set_index("row_id", inplace=True)
    return (gdf)


def linestring_to_points(feature, line):
    return {feature: line.coords}


def make_xsec_points(transects, merged_dem):
    # Debugging
    transects = transects
    all_points = pd.DataFrame()
    for count,transect in enumerate(transects.geometry):
        points = [Point(coord[0], coord[1]) for coord in transect.coords]
        x = [coord[0] for coord in transect.coords]
        y = [coord[1] for coord in transect.coords]
        ID = [count]*len(points)
        near_dist = [distance(points[0],point) for point in points]
        elevation = [x[0] for x in merged_dem.sample(transect.coords)]
        
        #elevation = [np.nan if i<100 else i for i in elevation]
        temp_df = pd.DataFrame({'ID_1':ID,'RASTERVALU':elevation,'NEAR_DIST':near_dist,'Easting':x,'Northing':y})
        all_points= pd.concat([all_points,temp_df])
    all_points = all_points.reset_index(drop=True)
    all_points['FID']= all_points.index
    all_points = all_points[['FID','ID_1','RASTERVALU','NEAR_DIST','Easting','Northing']]
    return(all_points)
 
       
def temp_merge(dem_files):
    mem = MemoryFile()
    merge(dem_files, dst_path=mem.name)
    merged_dem = rasterio.open(mem)
    return(merged_dem)
# %% Read tile indexes and other filenames/paths

os.chdir(home)
tiles2009 = gpd.read_file(r'.\Bayfield2009\wi2009_bayfieldcounty_lakesuperior_index.shp')
tiles2009 = tiles2009.to_crs('EPSG:32615')
tiles2019 = gpd.read_file(r'.\JABLTX_2019\usace2019_superior_index.shp')
tiles2019 = tiles2019.to_crs('EPSG:32615')

laz2009_root = r'D:\CJR\Bayfield2009\LAZ'
dem2009_root = r'D:\CJR\Bayfield2009\DEM'
dem2009_root_buffer = r'D:\CJR\Bayfield2009\DEM\buffer'
laz2019_root = r'D:\CJR\JABLTX_2019\LAZ'
dem2019_root = r'D:\CJR\JABLTX_2019\DEM'
dem2019_root_buffer = r'D:\CJR\JABLTX_2019\DEM\buffer'
json_pipeline = homestr+r'\PDAL\wi_lake_superior_pipeline_delaunay.json'
json_pipeline_mod = homestr+r'\PDAL\wi_lake_superior_pipeline_delaunay_mod.json'
dem_fill_2009_root = r'D:\CJR\Bayfield2009\DEMfill'
dem_fill_2019_root = r'D:\CJR\JABLTX_2019\DEMfill'

# Write tiles to folium map for viewing
# os.chdir(homestr+'\\Folium')
# tile_map = tiles2009.explore(name="2009 tiles")
# tile_map = tiles2019.explore(m=tile_map, color="red",name="2019 tiles")
# folium.LayerControl().add_to(tile_map)
# tile_map.save("tilesSuperior.html")


# %% Rasterize
def rasterize_points(count,tiles2009, laz2009_root, tiles2019, laz2019_root, dem2009_root_buffer, dem2019_root_buffer, dem_fill_2009_root, dem_fill_2019_root, json_pipeline, json_pipeline_mod):
    # Debugging
    #count = 417
    
    # Select a 2009 tile, find adjacent 2009 tiles
    tile2009_sing = tiles2009.iloc[count] # pull out a single 2009 tile
    tiles_2009_adj = tiles2009[shapely.intersects(buffer(tile2009_sing.geometry, 50), tiles2009.geometry).values] # buffer tile by 50 and find intersecting tiles
    merge_tile_2009_paths = [os.path.join(laz2009_root, i) for i in tiles_2009_adj['Name']]
    
    # calculate bounds
    bounds = buffer(tile2009_sing.geometry, 50).bounds
    bounds_json = str("("+"["+str(bounds[0])+","+str(+bounds[2])+"]"+","+"["+str(bounds[1])+","+str(bounds[3])+"]"+")")
    
    # identify adjacent 2019 tiles using the buffered 2009 tile
    tile2019_intersect = tiles2019[shapely.intersects(buffer(tile2009_sing.geometry, 50), tiles2019.geometry).values]
    tile2019_intersect = tile2019_intersect.reset_index()
    tile2019_intersect = tile2019_intersect.drop(['index','Index'],axis=1)
    merge_tile_2019_paths = [os.path.join(laz2019_root,i[3:]) for i in tile2019_intersect['Name']]
    
    # Process LAZ files
    if len(tile2019_intersect)>0:
        # Process 2009 LAZ
        filename_laz = os.path.join(laz2009_root,tile2009_sing['Name'])
        filename_tif = os.path.join(dem2009_root_buffer,os.path.splitext(os.path.basename(filename_laz))[0]+'.tif')
        
        # Writing merged JSON 2009
        with open(json_pipeline) as json_file:
            the_json = json.load(json_file)
        with open(json_pipeline_mod,'w',encoding='utf_8') as f:
            f.write("[\n")
            for i,val in enumerate(merge_tile_2009_paths):
                basename = os.path.splitext(val)[0][:]
                basename = re.escape(str(basename))
                string=str(basename+'.laz')
                #string = re.escape(string)
                if (i<(len(merge_tile_2009_paths)-1)):
                    f.write('\t'+"\""+string+"\""+",\n")
                if (i==(len(merge_tile_2009_paths)-1)):
                    f.write('\t'+"\""+string+"\",")
            for i,val1 in enumerate(the_json):
                part = val1
                json_str = str(part)
                json_str = json_str.replace('{','')
                json_str = json_str.replace(' ','')
                json_str = json_str.replace('}','')
                json_str = json_str.replace('\'','\"')
                # i is counter for json, j is counter for json part
                # Case for non-ending JSON block with a single line
                if (i<(len(the_json)-1)) & (len(part)<2):
                    json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t},')
                    f.write(json_str)
                # Case for non-ending JSON block with multiple lines
                if (i<(len(the_json)-1)) & (len(part)>1):
                    json_str = json_str.replace(',',',\n\t\t')
                    json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t},')
                    f.write(json_str)
                # Case for ending JSON block with a single line
                if (i==(len(the_json)-1)) & (len(part)<2):
                    json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t}\n]')
                    f.write(json_str)
                # Case for ending JSON block with multiple lines
                if (i==(len(the_json)-1)) & (len(part)>1):
                    json_str = json_str.replace(',',',\n\t\t')
                    json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t}\n]')
                    f.write(json_str)
        with open(json_pipeline_mod) as json_file:
            the_json = json.load(json_file) 
        run_pdal(json_pipeline_mod,bounds_json,filename_tif)

        basename = os.path.splitext(os.path.basename(filename_laz))[0]+'.tif'
        basename = basename[8:]
        basename = str(os.path.basename(merge_tile_2019_paths[0])[0:8]+basename)
        filename_tif = os.path.join(dem2019_root_buffer,basename)  

        # Writing merged JSON 2019
        with open(json_pipeline) as json_file:
            the_json = json.load(json_file)
            the_json[-2]["max_triangle_edge_length"]=500
        with open(json_pipeline_mod,'w',encoding='utf_8') as f:
            f.write("[\n")
            for i,val in enumerate(merge_tile_2019_paths):
                basename = os.path.splitext(val)[0][:]
                basename = re.escape(str(basename))
                string=str(basename+'.laz')
                #string = re.escape(string)
                if (i<(len(merge_tile_2019_paths)-1)):
                    f.write('\t'+"\""+string+"\""+",\n")
                if (i==(len(merge_tile_2019_paths)-1)):
                    f.write('\t'+"\""+string+"\",")
            for i,val1 in enumerate(the_json):
                part = val1
                json_str = str(part)
                json_str = json_str.replace('{','')
                json_str = json_str.replace(' ','')
                json_str = json_str.replace('}','')
                json_str = json_str.replace('\'','\"')
                # i is counter for json, j is counter for json part
                # Case for non-ending JSON block with a single line
                if (i<(len(the_json)-1)) & (len(part)<2):
                    json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t},')
                    f.write(json_str)
                # Case for non-ending JSON block with multiple lines
                if (i<(len(the_json)-1)) & (len(part)>1):
                    json_str = json_str.replace(',',',\n\t\t')
                    json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t},')
                    f.write(json_str)
                # Case for ending JSON block with a single line
                if (i==(len(the_json)-1)) & (len(part)<2):
                    json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t}\n]')
                    f.write(json_str)
                # Case for ending JSON block with multiple lines
                if (i==(len(the_json)-1)) & (len(part)>1):
                    json_str = json_str.replace(',',',\n\t\t')
                    json_str = str('\n\t'+'{'+'\n\t\t'+json_str+'\n\t}\n]')
                    f.write(json_str)
        with open(json_pipeline_mod) as json_file:
            the_json = json.load(json_file) 
        run_pdal(json_pipeline_mod,bounds_json,filename_tif)
        
        # Clip DEMs
        # clip_geom = gpd.GeoDataFrame(geometry=gpd.GeoSeries(tile2009_sing.geometry),crs='EPSG:6344')
        # filename_tif = os.path.join(dem2009_root,os.path.splitext(os.path.basename(filename_laz))[0]+'.tif')
        # [clip2009,errorState] = raster_clip(filename_tif,clip_geom,clip_geom.crs)
        # filename_tif = os.path.join(dem2009_root, 'clip',os.path.splitext(os.path.basename(filename_laz))[0]+'.tif')
        # DEM = read_file(clip2009).read(1,masked=True)
        # DEM.shape = [1,np.shape(DEM)[0],np.shape(DEM)[1]]
        # with rasterio.open(filename_tif, "w", **read_file(clip2009).meta) as dest:
        #     dest.write(DEM)
        # out_name = os.path.join(dem_fill_2009_root,os.path.splitext(os.path.basename(filename_laz))[0]+'.tif')
        # wbt.fill_missing_data(filename_tif,out_name,filter=15,weight=5.0,no_edges=True)
        # clip2009 = None
        # dest = None
        # filename_tif = None
        # DEM = None
        
        # filename_tif_2019 = os.path.join(dem2019_root,basename)  
        # [clip2019, errorState] = raster_clip(filename_tif_2019, clip_geom, clip_geom.crs)
        # filename_tif = os.path.join(dem2019_root, 'clip',basename)
        # DEM = read_file(clip2019).read(1,masked=True)
        # DEM.shape = [1,np.shape(DEM)[0],np.shape(DEM)[1]]
        # with rasterio.open(filename_tif, "w", **read_file(clip2019).meta) as dest:
        #     dest.write(DEM)
        # out_name = os.path.join(dem_fill_2019_root,basename)
        # wbt.fill_missing_data(filename_tif,out_name,filter=15,weight=5.0,no_edges=True)
        # clip2019 = None
        # dest = None
        # filename_tif = None
        # DEM = None
    else:
        print("No overlapping tiles")
        
    print("Finished with:",tile2009_sing['Name'])   

# from joblib import Parallel, delayed
# Parallel(n_jobs=4)(delayed(rasterize_points(count,tiles2009, laz2009_root, tiles2019, laz2019_root, dem2009_root, dem2019_root, json_pipeline, json_pipeline_mod) for count in range(0,len(tiles2009-1))))    

for count in range(0,len(tiles2009)-1):
    rasterize_points(count, tiles2009, laz2009_root, tiles2019, laz2019_root, dem2009_root_buffer, dem2019_root_buffer, dem_fill_2009_root, dem_fill_2019_root, json_pipeline, json_pipeline_mod)

# %% Clip DEMs
    
    
[dem_paths_2009, dem_filenames_2009] = read_paths(dem2009_root_buffer,'.tif')
for count, value in enumerate(dem_paths_2009):
    # Debugging 
    # count = 0
    # value = dem_paths_2009[count]
    
    substring = dem_filenames_2009[count][:-4]
    tile2009_sing = [tiles2009[tiles2009['Name']==i] for i in tiles2009['Name'] if i[:-4] in substring][0]
    clip_geom = gpd.GeoDataFrame(geometry=gpd.GeoSeries(tile2009_sing.geometry),crs='EPSG:32615')
    [clip2009,errorState] = raster_clip(value, clip_geom, clip_geom.crs)
    if clip2009==[]:
        print("No overlapping data ",count)
    else:
        filename_tif = os.path.join(dem2009_root, 'clip', dem_filenames_2009[count])
        DEM = read_file(clip2009).read(1,masked=True)
        DEM.shape = [1,np.shape(DEM)[0],np.shape(DEM)[1]]
        with rasterio.open(filename_tif, "w", **read_file(clip2009).meta) as dest:
            dest.write(DEM)
    clip2009 = None
    dest = None
    filename_tif = None
    DEM = None
 
[dem_paths_2019, dem_filenames_2019] = read_paths(dem2019_root_buffer,'.tif')
for count, value in enumerate(dem_paths_2019):
    # Debugging 
    # count = 0
    # value = dem_paths_2009[count]
    
    substring = dem_filenames_2019[count][8:-4]
    tile2009_sing = [tiles2009[tiles2009['Name']==i] for i in tiles2009['Name'] if i[14:-4] in substring][0]
    clip_geom = gpd.GeoDataFrame(geometry=gpd.GeoSeries(tile2009_sing.geometry),crs='EPSG:32616')
    [clip2019,errorState] = raster_clip(value,clip_geom,clip_geom.crs)
    if clip2019==[]:
        print("No overlapping data ",count)
    else:
        filename_tif = os.path.join(dem2020_root, 'clip', dem_filenames_2019[count])
        DEM = read_file(clip2019).read(1,masked=True)
        DEM.shape = [1,np.shape(DEM)[0],np.shape(DEM)[1]]
        with rasterio.open(filename_tif, "w", **read_file(clip2019).meta) as dest:
            dest.write(DEM)
    clip2019 = None
    dest = None
    filename_tif = None
    DEM = None
# %% Interpolate missing raster data
[dem_paths_2009, dem_filenames_2009] = read_paths(os.path.join(dem2012_root, 'clip'), '.tif')
os.chdir(dem_fill_2009_root)
wbt.set_compress_rasters(True)
for count,value in enumerate(dem_paths_2009):
    out_name = os.path.join(dem_fill_2009_root,dem_filenames_2009[count])
    wbt.fill_missing_data(value,out_name,filter=15,weight=5.0,no_edges=True,)

[dem_paths_2019, dem_filenames_2019] = read_paths(os.path.join(dem2020_root, 'clip'), '.tif')
os.chdir(dem_fill_2019_root)
for count,value in enumerate(dem_paths_2019):
    out_name = os.path.join(dem_fill_2019_root,dem_filenames_2019[count])
    wbt.fill_missing_data(value,out_name,filter=15,weight=5.0,no_edges=True)  

# %% Generate DODs

[dem_paths_2009, dem_filenames_2009] = read_paths(dem_fill_2009_root,'.tif')
[dem_paths_2019, dem_filenames_2019] = read_paths(dem_fill_2019_root,'.tif')
dod_path = Path(r'D:\CJR\lake_michigan_dod\tiles')
for count,value in enumerate(dem_paths_2009):
    # count=56
    # value = dem_paths_2009[count]
    template = dem_filenames_2009[count][-16:]
    # if count==62:     
    #     template = dem_filenames_2009[count][-17:] 
    filepath_2019 = [i for i in dem_paths_2019 if template in i]
    # outfile = r'D:\CJR\LakeSuperior_DOD\reprojtry.tif'
    # reproj_match(infile=filepath_2019[0],match=value,outfile=outfile)
    if filepath_2019==[]:
        print("No overlapping data ",count)
    else:
        [reproj2019,error_state] = reproj_match(infile=filepath_2019[0],match=value)
        DEM_2009 = read_file(value)
        if error_state==False:
            DOD = reproj2019.read(1,masked=True)-DEM_2009.read(1,masked=True)
            DOD.shape = [1,np.shape(DOD)[0],np.shape(DOD)[1]]
            outname_base = dem_filenames_2009[count][8:]
            outname = str("DOD"+outname_base)
            outname = os.path.join(dod_path,outname)
            with rasterio.open(outname, "w", **DEM_2009.meta) as dest:
                dest.write(DOD)
                print("DOD generated,",count)
        else:
             print("No overlapping data ",count)

# %% Generate transects using 2009 data
        
outdir = r"D:\CJR\lake_michigan_bluff_delineation\2012"
os.chdir(outdir)
os.chdir(r'..')
shoreline_clip_poly = gpd.read_file(r'wisconsin_shoreline_clip.shp')
shoreline_clip_poly = shoreline_clip_poly.to_crs('EPSG:32616')
shoreline = gpd.read_file(Path(homestr+r'.\LakeSuperior_BluffDelin\great_lakes_hardened_shorelines_cliff_delin_edit.shp'))
shoreline = shoreline.to_crs('EPSG:32616')
shoreline = gpd.clip(shoreline.geometry, shoreline_clip_poly.geometry)
tiles2012 = tiles2012.to_crs('EPSG:32616')
for count in range(0, len(tiles2012)-1):
    print('Working on tile ',count,' of',len(tiles2012),' .')
    # Debugging
    # count = 363
    # Define parameters
    tile2012_sing = tiles2012.iloc[count]  # pull out a single 2009 tile
    bounds = tile2012_sing.geometry.bounds
    shoreline_clip = gpd.clip(shoreline.geometry, tile2012_sing.geometry)
    if len(shoreline_clip)>0:
        # if len(shoreline_sig)>1:
        #     lines = [i for i in shoreline_sing.geometry]
        #     multi_line = MultiLineString(lines)
        #     shoreline_sing_mod = shapely.ops.linemerge(multi_line)
        tile_name = os.path.splitext(tile2012_sing.Name)[0]
        xsec_spacing = 5
        poslength = 120.0
        neglength = 30.0
        spacing = xsec_spacing
        simp_tolerance = 20
        step = 1.0
        crs = 32616
        
        merge_tiles = tiles2012[shapely.intersects(buffer(tile2012_sing.geometry, 200), tiles2012.geometry).values]
        merge_tiles = [os.path.join(dem_fill_2012_root, (os.path.splitext(i)[0]+'.tif')) for i in merge_tiles['Name']]
        merge_tiles = list(set(dem_paths_2009).intersection(set(merge_tiles)))
        merged_dem = temp_merge(merge_tiles)
        gen_xsec_wrap(shoreline_clip, outdir, tile_name, poslength, neglength, xsec_spacing, simp_tolerance, step, merged_dem, crs=crs)
        os.chdir(os.path.join(outdir,'Transects'))
        search_pattern=str(tile_name+'*.shp')
        transect_files = (glob.glob(search_pattern))
        transects = pd.DataFrame()
        for file in transect_files:
            transect = gpd.read_file(file)
            transects = pd.concat([transects,transect]) 
        # transect_file = os.path.join(outdir,'Transects',(tile_name+'_transects.shp'))
        xsec_points = make_xsec_points(transects,merged_dem)
        xsec_points.to_csv(os.path.join(outdir,'delineation_points_text',(tile_name+'_points.txt')),index=False)
        dem_merge = None
# %% Generate 2019 transect data
outdir = r'D:\CJR\lake_michigan_bluff_delineation\2020'
os.chdir(outdir)
os.chdir(r'..')
shoreline_clip_poly = gpd.read_file(r'wisconsin_shoreline_clip.shp')
shoreline_clip_poly = shoreline_clip_poly.to_crs('EPSG:32616')
shoreline = gpd.read_file(Path(homestr+r'.\LakeSuperior_BluffDelin\great_lakes_hardened_shorelines_cliff_delin_edit.shp'))
shoreline = shoreline.to_crs('EPSG:32616')
shoreline = gpd.clip(shoreline.geometry, shoreline_clip_poly.geometry)
tiles2012 = tiles2012.to_crs('EPSG:32616')
tiles2020 = tiles2020.to_crs('EPSG:32616')
for count in range(0, len(tiles2012)-1):
    # Debugging
    # count = 44
    tile2012_sing = tiles2012.iloc[count]  # pull out a single 2009 tile
    bounds = tile2012_sing.geometry.bounds
    shoreline_clip = gpd.clip(shoreline.geometry, tile2012_sing.geometry)
    if len(shoreline_clip)>0:
        tile_name = os.path.splitext(tile2012_sing.Name)[0]
        merge_tiles = tiles2012[shapely.intersects(buffer(tile2012_sing.geometry, 200), tiles2012.geometry).values]
        merge_tiles = [str(os.path.splitext(i)[0][14:]+'.tif') for i in merge_tiles['Name']]
        merge_tile_paths = []
        for filepath in dem_paths_2019:
            if any(merge_tile in filepath for merge_tile in merge_tiles):
                merge_tile_paths.append(filepath)
        merged_dem = temp_merge(merge_tile_paths)
        os.chdir(r'D:\CJR\LakeSuperior_BluffDelin\2009\Transects')
        search_pattern=str(tile_name+'*.shp')
        transect_files = (glob.glob(search_pattern))
        transects = pd.DataFrame()
        for file in transect_files:
            transect = gpd.read_file(file)
            transects = pd.concat([transects,transect]) 
        # transect_file = os.path.join(outdir,'Transects',(tile_name+'_transects.shp'))
        xsec_points = make_xsec_points(transects,merged_dem)
        xsec_points.to_csv(os.path.join(outdir,'delineation_points_text',(tile_name+'_points.txt')),index=False)
        dem_merge = None

# %% Create merged transects

os.chdir(r'D:\CJR\LakeSuperior_BluffDelin\2009\Transects')
all_transects = pd.DataFrame()
for number, fileName in enumerate(glob.glob('*.shp')):
    transects = gpd.read_file(fileName)
    all_transects = pd.concat([all_transects, transects])
os.chdir(os.path.abspath(os.path.join(os.getcwd(), r'..\..')))
all_transects.to_file(f'lake_superior_transects.shp')

# %% Generating convex hull of cliff points, break into segments for data aggregation


# Append points
os.chdir(r'D:\CJR\LakeSuperior_BluffDelin\2009\params5_106')
base_2009_pts = gpd.read_file(r'2009_params5_base_points.shp')
top_2009_pts = gpd.read_file(r'2009_params5_top_points.shp')
os.chdir(r'D:\CJR\LakeSuperior_BluffDelin\2019\params5_106')
base_2019_pts = gpd.read_file(r'2019_base_points.shp')
top_2019_pts = gpd.read_file(r'2019_top_points.shp')
all_points = pd.concat([base_2009_pts, top_2009_pts, base_2019_pts, top_2019_pts])
points = np.stack([all_points.geometry.x, all_points.geometry.y], axis=-1)
points = points[~np.isnan(points[:, 0]),]

# Create hull of points
conchull = alphashape.alphashape(points, alpha=0.01)
ConcHullGDF = gpd.GeoDataFrame(index=[0], crs=all_points.crs, geometry=[conchull])
ConcHullGDF.plot()
os.chdir(Path(r'D:\CJR\LakeSuperior_BluffDelin'))
# ConcHullGDF.to_file('lake_superior_hull_params5_106.shp')

# Merge simpllfied shoreline segments
all_shorelines = pd.DataFrame()
os.chdir(r'D:\CJR\LakeSuperior_BluffDelin\2009\Shorelines')
for number, fileName in enumerate(glob.glob('*.shp')):
    shorelines = gpd.read_file(fileName)
    all_shorelines = pd.concat([all_shorelines, shorelines])
os.chdir(r'D:\CJR\LakeSuperior_BluffDelin')
# all_shorelines.to_file(f'lake_superior_simplified_shorelines.shp')

# Clip and simplify hardened shoreline segments from transect generation
outdir = r"D:\CJR\LakeSuperior_BluffDelin"
os.chdir(outdir)
shoreline_clip_poly = gpd.read_file(r'BayfieldShorelineClip.shp')
shoreline_clip_poly = shoreline_clip_poly.to_crs('EPSG:32615')
shoreline = gpd.read_file(Path(homestr+r'.\LakeSuperior_BluffDelin\great_lakes_hardened_shorelines_cliff_delin_edit.shp'))
shoreline = shoreline.to_crs('EPSG:32615')
shoreline = gpd.clip(shoreline.geometry, shoreline_clip_poly.geometry)
shoreline_merge = shapely.ops.unary_union(shoreline.geometry)
shoreline_merge = gpd.GeoSeries(data=shoreline_merge, crs=shoreline.crs)

# Simplify and break clipped entire shoreline into segments for data aggregation, intersect with convex hull polygon to generate segmented polygons
reach_length = 500.0 
tolerance = 0.1
line = shoreline_merge.geometry
line_merge = shapely.ops.linemerge(shoreline_merge[0])
points = []
for i in [0,1,2]:
    points_sing = [Point(xy) for xy in zip(line_merge.geoms[i].coords.xy[0], line_merge.geoms[i].coords.xy[1])]
    if i == 0:
        points_sing.reverse()
    points.extend(points_sing)
line_merge = LineString(points)

simp_tolerance = 20
simplify_baseline = line_merge.simplify(simp_tolerance)
smooth_baseline = shapelysmooth.catmull_rom_smooth(simplify_baseline,alpha=0.9)
smooth_baseline = gpd.GeoSeries(data=smooth_baseline, crs=shoreline.crs)
line = smooth_baseline.geometry
segments = int(math.ceil(line[0].length/reach_length))
splitter = MultiPoint([line.interpolate((i/segments), normalized=True)[0] for i in range(1, segments)])
splitter2 = MultiPoint([line.interpolate(line.project(i))[0] for i in splitter.geoms])
line_split = split(snap(line[0], splitter2, tolerance), splitter2)
count = 0
os.chdir(r'D:\CJR\LakeSuperior_DOD\buffer_polys')
for geom in line_split.geoms:
    count_str = f"{count:03d}"
    buffer_pos = geom.buffer(150, single_sided=True)
    buffer_neg = geom.buffer(-200, single_sided=True)
    merge_buffer = shapely.ops.unary_union([buffer_pos, buffer_neg])
    # merge_buffer = gpd.GeoSeries(data=merge_buffer, crs = shoreline.crs)
    merge_buffer = gpd.GeoDataFrame(geometry=[merge_buffer], crs = shoreline.crs)
    merge_buffer['length_m'] = geom.length
    if count > 0:
        count_lag = count - 1
        count_lag_str = f"{count_lag:03d}"
        merge_buffer_prior = gpd.read_file(f'buffer_polygon_{count_lag_str}.shp')
        merge_buffer = merge_buffer.overlay(merge_buffer_prior, how='difference')
        merge_buffer.to_file(f'buffer_polygon_{count_str}.shp')
    else:
        merge_buffer.to_file(f'buffer_polygon_{count_str}.shp')
    count += 1

# %% Load manually edited polygon segments, clip DOD's and calculate volume change statistics
#  Vertical uncertainty 
#  Bayfield 2009 - 0.036 meters RMSE
#  JALBTCX 2019 - Compiled to meet 20 cm vertical accuracy at 95% confidence leve
#  Acquisitions dates
#  Bayfield 2009 - March 2009
#  JALBTXC 2019 - 20190821-20190907
[buffer_poly_filenames, buffer_poly_filepaths] = read_paths(r'D:\CJR\LakeSuperior_DOD\buffer_polys', '.shp')
coastal_clip = gpd.read_file(r'D:\CJR\LakeSuperior_BluffDelin\lake_superior_hull_params5_106_edit_bark_bay.shp')
dod_root = Path(r'D:\CJR\LakeSuperior_DOD\tiles')
shoreline = gpd.read_file(Path(homestr+r'.\LakeSuperior_BluffDelin\great_lakes_hardened_shorelines_cliff_delin_edit.shp'))
shoreline = shoreline.to_crs('EPSG:32615')
shoreline = gpd.clip(shoreline, shoreline_clip_poly.geometry)

lod = 0.20
count = 0
for buffer_sing in buffer_poly_filenames:
    # Debugging
    # buffer_sing = gpd.read_file(buffer_poly_filenames[5])
    count_str = f"{count:03d}"
    buffer_sing = gpd.read_file(buffer_sing)
    shoreline_sing = gpd.clip(shoreline, buffer_sing.geometry[0])
    if len(shoreline_sing) > 0:
        shoreline_sing.length_m = shoreline_sing.length
        dom_type = shoreline_sing['Shorelin_1'][shoreline_sing.length_m.idxmax()]
    else:
        dom_type = shoreline_sing['Shorelin_1'][0]
    dod_cell = coastal_clip.overlay(buffer_sing, how='intersection')
    if len(dod_cell) > 0:
        dod_tiles = tiles2009[tiles2009.geometry.intersects(dod_cell.geometry[0])] 
        dod_tiles = dod_tiles.reset_index()
        merge_dod_paths = [os.path.join(dod_root,'DOD' + os.path.splitext(i)[0][-9:] + '.tif') for i in dod_tiles['Name']]
        dods = [read_file(i) for i in merge_dod_paths]
        dod_merge, output_transform = merge(dods)
        dod_merge.shape = [1, np.shape(dod_merge)[1], np.shape(dod_merge)[2]]
        meta = dods[0].meta
        meta.update({"height":dod_merge.shape[1],
                     "width": dod_merge.shape[2],
                     "transform": output_transform})
        dod_merge = open_memory_tif(dod_merge[0], meta)
        
        [dod_coastal_clip, error_state] = raster_clip_from_dataset(dod_merge, dod_cell.geometry, coastal_clip.crs)
        os.chdir(r'D:\CJR\LakeSuperior_DOD\summary_segments')
        filename_tif = f'2019min2009_demfill_coastal_clip_{count_str}.tif'
        dod_clip = read_file(dod_coastal_clip).read(1,masked=True)
        dod_clip.shape = [1,np.shape(dod_clip)[0],np.shape(dod_clip)[1]]
        with rasterio.open(filename_tif, "w", **read_file(dod_coastal_clip).meta) as dest:
            dest.write(dod_clip)
        stats_df = compute_volume_change_stats(dod_clip, dod_merge, lod)
        stats_df = stats_df.apply(pd.to_numeric, errors='ignore')
        dod_cell_stats = pd.concat([stats_df, dod_cell], axis=1)
        dod_cell_stats = gpd.GeoDataFrame(data=dod_cell_stats, crs=coastal_clip.crs)
        dod_cell_stats['shoreline'] = dom_type
        os.chdir(r'D:\CJR\LakeSuperior_DOD\buffer_polys_coastal')
        filename_poly = f'2019min2009_demfill_coastal_clip_{count_str}.shp'
        dod_cell_stats.to_file(filename_poly)
    count += 1
    
# Merge polygons to single shapefile
all_shorelines = pd.DataFrame()
os.chdir(r'D:\CJR\LakeSuperior_DOD\buffer_polys_coastal')
for number, fileName in enumerate(glob.glob('*.shp')):
    shorelines = gpd.read_file(fileName)
    all_shorelines = pd.concat([all_shorelines, shorelines])
os.chdir(r'D:\CJR\LakeSuperior_DOD\buffer_polys_coastal')
os.chdir(os.path.abspath(r'..'))
all_shorelines.to_file(f'lake_superior_coastal_dod_polys.shp')

# %% Generate crest and toe recession rates, extract bluff heights and slopes, add uncertainty

# Read data
coastal_clip = gpd.read_file(r'D:\CJR\LakeSuperior_BluffDelin\lake_superior_hull_params5_106_edit_bark_bay.shp')
transects = gpd.read_file(r'D:\CJR\LakeSuperior_BluffDelin\lake_superior_transects.shp')
crest_2009 = gpd.read_file(r'D:\CJR\LakeSuperior_BluffDelin\2009\params5_106\2009_params5_top_points.shp')
toe_2009 = gpd.read_file(r'D:\CJR\LakeSuperior_BluffDelin\2009\params5_106\2009_params5_base_points.shp')
crest_2019 = gpd.read_file(r'D:\CJR\LakeSuperior_BluffDelin\2019\params5_106\2019_top_points.shp')
toe_2019 =  gpd.read_file(r'D:\CJR\LakeSuperior_BluffDelin\2019\params5_106\2019_base_points.shp')
dem_2009 = read_file(r'D:\CJR\Bayfield2009\2009_dem_fill_cog_lzw.tif')
dem_2019 = read_file(r'D:\CJR\JABLTX_2019\2019_DEM_NOAA_merge_cog_lzw.tif')

point_uncert = 2
years_interval = 10.5

# Extract point data to transects
transects_filt = transects[transects.geometry.intersects(coastal_clip.geometry[0])] 
# transects_filt = transects_filt.reset_index()
# toe_2009_filt = toe_2009[toe_2009.geometry.intersects(transects_filt.geometry)]

dist_toe_2009_all = []
elev_toe_2009_all = []
dist_toe_2019_all = []
elev_toe_2019_all = []
dist_crest_2009_all = []
elev_crest_2009_all = []
dist_crest_2019_all = []
elev_crest_2019_all = []

for count, transect in enumerate(transects_filt.geometry):
    [dist_toe_2009, elev_toe_2009] = point_transect_distance(transect, toe_2009, dem_2009)
    dist_toe_2009_all.append(dist_toe_2009), elev_toe_2009_all.append(elev_toe_2009)
    [dist_toe_2019, elev_toe_2019] = point_transect_distance(transect, toe_2019, dem_2019)
    dist_toe_2019_all.append(dist_toe_2019), elev_toe_2019_all.append(elev_toe_2019)
    [dist_crest_2009, elev_crest_2009] = point_transect_distance(transect, crest_2009, dem_2009)
    dist_crest_2009_all.append(dist_crest_2009), elev_crest_2009_all.append(elev_crest_2009)
    [dist_crest_2019, elev_crest_2019] = point_transect_distance(transect, crest_2019, dem_2019)
    dist_crest_2019_all.append(dist_crest_2019), elev_crest_2019_all.append(elev_crest_2019)
    
transects_filt['dist_toe_2009'] = dist_toe_2009_all
transects_filt['dist_toe_2019'] = dist_toe_2019_all
transects_filt['dist_crest_2009'] = dist_crest_2009_all
transects_filt['dist_crest_2019'] = dist_crest_2019_all
transects_filt['elev_toe_2009'] = elev_toe_2009_all
transects_filt['elev_toe_2019'] = elev_toe_2019_all
transects_filt['elev_crest_2009'] = elev_crest_2009_all
transects_filt['elev_crest_2019'] = elev_crest_2019_all

# Filter to  transecs that have  a contemporaneous toe and crest, calculate height metrics
transects_elev_filt_2009 = transects_filt[(transects_filt['elev_toe_2009'] > 0.0) & (transects_filt['elev_crest_2009'] > 0.0)]
transects_elev_filt_2009['bluff_height'] = np.abs(transects_elev_filt_2009.elev_crest_2009 - transects_elev_filt_2009.elev_toe_2009)
transects_elev_filt_2009['slope_length'] = np.abs(transects_elev_filt_2009.dist_crest_2009 - transects_elev_filt_2009.dist_toe_2009)
transects_elev_filt_2009['slope'] = transects_elev_filt_2009.bluff_height / transects_elev_filt_2009.slope_length
transects_elev_filt_2009['slope_deg'] = np.rad2deg(np.arctan(transects_elev_filt_2009.slope))
transects_elev_filt_2019 = transects_filt[(transects_filt['elev_toe_2019'] > 0.0) & (transects_filt['elev_crest_2019'] > 0.0)]
transects_elev_filt_2019['bluff_height'] = transects_elev_filt_2019.elev_crest_2019 - transects_elev_filt_2009.elev_toe_2019
transects_elev_filt_2019['slope_length'] = np.abs(transects_elev_filt_2019.dist_crest_2019 - transects_elev_filt_2019.dist_toe_2019)
transects_elev_filt_2019['slope'] = transects_elev_filt_2019.bluff_height / transects_elev_filt_2019.slope_length
transects_elev_filt_2019['slope_deg'] = np.rad2deg(np.arctan(transects_elev_filt_2019.slope))

# Filter to transects that have either both crests or boths toes, separately
transects_toe_filt = transects_filt[(transects_filt['dist_toe_2009'] > 0) & (transects_filt['dist_toe_2019'] > 0)]
transects_toe_filt['net_toe_movement'] = transects_toe_filt['dist_toe_2019'] - transects_toe_filt['dist_toe_2009']
transects_toe_filt['toe_epr'] =  transects_toe_filt['net_toe_movement'] / years_interval
transects_crest_filt = transects_filt.drop(transects_filt.index[0:178])
transects_crest_filt = transects_crest_filt.reset_index()
transects_crest_filt = transects_crest_filt[(transects_crest_filt['dist_crest_2009'] > 0) & (transects_crest_filt['dist_crest_2019'] > 0)]
transects_crest_filt['net_crest_movement'] = transects_crest_filt['dist_crest_2019'] - transects_crest_filt['dist_crest_2009']
transects_crest_filt['crest_epr'] =  transects_crest_filt['net_crest_movement'] / years_interval

# Join all transect data back to all transects within DOD
transects_filt_2009 = transects_filt.sjoin(transects_elev_filt_2009, how='left', predicate='covers')[['bluff_height', 'slope_length', 'slope', 'slope_deg']]
transects_filt_2009 = transects_filt_2009.rename(columns={'bluff_height':'bh_2009', 'slope_length':'sl_2009', 'slope':'slp2009', 'slope_deg':'slpd2009'})
transects_filt_2019 = transects_filt.sjoin(transects_elev_filt_2019, how='left', predicate='covers')[['bluff_height', 'slope_length', 'slope', 'slope_deg']]
transects_filt_2019 = transects_filt_2019.rename(columns={'bluff_height':'bh_2019', 'slope_length':'sl_2019', 'slope':'slp2019', 'slope_deg':'slpd2019'})
transects_filt_toe = transects_filt.sjoin(transects_toe_filt, how='left', predicate='covers')[['net_toe_movement', 'toe_epr']]
transects_filt_toe = transects_filt_toe.rename(columns={'net_toe_movement':'nsm_toe'})
transects_filt_crest = transects_filt.sjoin(transects_crest_filt, how='left', predicate='covers')[['net_crest_movement', 'crest_epr']]
transects_filt_crest = transects_filt_crest.rename(columns={'net_crest_movement':'nsm_crest'})


transects_filt2 = transects_filt.merge(transects_filt_2009, how='left', on=transects_filt.index)
transects_filt2 = transects_filt2.drop(['key_0'], axis=1)
transects_filt2 = transects_filt2.merge(transects_filt_2019, how ='left', on=transects_filt.index)
transects_filt2 = transects_filt2.drop(['key_0'], axis=1)
transects_filt2 = transects_filt2.merge(transects_filt_toe, how='left', on=transects_filt.index)
transects_filt2 = transects_filt2.drop(['key_0'], axis=1)
transects_filt2 = transects_filt2.merge(transects_filt_crest, how='left', on=transects_filt.index)
transects_filt2 = transects_filt2.drop(['key_0'], axis=1)

transects_filt2['slp_chg'] = transects_filt2['slpd2019'] - transects_filt2['slpd2009']
transects_filt2['nsm_dif'] = transects_filt2['nsm_toe'] - transects_filt2['nsm_crest']
transects_filt2.to_file(r'D:\CJR\LakeSuperior_BluffDelin\lake_superior_transects_attributes.shp')

os.chdir(r'D:\CJR\LakeSuperior_DOD\buffer_polys_coastal')
os.chdir(os.path.abspath(r'..'))
dod_segments = gpd.read_file(f'lake_superior_coastal_dod_polys.shp')

colnames = ['FID', 't_rec_md', 't_rec_mn', 't_rec_max', 't_rec_min', 't_rec_std', 
            'frac_t_rl', 'c_rec_md', 'c_rec_mn', 'c_rec_max', 'c_rec_min',
            'c_rec_std', 'frac_c_rl', 'frac_bh09', 'frac_bh19', 'bh09_md',
            'bh09_mn', 'bh09_max', 'bh09_min', 'bh09_std', 'slpd09_mn', 'slpd09_md',
            'slpd09_max', 'slpd09_min', 'slpd09_std', 'bh19_md',  'bh19_mn',
            'bh19_max', 'bh19_min', 'bh19_std', 'slpd19_mn', 'slpd19_md',
            'slpd19_max', 'slpd19_min', 'slpd19_std', 'slpchg_md', 'slpchg_mn',
            'slpchg_max', 'slpchg_min', 'slpchg_std', 'frac_nsm_rl', 'nsm_df_mn',
            'nsm_df_md', 'nsm_df_max', 'nsm_df_min', 'nsm_df_std']
recession_df_all = pd.DataFrame(columns = colnames)
dod_segments['FID'] = dod_segments.index
for count, poly in enumerate(dod_segments.geometry):
    #Debugging
    # count = 30
    # poly = dod_segments.geometry[30]
    transects_filt_poly = transects_filt2[transects_filt2.geometry.intersects(poly)]
    recession_df_sing = pd.DataFrame(columns = colnames)
    recession_df_sing.at[0, 'frac_t_rl'] = np.count_nonzero(transects_filt_poly['nsm_crest'].notnull()) / np.count_nonzero(transects_filt_poly['FID']) 
    recession_df_sing.at[0, 'frac_c_rl'] = np.count_nonzero(transects_filt_poly['nsm_crest'].notnull()) / np.count_nonzero(transects_filt_poly['FID']) 
    recession_df_sing.at[0, 'frac_bh09'] = np.count_nonzero(transects_filt_poly['bh_2009'].notnull()) / np.count_nonzero(transects_filt_poly['FID']) 
    recession_df_sing.at[0, 'frac_bh19'] = np.count_nonzero(transects_filt_poly['bh_2019'].notnull()) / np.count_nonzero(transects_filt_poly['FID']) 
    recession_df_sing.at[0, 'frac_nsm_rl'] = np.count_nonzero(transects_filt_poly['nsm_dif'].notnull()) / np.count_nonzero(transects_filt_poly['FID']) 
    recession_df_sing.at[0, 'FID'] = dod_segments['FID'][count]
    if recession_df_sing.at[0, 'frac_t_rl'] > 0.2:
        recession_df_sing.at[0, 't_rec_md'] = np.nanmedian(transects_filt_poly['nsm_toe']) 
        recession_df_sing.at[0, 't_rec_mn'] = np.nanmean(transects_filt_poly['nsm_toe']) 
        recession_df_sing.at[0, 't_rec_max'] = np.max(transects_filt_poly['nsm_toe']) 
        recession_df_sing.at[0, 't_rec_min'] = np.min(transects_filt_poly['nsm_toe']) 
        recession_df_sing.at[0, 't_rec_std'] = np.std(transects_filt_poly['nsm_toe']) 
    else:
        recession_df_sing.at[0, 't_rec_md'] = int(-9999)
        recession_df_sing.at[0, 't_rec_mn'] = -9999
        recession_df_sing.at[0, 't_rec_max'] = -9999
        recession_df_sing.at[0, 't_rec_min'] = -9999
        recession_df_sing.at[0, 't_rec_std'] = -9999
    if recession_df_sing.at[0, 'frac_c_rl'] > 0.2:
        recession_df_sing.at[0, 'c_rec_md'] = np.nanmedian(transects_filt_poly['nsm_crest']) 
        recession_df_sing.at[0, 'c_rec_mn'] = np.nanmean(transects_filt_poly['nsm_crest']) 
        recession_df_sing.at[0, 'c_rec_max'] = np.max(transects_filt_poly['nsm_crest']) 
        recession_df_sing.at[0, 'c_rec_min'] = np.min(transects_filt_poly['nsm_crest']) 
        recession_df_sing.at[0, 'c_rec_std'] = np.std(transects_filt_poly['nsm_crest']) 
    else:
        recession_df_sing.at[0, 'c_rec_md'] = -9999
        recession_df_sing.at[0, 'c_rec_mn'] = -9999
        recession_df_sing.at[0, 'c_rec_max'] = -9999
        recession_df_sing.at[0, 'c_rec_min'] = -9999
        recession_df_sing.at[0, 'c_rec_std'] = -9999
    if recession_df_sing.at[0, 'frac_bh09'] > 0.2:
        recession_df_sing.at[0, 'bh09_md'] = np.nanmedian(transects_filt_poly['bh_2009']) 
        recession_df_sing.at[0, 'bh09_mn'] = np.nanmean(transects_filt_poly['bh_2009']) 
        recession_df_sing.at[0, 'bh09_max'] = np.max(transects_filt_poly['bh_2009']) 
        recession_df_sing.at[0, 'bh09_min'] = np.min(transects_filt_poly['bh_2009']) 
        recession_df_sing.at[0, 'bh09_std'] = np.std(transects_filt_poly['bh_2009']) 
        recession_df_sing.at[0, 'slpd09_md'] = np.nanmedian(transects_filt_poly['slpd2009']) 
        recession_df_sing.at[0, 'slpd09_mn'] = np.nanmean(transects_filt_poly['slpd2009']) 
        recession_df_sing.at[0, 'slpd09_max'] = np.max(transects_filt_poly['slpd2009']) 
        recession_df_sing.at[0, 'slpd09_min'] = np.min(transects_filt_poly['slpd2009']) 
        recession_df_sing.at[0, 'slpd09_std'] = np.std(transects_filt_poly['slpd2009']) 
    else:
        recession_df_sing.at[0, 'bh09_md'] = -9999
        recession_df_sing.at[0, 'bh09_mn'] = -9999
        recession_df_sing.at[0, 'bh09_max'] = -9999
        recession_df_sing.at[0, 'bh09_min'] = -9999
        recession_df_sing.at[0, 'bh09_std'] = -9999
        recession_df_sing.at[0, 'slpd09_md'] = -9999
        recession_df_sing.at[0, 'slpd09_mn'] = -9999
        recession_df_sing.at[0, 'slpd09_max'] = -9999
        recession_df_sing.at[0, 'slpd09_min'] = -9999
        recession_df_sing.at[0, 'slpd09_std'] = -9999
    if recession_df_sing.at[0, 'frac_bh19'] > 0.2:
        recession_df_sing.at[0, 'bh19_md'] = np.nanmedian(transects_filt_poly['bh_2019']) 
        recession_df_sing.at[0, 'bh19_mn'] = np.nanmean(transects_filt_poly['bh_2019']) 
        recession_df_sing.at[0, 'bh19_max'] = np.max(transects_filt_poly['bh_2019']) 
        recession_df_sing.at[0, 'bh19_min'] = np.min(transects_filt_poly['bh_2019']) 
        recession_df_sing.at[0, 'bh19_std'] = np.std(transects_filt_poly['bh_2019']) 
        recession_df_sing.at[0, 'slpd19_md'] = np.nanmedian(transects_filt_poly['slpd2019']) 
        recession_df_sing.at[0, 'slpd19_mn'] = np.nanmean(transects_filt_poly['slpd2019']) 
        recession_df_sing.at[0, 'slpd19_max'] = np.max(transects_filt_poly['slpd2019']) 
        recession_df_sing.at[0, 'slpd19_min'] = np.min(transects_filt_poly['slpd2019']) 
        recession_df_sing.at[0, 'slpd19_std'] = np.std(transects_filt_poly['slpd2019']) 
    else:
        recession_df_sing.at[0, 'bh19_md'] = -9999
        recession_df_sing.at[0, 'bh19_mn'] = -9999
        recession_df_sing.at[0, 'bh19_max'] = -9999
        recession_df_sing.at[0, 'bh19_min'] = -9999
        recession_df_sing.at[0, 'bh19_std'] = -9999
        recession_df_sing.at[0, 'slpd19_md'] = -9999
        recession_df_sing.at[0, 'slpd19_mn'] = -9999
        recession_df_sing.at[0, 'slpd19_max'] = -9999
        recession_df_sing.at[0, 'slpd19_min'] = -9999
        recession_df_sing.at[0, 'slpd19_std'] = -9999
    if recession_df_sing.at[0, 'frac_nsm_rl'] > 0.2:
        recession_df_sing.at[0, 'slpchg_md'] = np.nanmedian(transects_filt_poly['slp_chg']) 
        recession_df_sing.at[0, 'slpchg_mn'] = np.nanmean(transects_filt_poly['slp_chg']) 
        recession_df_sing.at[0, 'slpchg_max'] = np.max(transects_filt_poly['slp_chg']) 
        recession_df_sing.at[0, 'slpchg_min'] = np.min(transects_filt_poly['slp_chg']) 
        recession_df_sing.at[0, 'slpchg_std'] = np.std(transects_filt_poly['slp_chg']) 
        recession_df_sing.at[0, 'nsm_df_md'] = np.nanmedian(transects_filt_poly['nsm_dif']) 
        recession_df_sing.at[0, 'nsm_df_mn'] = np.nanmean(transects_filt_poly['nsm_dif']) 
        recession_df_sing.at[0, 'nsm_df_max'] = np.max(transects_filt_poly['nsm_dif']) 
        recession_df_sing.at[0, 'nsm_df_min'] = np.min(transects_filt_poly['nsm_dif']) 
        recession_df_sing.at[0, 'nsm_df_std'] = np.std(transects_filt_poly['nsm_dif']) 
    else:
        recession_df_sing.at[0, 'slpchg_md'] = -9999
        recession_df_sing.at[0, 'slpchg_mn'] = -9999 
        recession_df_sing.at[0, 'slpchg_max'] = -9999
        recession_df_sing.at[0, 'slpchg_min'] = -9999
        recession_df_sing.at[0, 'slpchg_std'] = -9999
        recession_df_sing.at[0, 'nsm_df_md'] = -9999 
        recession_df_sing.at[0, 'nsm_df_mn'] = -9999
        recession_df_sing.at[0, 'nsm_df_max'] = -9999
        recession_df_sing.at[0, 'nsm_df_min'] = -9999
        recession_df_sing.at[0, 'nsm_df_std'] = -9999
    recession_df_all = pd.concat([recession_df_all, recession_df_sing])

recession_df_all = recession_df_all.reset_index()
coastal_poly = dod_segments.merge(recession_df_all, on='FID')
os.chdir(r'D:\CJR\LakeSuperior_DOD\buffer_polys_coastal')
os.chdir(os.path.abspath(r'..'))
coastal_poly.to_file(f'lake_superior_coastal_dod_polys_va.shp')
# %% Manuscript figures

os.chdir(r'D:\CJR\LakeSuperior_DOD\buffer_polys_coastal')
os.chdir(os.path.abspath(r'..'))
dod_segments = gpd.read_file(f'lake_superior_coastal_dod_polys_va.shp')
dod_segments['cumulative_length'] = np.cumsum(dod_segments['length_m'])
dod_segments['norm_net_vol'] = dod_segments['net_sum'] / dod_segments['length_m'] / 10.5
dod_segments['norm_net_vol_err'] = np.abs((dod_segments['net_sum_lo'] / dod_segments['length_m'] / 10.5) - dod_segments['norm_net_vol'])
# for count, value in enumerate(dod_segments.values[0]):
#     if np.ma.is_masked(value):
#         stats_df.values[0][count] = 0.0

crest_filt = dod_segments[pd.to_numeric(dod_segments['c_rec_md']) > -1000]     
toe_filt = dod_segments[pd.to_numeric(dod_segments['t_rec_md']) > -1000]  
slp_filt = dod_segments[pd.to_numeric(dod_segments['slpchg_md']) > -1000]          

miny = np.min(dod_segments.bounds.miny)
maxy = np.max(dod_segments.bounds.maxy)
minx = np.min(dod_segments.bounds.minx)
maxx = np.max(dod_segments.bounds.maxx)
# %%
# Major ticks every 20, minor ticks every 5
plt.rcParams['text.usetex']=False
major_ticks = np.arange(0, 80, 20)
minor_ticks = np.arange(0, 80, 10)
colors = {'Artificial Depositional (e.g., jetty, groin fill)': 'indianred',
          'Artificial Moderate Quality Moderately Engineered':'darkred',
       'Sandy Beach / Dune (relict deposits, areas with no new deposition)':'moccasin',
       'Open Shore Wetlands':'darkgreen',
       'Composite Bluffs (sand content 20-50%)':'lightblue',
       'Composite Bluffs (sand content 0-20%)':'mediumspringgreen',
       'Marine / Leda Clay Bluffs':'violet',
       'Sandy Beach / Dune Complex':'yellow',
       'Pocket Beach':'goldenrod',
       'Gravel Beaches':'gray',
       'Boulder Beaches':'navy',
       'Coarse Beaches':'darkgray'}
shoreline_colors = list(dod_segments['shoreline'].map(colors))

elinewidth = 0.5
labelfontsize = 8
tickfontsize = 8
plt.close('all')
color1 = 'black'
fig, axa = plt.subplots(7, 1, figsize=(8.5, 11), height_ratios=[5, 3, 1, 1, 1, 1, 1])
# dod_segments.plot(color = 'black', ax = axa[0])
# cx.add_basemap(ax = axa[0], source=cx.providers.OpenTopoMap, crs=dod_segments.crs, zoom=12, attribution = False)
# cx.add_basemap(ax = axa[0], source = cx.providers.Esri.WorldImagery, crs=dod_segments.crs, alpha = 0.7, attribution = False)
colors_bars = axa[1].bar(dod_segments['cumulative_length'], dod_segments['norm_net_vol'], 500,
           edgecolor = color1, linewidth=0.5, alpha =1, color = shoreline_colors, zorder=3)
axa[1].errorbar(dod_segments['cumulative_length'], dod_segments['norm_net_vol'],
                yerr = dod_segments['norm_net_vol_err'], alpha=0.8,
                zorder=5, marker='', linestyle='', ecolor=color1, elinewidth=elinewidth)
axa[2].bar(toe_filt['cumulative_length'], pd.to_numeric(toe_filt['t_rec_md']), 500, edgecolor = color1, linewidth=0.5, alpha = 1, color = 'lightgray', zorder=3)
axa[2].errorbar(toe_filt['cumulative_length'], pd.to_numeric(toe_filt['t_rec_md']),
                yerr = 1 * pd.to_numeric(toe_filt['t_rec_std']), alpha=0.8,
                zorder=5, marker='', linestyle='', ecolor=color1, elinewidth=elinewidth)
axa[3].bar(crest_filt['cumulative_length'], pd.to_numeric(crest_filt['c_rec_md']), 500, edgecolor = color1, linewidth=0.5, alpha = 1, color = 'lightgray', zorder=3)
axa[3].errorbar(crest_filt['cumulative_length'], pd.to_numeric(crest_filt['c_rec_md']),
                yerr = pd.to_numeric(crest_filt['c_rec_std']), alpha=0.8,
                zorder=5, marker='', linestyle='', ecolor=color1, elinewidth=elinewidth)
axa[4].bar(slp_filt['cumulative_length'], pd.to_numeric(slp_filt['slpchg_md']), 500, edgecolor = color1, linewidth=0.5, alpha = 1, color = 'lightgray', zorder=3)
axa[4].errorbar(slp_filt['cumulative_length'], pd.to_numeric(slp_filt['slpchg_md']),
                yerr = pd.to_numeric(slp_filt['slpchg_std']), alpha=0.8,
                zorder=5, marker='', linestyle='', ecolor=color1, elinewidth=elinewidth)
axa[5].bar(slp_filt['cumulative_length'], pd.to_numeric(slp_filt['bh09_md']), 500, edgecolor = color1, linewidth=0.5, alpha = 1, color = 'lightgray', zorder=3)
axa[5].errorbar(slp_filt['cumulative_length'], pd.to_numeric(slp_filt['bh09_md']),
                yerr = pd.to_numeric(slp_filt['bh09_std']), alpha=0.8,
                zorder=5, marker='', linestyle='', ecolor=color1, elinewidth=elinewidth)
axa[6].bar(slp_filt['cumulative_length'], pd.to_numeric(slp_filt['slpd09_md']), 500, edgecolor = color1, linewidth=0.5, alpha = 1, color = 'lightgray', zorder=3)
axa[6].errorbar(slp_filt['cumulative_length'], pd.to_numeric(slp_filt['slpd09_md']),
                yerr = pd.to_numeric(slp_filt['slpd09_md']), alpha=0.8,
                zorder=5, marker='', linestyle='', ecolor=color1, elinewidth=elinewidth)

plt.yticks(fontsize=12,color=color1,weight='normal')
plt.xticks(fontsize=12,color=color1,weight='normal')
axa[0].set_xticks([])
axa[0].set_xticklabels([])
axa[1].set_xticklabels([])
axa[2].set_xticklabels([])
axa[3].set_xticklabels([])
axa[4].set_xticklabels([])
axa[5].set_xticklabels([])
# axa[6].set_xticklabels([])

axa[0].set_yticks([])
axa[1].invert_yaxis()
# axa[1].set_ylabel(r"Normalized""\n""volume""\n""change rate""\n""($\mathrm{m}^3 \mathrm{m}^{-1} \mathrm{a}^{-1}$)",fontsize=labelfontsize,color=color1, weight='normal', rotation=0, horizontalalignment='left', verticalalignment='top')
axa[2].set_ylabel("Toe retreat\n(m)",fontsize=labelfontsize,color=color1, weight='normal', rotation=0, horizontalalignment='left', verticalalignment='top')
axa[3].set_ylabel(r"Crest retreat""\n""(m)",fontsize=labelfontsize,color=color1, weight='normal', rotation = 0, horizontalalignment='left', verticalalignment='top')
axa[4].set_ylabel(r"Slope increase ($^\circ$)",fontsize=labelfontsize,color=color1, weight='normal', rotation = 0, horizontalalignment='left', verticalalignment='top')
axa[5].set_ylabel("Bluff height in 2009 (m)",fontsize=labelfontsize,color=color1, weight='normal', rotation = 0, horizontalalignment='left', verticalalignment='top')
axa[6].set_ylabel(r"Bluff slope in 2009 ($^\circ$)",fontsize=labelfontsize,color=color1, weight='normal', rotation = 0, horizontalalignment='left', verticalalignment='top')
# axa[7].set_ylabel("Toe - crest retreat (meters)",fontsize=10,color=color1, weight='normal', rotation = 0)
axa[6].set_xlabel("Alongshore distance (meters)",fontsize=labelfontsize, color=color1, weight='normal', rotation = 0, horizontalalignment='left', verticalalignment='top')
for count, value in enumerate(axa):
    if (count > 1) & (count != 4):
        value.set_xlim(axa[1].get_xlim())
        value.set_ylim(0, axa[count].get_ylim()[1])
    if count == 4:
            value.set_ylim(-5, axa[count].get_ylim()[1])
for count, value in enumerate(axa):
    if (count > 0) & (count < 5):
        value.xaxis.set_ticks_position('none') 
        value.yaxis.set_label_coords(0.01, 0.9)
        # value.xaxis.set_major_locator(MultipleLocator(10000))
        # value.xaxis.set_minor_locator(AutoMinorLocator(5000))
        value.yaxis.set_major_locator(MultipleLocator(10))
        value.yaxis.set_minor_locator(AutoMinorLocator(5))
        value.grid(which='major', color='#CCCCCC', linestyle='--')
        value.grid(which='minor', color='#CCCCCC', linestyle=':')
    if (count == 5):
        value.xaxis.set_ticks_position('none') 
        value.yaxis.set_label_coords(0.01, 0.9)
        # value.xaxis.set_major_locator(MultipleLocator(10000))
        # value.xaxis.set_minor_locator(AutoMinorLocator(5000))
        value.yaxis.set_major_locator(MultipleLocator(20))
        value.yaxis.set_minor_locator(AutoMinorLocator(10))
        value.grid(which='major', color='#CCCCCC', linestyle='--')
        value.grid(which='minor', color='#CCCCCC', linestyle=':')
    if (count == 6):
        value.yaxis.set_label_coords(0.01, 0.9)
        # value.xaxis.set_major_locator(MultipleLocator(10000))
        # value.xaxis.set_minor_locator(AutoMinorLocator(5000))
        value.yaxis.set_major_locator(MultipleLocator(20))
        value.yaxis.set_minor_locator(AutoMinorLocator(10))
        value.grid(which='major', color='#CCCCCC', linestyle='--')
        value.grid(which='minor', color='#CCCCCC', linestyle=':')

axa[6].set_yticks(major_ticks)
axa[6].set_yticks(minor_ticks, minor=True)
axa[5].set_yticks(major_ticks)
axa[5].set_yticks(minor_ticks, minor=True)

patches = []
for label, color in colors.items():
    patches.append(mpatches.Patch(color=color, label=label)) 
axa[1].legend(handles=patches, fontsize=5, ncols = 2)

major_ticks = np.arange(0, 75000, 10000)
minor_ticks = np.arange(0, 75000, 5000)
for count, value in enumerate(axa):
    if count > 0:
        value.set_xticks(major_ticks)
        value.set_xticks(minor_ticks, minor=True)
        value.xaxis.set_tick_params(labelsize=8)
        value.yaxis.set_tick_params(labelsize=8)

# fig.savefig('lake_superior_metrics.png', bbox_inches='tight', dpi=300)
# fig.savefig('lake_superior_metrics.svg', bbox_inches='tight', dpi=300)


# %% Copy of Figure 1 for experimenting with axes for box & whisker
# Major ticks every 20, minor ticks every 5
plt.rcParams['text.usetex']=False
major_ticks = np.arange(0, 80, 20)
minor_ticks = np.arange(0, 80, 10)
colors = {'Artificial Depositional (e.g., jetty, groin fill)': 'indianred',
          'Artificial Moderate Quality Moderately Engineered':'darkred',
       'Sandy Beach / Dune (relict deposits, areas with no new deposition)':'moccasin',
       'Open Shore Wetlands':'darkgreen',
       'Composite Bluffs (sand content 20-50%)':'lightblue',
       'Composite Bluffs (sand content 0-20%)':'mediumspringgreen',
       'Marine / Leda Clay Bluffs':'violet',
       'Sandy Beach / Dune Complex':'yellow',
       'Pocket Beach':'goldenrod',
       'Gravel Beaches':'gray',
       'Boulder Beaches':'navy',
       'Coarse Beaches':'darkgray'}
shoreline_colors = list(dod_segments['shoreline'].map(colors))

elinewidth = 0.5
labelfontsize = 8
tickfontsize = 8
plt.close('all')
color1 = 'black'
fig, axa = plt.subplots(7, 2, figsize=(8.5, 11), 
                        height_ratios=[5, 3, 1, 1, 1, 1, 1], 
                        width_ratios=[10, 0.5])
# dod_segments.plot(color = 'black', ax = axa[0])
# cx.add_basemap(ax = axa[0], source=cx.providers.OpenTopoMap, crs=dod_segments.crs, zoom=12, attribution = False)
# cx.add_basemap(ax = axa[0], source = cx.providers.Esri.WorldImagery, crs=dod_segments.crs, alpha = 0.7, attribution = False)
colors_bars = axa[1, 0].bar(dod_segments['cumulative_length'], dod_segments['norm_net_vol'], 500,
           edgecolor = color1, linewidth=0.5, alpha =1, color = shoreline_colors, zorder=3)
axa[1, 0].errorbar(dod_segments['cumulative_length'], dod_segments['norm_net_vol'],
                yerr = dod_segments['norm_net_vol_err'], alpha=0.8,
                zorder=5, marker='', linestyle='', ecolor=color1, elinewidth=elinewidth)
axa[1, 1].violinplot(pd.to_numeric(dod_segments['norm_net_vol']))
axa[2, 0].bar(toe_filt['cumulative_length'], pd.to_numeric(toe_filt['t_rec_md']), 500, edgecolor = color1, linewidth=0.5, alpha = 1, color = 'lightgray', zorder=3)
axa[2, 0].errorbar(toe_filt['cumulative_length'], pd.to_numeric(toe_filt['t_rec_md']),
                yerr = 1 * pd.to_numeric(toe_filt['t_rec_std']), alpha=0.8,
                zorder=5, marker='', linestyle='', ecolor=color1, elinewidth=elinewidth)
axa[2, 1].violinplot(pd.to_numeric(toe_filt['t_rec_md']))
axa[3, 0].bar(crest_filt['cumulative_length'], pd.to_numeric(crest_filt['c_rec_md']), 500, edgecolor = color1, linewidth=0.5, alpha = 1, color = 'lightgray', zorder=3)
axa[3, 0].errorbar(crest_filt['cumulative_length'], pd.to_numeric(crest_filt['c_rec_md']),
                yerr = pd.to_numeric(crest_filt['c_rec_std']), alpha=0.8,
                zorder=5, marker='', linestyle='', ecolor=color1, elinewidth=elinewidth)
axa[3, 1].violinplot(pd.to_numeric(crest_filt['c_rec_md']))
axa[4, 0].bar(slp_filt['cumulative_length'], pd.to_numeric(slp_filt['slpchg_md']), 500, edgecolor = color1, linewidth=0.5, alpha = 1, color = 'lightgray', zorder=3)
axa[4, 0].errorbar(slp_filt['cumulative_length'], pd.to_numeric(slp_filt['slpchg_md']),
                yerr = pd.to_numeric(slp_filt['slpchg_std']), alpha=0.8,
                zorder=5, marker='', linestyle='', ecolor=color1, elinewidth=elinewidth)
axa[4, 1].violinplot(pd.to_numeric(slp_filt['slpchg_md']))
axa[5, 0].bar(slp_filt['cumulative_length'], pd.to_numeric(slp_filt['bh09_md']), 500, edgecolor = color1, linewidth=0.5, alpha = 1, color = 'lightgray', zorder=3)
axa[5, 0].errorbar(slp_filt['cumulative_length'], pd.to_numeric(slp_filt['bh09_md']),
                yerr = pd.to_numeric(slp_filt['bh09_std']), alpha=0.8,
                zorder=5, marker='', linestyle='', ecolor=color1, elinewidth=elinewidth)
axa[5, 1].violinplot(pd.to_numeric(slp_filt['bh09_md']))
axa[6, 0].bar(slp_filt['cumulative_length'], pd.to_numeric(slp_filt['slpd09_md']), 500, edgecolor = color1, linewidth=0.5, alpha = 1, color = 'lightgray', zorder=3)
axa[6, 0].errorbar(slp_filt['cumulative_length'], pd.to_numeric(slp_filt['slpd09_md']),
                yerr = pd.to_numeric(slp_filt['slpd09_md']), alpha=0.8,
                zorder=5, marker='', linestyle='', ecolor=color1, elinewidth=elinewidth)
axa[6, 1].violinplot(pd.to_numeric(slp_filt['slpd09_md']))


plt.yticks(fontsize=12,color=color1,weight='normal')
plt.xticks(fontsize=12,color=color1,weight='normal')
axa[0, 0].set_xticks([])
axa[0, 0].set_xticklabels([])
axa[1, 0].set_xticklabels([])
axa[2, 0].set_xticklabels([])
axa[3, 0].set_xticklabels([])
axa[4, 0].set_xticklabels([])
axa[5, 0].set_xticklabels([])
# axa[6].set_xticklabels([])

axa[0, 0].set_yticks([])
axa[1, 0].invert_yaxis()
axa[1, 1].invert_yaxis()
axa[1, 0].set_ylabel("Normalized\nvolume\nchange rate\n($\mathrm{m}^3 \mathrm{m}^{-1} \mathrm{a}^{-1}$)",fontsize=labelfontsize,color=color1, weight='normal', rotation=0, horizontalalignment='left', verticalalignment='top')
axa[2, 0].set_ylabel("Toe retreat\n(m)",fontsize=labelfontsize,color=color1, weight='normal', rotation=0, horizontalalignment='left', verticalalignment='top')
axa[3, 0].set_ylabel(r"Crest retreat""\n""(m)",fontsize=labelfontsize,color=color1, weight='normal', rotation = 0, horizontalalignment='left', verticalalignment='top')
axa[4, 0].set_ylabel(r"Slope increase ($^\circ$)",fontsize=labelfontsize,color=color1, weight='normal', rotation = 0, horizontalalignment='left', verticalalignment='top')
axa[5, 0].set_ylabel("Bluff height in 2009 (m)",fontsize=labelfontsize,color=color1, weight='normal', rotation = 0, horizontalalignment='left', verticalalignment='top')
axa[6, 0].set_ylabel(r"Bluff slope in 2009 ($^\circ$)",fontsize=labelfontsize,color=color1, weight='normal', rotation = 0, horizontalalignment='left', verticalalignment='top')
# axa[7].set_ylabel("Toe - crest retreat (meters)",fontsize=10,color=color1, weight='normal', rotation = 0)
axa[6, 0].set_xlabel("Alongshore distance (meters)",fontsize=labelfontsize, color=color1, weight='normal', rotation = 0, horizontalalignment='left', verticalalignment='top')
for count, value in enumerate(axa[:,0]):
    if (count > 1) & (count != 4):
        value.set_xlim(axa[1, 0].get_xlim())
        value.set_ylim(-5, axa[count, 0].get_ylim()[1])
    if count == 4:
            value.set_ylim(-5, axa[count, 0].get_ylim()[1])
for count, value in enumerate(axa[:,0]):
    if (count > 0) & (count < 5):
        value.xaxis.set_ticks_position('none') 
        value.yaxis.set_label_coords(0.01, 0.9)
        # value.xaxis.set_major_locator(MultipleLocator(10000))
        # value.xaxis.set_minor_locator(AutoMinorLocator(5000))
        value.yaxis.set_major_locator(MultipleLocator(10))
        value.yaxis.set_minor_locator(AutoMinorLocator(5))
        value.grid(which='major', color='#CCCCCC', linestyle='--')
        value.grid(which='minor', color='#CCCCCC', linestyle=':')
    if (count == 5):
        value.xaxis.set_ticks_position('none') 
        value.yaxis.set_label_coords(0.01, 0.9)
        # value.xaxis.set_major_locator(MultipleLocator(10000))
        # value.xaxis.set_minor_locator(AutoMinorLocator(5000))
        value.yaxis.set_major_locator(MultipleLocator(20))
        value.yaxis.set_minor_locator(AutoMinorLocator(10))
        value.grid(which='major', color='#CCCCCC', linestyle='--')
        value.grid(which='minor', color='#CCCCCC', linestyle=':')
    if (count == 6):
        value.yaxis.set_label_coords(0.01, 0.9)
        # value.xaxis.set_major_locator(MultipleLocator(10000))
        # value.xaxis.set_minor_locator(AutoMinorLocator(5000))
        value.yaxis.set_major_locator(MultipleLocator(20))
        value.yaxis.set_minor_locator(AutoMinorLocator(10))
        value.grid(which='major', color='#CCCCCC', linestyle='--')
        value.grid(which='minor', color='#CCCCCC', linestyle=':')

axa[6, 0].set_yticks(major_ticks)
axa[6, 0].set_yticks(minor_ticks, minor=True)
axa[5, 0].set_yticks(major_ticks)
axa[5, 0].set_yticks(minor_ticks, minor=True)

for count, value in enumerate(axa[:,1]):
    axa[count,1].set_ylim(axa[count, 0].get_ylim()[0], axa[count, 0].get_ylim()[1])
    value.axis('off')
patches = []
for label, color in colors.items():
    patches.append(mpatches.Patch(color=color, label=label)) 
axa[1,0].legend(handles=patches, fontsize=5, ncols = 2)

os.chdir(r'D:\CJR\LakeSuperior_DOD\buffer_polys_coastal')
os.chdir(os.path.abspath(r'..'))
plt.subplots_adjust(wspace=0.05)
fig.savefig('lake_superior_metrics_violin.png', bbox_inches='tight', dpi=300)
fig.savefig('lake_superior_metrics_violin.svg', bbox_inches='tight', dpi=300)
# %% Generate points for labeling longshore distance

# Clip and simplify hardened shoreline segments from transect generation
outdir = r"D:\CJR\LakeSuperior_BluffDelin"
os.chdir(outdir)
shoreline_clip_poly = gpd.read_file(r'BayfieldShorelineClip.shp')
shoreline_clip_poly = shoreline_clip_poly.to_crs('EPSG:32615')
shoreline = gpd.read_file(Path(homestr+r'.\LakeSuperior_BluffDelin\great_lakes_hardened_shorelines_cliff_delin_edit.shp'))
shoreline = shoreline.to_crs('EPSG:32615')
shoreline = gpd.clip(shoreline.geometry, shoreline_clip_poly.geometry)
shoreline_merge = shapely.ops.unary_union(shoreline.geometry)
shoreline_merge = gpd.GeoSeries(data=shoreline_merge, crs=shoreline.crs)

points = np.arange(0, 76000, 5000)

# Simplify and break clipped entire shoreline into segments for data aggregation, intersect with convex hull polygon to generate segmented polygons
line = shoreline_merge.geometry
line_merge = shapely.ops.linemerge(shoreline_merge[0])
points = []
for i in [0,1,2]:
    points_sing = [Point(xy) for xy in zip(line_merge.geoms[i].coords.xy[0], line_merge.geoms[i].coords.xy[1])]
    if i == 0:
        points_sing.reverse()
    points.extend(points_sing)
line_merge = LineString(points)

shoreline_lab = line_merge.interpolate(points)
shoreline_lab = gpd.GeoDataFrame(geometry=shoreline_lab, crs='EPSG:32615')
shoreline_lab['label'] = points
os.chdir(r'D:\CJR\LakeSuperior_DOD')
shoreline_lab.to_file(f'shoreline_distance_labels.shp')
# line_merge.to_file(f'shorline_merge_for_figure.shp')
# %% Some area of interest summary metrics
net_vol = np.sum(dod_segments.net_sum)
net_vol_low = np.sum(dod_segments.net_sum_lo)
net_vol_high = np.sum(dod_segments.net_sum_hi)
total_length = np.sum(dod_segments.length_m)
net_vol / total_length / 10.5

np.nanmean(pd.to_numeric(toe_filt['t_rec_mn']))
np.nanmedian(pd.to_numeric(toe_filt['t_rec_std']))
np.nanmean(pd.to_numeric(toe_filt['t_rec_md']))/10.5
np.nanmean(pd.to_numeric(toe_filt['t_rec_std']))/10.5
np.nanmedian(pd.to_numeric(toe_filt['c_rec_std']))/10.5
np.nanmedian(pd.to_numeric(toe_filt['c_rec_std']))
np.nanmedian(pd.to_numeric(toe_filt['c_rec_md']))/10.5
np.nanmean(pd.to_numeric(toe_filt['c_rec_mn']))
# %% Modifying wave information study statistics

os.chdir(r'D:\CJR\depth_of_closure')
wis_stations = pd.read_csv(r'greatlakes_cumulative_stations.csv')
wis_data = pd.read_csv(r'greatlakes_cumulative_wave_stats.csv')
wis_data2 = wis_stations.merge(wis_data, on='WIS_Station')
points_sing = [Point(xy) for xy in zip(wis_stations.X, wis_stations.Y)]
wis_data_gdf = gpd.GeoDataFrame(data=wis_data2, geometry = points_sing, crs='EPSG:4269')
wis_data_gdf = wis_data_gdf.to_crs('EPSG:32615')
# wis_data_gdf.to_file(r'great_lakes_waveinformationstudy.shp')

# %% Automatic vs manual bluff features
shoreline = gpd.read_file(Path(homestr+r'.\LakeSuperior_BluffDelin\great_lakes_hardened_shorelines_cliff_delin_edit.shp'))
shoreline = shoreline.to_crs('EPSG:32615')
shoreline = gpd.clip(shoreline, coastal_clip.geometry.buffer(100))
shoreline_types = shoreline['Shorelin_1'].unique()
shoreline_bluff = shoreline[shoreline['Shorelin_1'].isin([shoreline_types[1] , shoreline_types[7], shoreline_types[11]])]
shoreline_hard = shoreline[shoreline['Shorelin_1'].isin([shoreline_types[4] , shoreline_types[5]])]
shoreline_beach = shoreline[shoreline['Shorelin_1'].isin([shoreline_types[2] , shoreline_types[6]])]

transects_filt = transects[transects.geometry.intersects(coastal_clip.geometry[0])] 
transects_filt_mod = shapely.ops.unary_union(transects_filt.geometry)
os.chdir(r'D:\CJR\LakeSuperior_BluffDelin\2009')
orig_top = gpd.read_file(r'params5_106\2009_params5_top_points_orig.shp')
orig_top2 = orig_top[orig_top.geometry.intersects(transects_filt_mod)]

orig_base = gpd.read_file(r'params5_106\2009_params5_base_points_orig.shp')
orig_base2 = orig_base[orig_base.geometry.intersects(transects_filt_mod)]

mod_top = gpd.read_file(r'params5_106\2009_params5_top_points.shp')
mod_top2 = mod_top[mod_top.geometry.intersects(transects_filt_mod)]

mod_base = gpd.read_file(r'params5_106\2009_params5_base_points.shp')
mod_base2 = mod_base[mod_base.geometry.intersects(transects_filt_mod)]

np.nanmean(pd.to_numeric(dod_segments['t_rec_mn'][(pd.to_numeric(dod_segments['t_rec_mn'])>-100) & (dod_segments['shoreline']==shoreline_types[7])]))/10.5
np.sum(pd.to_numeric(dod_segments['net_sum'].iloc[0:7]))

np.nanmedian(pd.to_numeric(dod_segments['t_rec_md'][(pd.to_numeric(dod_segments['t_rec_md'])>-100) & (dod_segments['shoreline']==shoreline_types[7])]))/10.5
np.nanmean(pd.to_numeric(dod_segments['c_rec_md'][(pd.to_numeric(dod_segments['c_rec_md'])>-100) & (dod_segments['shoreline']==shoreline_types[7])]))/10.5
np.nanmean(pd.to_numeric(dod_segments['c_rec_md'][(pd.to_numeric(dod_segments['c_rec_md'])>-100) & (dod_segments['shoreline']==shoreline_types[1])]))/10.5

