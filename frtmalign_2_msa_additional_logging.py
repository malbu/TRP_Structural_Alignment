import argparse
import errno
import glob
import itertools
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os 
import pandas as pd
import pickle
import re
import requests
import seaborn as sns
import shutil
import subprocess
import sys
import tempfile
from Bio import AlignIO
from Bio.PDB.DSSP import DSSP, dssp_dict_from_pdb_file
from Bio.PDB.PDBParser import PDBParser
from Bio.Seq import Seq
from MDAnalysis.analysis import hole2
from mdahole2.analysis.hole import hole
import logging
import shutil
import time
from functools import wraps
#import MDAnalysis.analysis.hole2
#print(dir(MDAnalysis.analysis.hole2))
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
try:
  from pyali.mrgali import *
except:
  print("[ERROR]: pyali has not been installed.  To install, run `python setup.py install` from project directory.")

#import subprocess
#from unittest import mock


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a separate logger for timing information
timing_logger = logging.getLogger('timing')
timing_logger.setLevel(logging.DEBUG)
timing_handler = logging.StreamHandler()
timing_handler.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d - TIMING - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
timing_logger.addHandler(timing_handler)

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        timing_logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def paths_dic(locations='./paths.txt'):
    logging.info(f"Reading paths from {locations}")
    paths = {}
    with open(locations, 'r') as f:
        for line in f:
            if not line.strip().startswith('#'):
                fields = line.strip().split('\t')
                if len(fields) >= 2:
                    key = fields[0].strip()
                    value = fields[1].strip()
                    if key == 'work_dir':
                        paths['work_dir'] = os.path.abspath(value) + '/'
                    else:
                        paths[key] = value
                elif fields[0].strip() == 'work_dir':
                    print('Info: Setting working directory to current directory, ' + os.getcwd() + '/. Change paths.txt for an alternative location.')
                    paths['work_dir'] = os.getcwd() + '/'
                else:
                    raise SystemExit('Error: Path for ' + fields[0].strip() + ' not set in paths.txt.')

    try:
        os.mkdir(paths['work_dir'])
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        else:
            os.chdir(paths['work_dir'])
    
    # establish project's subdirectory structure
    paths['pdb_dir'] = paths['work_dir'] + '1_original_structures_OPM/'
    paths['clean_dir'] = paths['work_dir'] + '2_clean_structures/'
    paths['frtmalign_dir'] = paths['work_dir'] + '3_aligned_structures/'
    try:
        os.mkdir(paths['pdb_dir'])
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    try:
        os.mkdir(paths['clean_dir'])
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
                                       
    try:
        os.mkdir(paths['frtmalign_dir'])
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    if not os.path.exists(paths['structs_info']):
        raise SystemExit(paths['structs_info']+' path to .xml file specified in paths.txt does not exist')
    if not os.path.exists(paths['frtmalign']):
        raise SystemExit(paths['frtmalign']+' path to Fr-TM-Align executable specified in paths.txt does not exist')
    if not os.path.exists(paths['hole']):
        raise SystemExit(paths['hole']+' path to HOLE executable specified in paths.txt does not exist')
    if not os.path.exists(paths['vdw_radius_file']):
        raise SystemExit(paths['vdw_radius_file']+' path to Van der Waals radius file specified in paths.txt does not exist')

    for dir_path in [paths['work_dir'], paths['pdb_dir'], paths['clean_dir'], paths['frtmalign_dir']]:
        try:
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Created directory: {dir_path}")
        except Exception as e:
            logging.error(f"Failed to create directory {dir_path}: {str(e)}")
    
    return paths

def get_struct(pdbid, savedir, provdir):
    logging.info(f"Attempting to get structure for PDB ID: {pdbid}")
    # First tries to get structure from OPM
    url = 'https://opm-assets.storage.googleapis.com/pdb/' + pdbid + '.pdb'
    res = requests.get(url, allow_redirects=True)
    if not res.status_code == 200:
        # If structure not found in OPM, look for user-provided structure file
        print("Warning: Found no record in OPM for %s. Checking %s." % (pdbid, provdir))
        try:
            shutil.copyfile(provdir+pdbid+'.pdb', savedir+pdbid+ '.pdb')
        except:
            # If no user-provided structure file, try to get structure from PDB
            print("Warning: Found no provided structure file for %s in %s. Checking PDB." %(pdbid, provdir))
            pdb_url = 'https://files.rcsb.org/download/' + pdbid + '.pdb'
            pdb_res = requests.get(url, allow_redirects=True)
            if not pdb_res.status_code == 200:
                # If structure not found in PDB, print warning and skip structure
                print("Warning: found no record in OPM, %s, or PDB for %s, so it will be ignored." % (provdir, pdbid))
                return False
            else:
                open(savedir + pdbid + '.pdb', 'wb').write(pdb_res.content)
                return True
        return True
    else:
        open(savedir + pdbid + '.pdb', 'wb').write(res.content)
        return True

def xml_parser(xml_file):
    logging.info(f"Parsing XML file: {xml_file}")
    xml = open(xml_file, 'r')
    pdb_dic = {}
    for line in xml:
        if line.startswith('</Structure>'):
            pdbid =''
        if '<PDBID' in line:
            pdbid = line.split('"')[1]
            pdb_dic[pdbid] = {'id': '', 'tm_chains': [], 'subfamily': '','environment': '','method': '', 'ligand':''}
            pdb_dic[pdbid]['id'] = pdbid
            #print(pdbid)

        if '<ChainId' in line:
            chainid = line.split('"')[1]
            #print(chainid)
            resid_list = []
        if '<start' in line:
            start = line.split('"')[1]
        if '<end' in line:
            end = line.split('"')[1]
            resid_list.append(range(int(start), int(end)))
        if '</Chain>' in line:
            #print(pdbid, chainid, resid_list)
            pdb_dic[pdbid]['tm_chains'].append([chainid, resid_list])
            chainid = ''

        if '<Subfamily' in line:
            pdb_dic[pdbid]['subfamily'] = line.split('"')[1]
        if '<Method' in line:
            pdb_dic[pdbid]['method'] = line.split('"')[1]
        if '<Environment' in line:
            pdb_dic[pdbid]['environment'] = line.split('"')[1]
        if '<Ligand' in line:
            pdb_dic[pdbid]['ligand'] = line.split('"')[1]

    xml.close()
    
    # convert struct_info_dict into dataframe containing pdbid, tm_chains, subfamily, environment, method, and ligand
    all_info_list = []
    for pdbid, info_dict in pdb_dic.items():
        info_list = [info_dict['id'], info_dict['tm_chains'], info_dict['subfamily'], info_dict['environment'], info_dict['method'], info_dict['ligand']]
        all_info_list.append(info_list)
    pdb_df = pd.DataFrame(all_info_list, columns=['PDB ID', 'TM chains', 'Subfamily', 'Environment', 'Method', 'Ligand'])

    #print(pdb_dic)
    return pdb_dic, pdb_df


def from3to1_general(resname):
  f3t1 = {'ALA' : 'A',
          'ARG' : 'R',
          'ASN' : 'N',
          'ASP' : 'D',
          'CYS' : 'C',
          'GLN' : 'Q',
          'GLU' : 'E',
          'GLY' : 'G',
          'HIS' : 'H',
          'ILE' : 'I',
          'LEU' : 'L',
          'LYS' : 'K',
          'MET' : 'M',
          'PHE' : 'F',
          'PRO' : 'P',
          'SER' : 'S',
          'THR' : 'T',
          'TRP' : 'W',
          'TYR' : 'Y',
          'VAL' : 'V',
          'MSE' : 'M'}

  if resname in list(f3t1.keys()):
    return f3t1[resname]
  else:
    return '0'

def strip_tm_chains(wkdir,inputf,pdb_path,chains_data):
    """
    Extract only the specified chains atoms that are properly specified and enforce the user-specified chain order.
    Note that chains_data is a list of the form [['A',[range(2,120),range(240,300)]],['C']], where the ranges specify 
    the residue ranges of each chain that should be included in the final structure. If chains_data is empty, all 
    properly specified atoms in the PDB file will be included.
    Note that TER entries are ignored.
    Returns the number of residues in the final structure.
    """

    f=open(pdb_path,'r')
    altloc=' '
    flag=0
    for line in f:
        if line.startswith("ATOM") and line[12:16].strip()=='CA' and line[16:17]!=' ' and (float(line[54:60])>0.5 or flag==0):
            altloc=line[16:17]
            flag=1
    f.seek(0)
  
    if len(chains_data)>0:
        chains = [i[0] for i in chains_data] # chains = ['A','C'], for example
    else:
        chains = ""

    o=open(wkdir+inputf+"_clean.pdb","w+")
    num=0
    resid_count = 0
    LINELEM="{:76s}{:>2s}\n"
    for ind, chain in enumerate(chains):   
        atomnames=[]
        old_resid='-88'
        f.seek(0)
        for line in f:
            if line.startswith("ATOM") and line[21:22]==chain and from3to1_general(line[17:20].strip())!=0:
                if len(chains_data[ind])>1 and not any([True for k in chains_data[ind][1] if int(line[22:26]) in k]): 
                    continue
                
                if old_resid!='-88' and line[22:26]==old_resid and line[12:16] in atomnames or (line[16:17]!=altloc and line[16:17]!=' '): #sort out disordered atoms
                    continue
                elif (old_resid=='-88' or line[22:26]!=old_resid):
                    old_resid=line[22:26]
                    resid_count +=1
                    atomnames=[]
                    #print(line)
                atomnames.append(line[12:16])

                if line[76:78].strip()!='': #ensure that the element symbol is included
                    o.write(line[0:22] + "{:>4s}".format(str(resid_count)) + line[26:])
                    num+=1
                else:
                    atom=line[12:16].strip()
                    elem=atom[0]
                    o.write(LINELEM.format(line[0:22] + "{:>4s}".format(str(resid_count)) + line[26:76],elem))
                    num+=1
            if line.startswith("HETATM") and line[17:20]=='MSE' and line[21:22]==chain:
          
                if len(chains_data[ind])>1 and not any([True for k in chains_data[ind][1] if int(line[22:26]) in k]): # check whether the residue is in the specified atom ranges for the chain
                    continue
             
                if old_resid!='-88' and line[22:26]==old_resid and line[12:16] in atomnames or (line[16:17]!=altloc and line[16:17]!=' '):
                    continue
                elif (old_resid=='-88' or line[22:26]!=old_resid):
                    old_resid=line[22:26]
                    resid_count +=1
                    atomnames=[]
                atomnames.append(line[12:16])

                if line[76:78].strip()!='':
                    o.write("ATOM  "+line[6:22] + "{:>4s}".format(str(resid_count)) + line[26:])
                    num+=1
                else:
                    atom=line[12:16].strip()
                    elem=atom[0]
                    o.write("ATOM  "+ line[6:22]+ "{:>4s}".format(str(resid_count)) + line[26:76] +" "+elem+"\n")
                    num+=1
     
    o.write("END\n")
    f.close()
    o.close()
    return resid_count
    
@log_execution_time
def batch_frtmalign(in_file_path, out_dir, frtmalign_path, original_dir, clean_dir):
    logging.info("Starting batch_frtmalign")
    arg_list = []
    total_alignments = 0
    start_time = time.time()
    with tempfile.TemporaryDirectory() as tmpdirname:
        logging.info(f"Created temporary directory: {tmpdirname}")
        tmpdirnamefull = tmpdirname+"/"
        for pdb_file in glob.glob(in_file_path + "*pdb"):
            file_name = os.path.split(pdb_file)
            shutil.copy2(pdb_file, tmpdirnamefull+file_name[1])
        for station_file in glob.glob(tmpdirnamefull + "*.pdb"):
            station_name = station_file[-14:-10]
            out_file_path = "%sstationary_%s/" %(tmpdirnamefull, station_name)
            outfilename = out_file_path + station_name + ".pdb"
            os.makedirs(os.path.dirname(outfilename), exist_ok=True)
            for mobile_file in glob.glob(tmpdirnamefull + "*.pdb"):
                mobile_name = mobile_file[-14:-10]
                arg_list.append((mobile_file, station_file, out_file_path, mobile_name, station_name, frtmalign_path, original_dir, clean_dir))
                total_alignments += 1
        
        timing_logger.info(f"Total alignments to be performed: {total_alignments}")
        
        n_cpus = mp.cpu_count()
        timing_logger.info(f"Using {n_cpus} CPUs for parallel processing")
        pool = mp.Pool(n_cpus)
        results = [pool.apply(single_frtmalign, args=arg_tup) for arg_tup in arg_list]
        
        for root, dirs, files in os.walk(tmpdirname, topdown=False):
            dir_name = os.path.split(root)[1]
            if dir_name.startswith('stationary'):
                try:
                    os.mkdir(os.path.join(out_dir, dir_name))
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise
                    pass
                for item in files:
                    shutil.copy2(os.path.join(root, item), os.path.join(out_dir, dir_name, item))
        shutil.copytree(tmpdirname, out_dir + "/temp_frtmalign", dirs_exist_ok=True)
        logging.info(f"Copied temporary directory to {out_dir}/temp_frtmalign for inspection")
    
    end_time = time.time()
    total_time = end_time - start_time
    timing_logger.info(f"Batch Fr-TM-Align completed. Total time: {total_time:.2f} seconds")
    timing_logger.info(f"Average time per alignment: {total_time / total_alignments:.2f} seconds")


@log_execution_time
def single_frtmalign(mobile_file, station_file, out_file_path, mobile_name, station_name, frtmalign_path, original_dir, clean_dir):
    start_time = time.time()
    timing_logger.info(f"Running Fr-TM-Align: mobile={mobile_name}, stationary={station_name}")
    print('m: %s, s: %s' %(mobile_name, station_name))
    bash_frtmalign = "%s %s %s -o %s%s_%s.sup -m 1" %(frtmalign_path, mobile_file, station_file, out_file_path, mobile_name, station_name)
    
    # Use a context manager to ensure the file is properly closed
    with open("%s%s_%s.frtxt" %(out_file_path, mobile_name, station_name), "w+") as fileout:
        p = subprocess.Popen(bash_frtmalign.split(), stdout=fileout, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        
        if p.returncode != 0:
            logging.error(f"Fr-TM-Align failed: {stderr.decode()}")
        else:
            logging.info("Fr-TM-Align completed successfully")

    curr_dir = os.getcwd() + '/'
    if os.path.exists(curr_dir + 'trf.mat'):
        shutil.copy2(curr_dir + 'trf.mat', out_file_path + mobile_name + '_' + station_name + '.mat')
    else:
        raise SystemExit(curr_dir + 'trf.mat does not exist.')
    
    transform_pdb("%s%s.pdb" %(original_dir, mobile_name), "%s%s_%s.mat" %(out_file_path, mobile_name, station_name), mobile_name, station_name, out_file_path, "_full_align.pdb")
    transform_pdb("%s%s_clean.pdb" %(clean_dir, mobile_name), "%s%s_%s.mat" %(out_file_path, mobile_name, station_name), mobile_name, station_name, out_file_path, "_tmem_align.pdb")
    
    end_time = time.time()
    timing_logger.info(f"Single Fr-TM-Align (mobile={mobile_name}, stationary={station_name}) completed in {end_time - start_time:.2f} seconds")

@log_execution_time
def transform_pdb(pdbin, transformin, mobile_name, station_name, file_path, suffix):
  timing_logger.info(f"Starting transform_pdb for {mobile_name} and {station_name}")
  # make x, y, and z coordinate lists
  x_coord = list()
  y_coord = list()
  z_coord = list()
  pdb_file = open(pdbin, "r")
  for line in pdb_file:
    # read in coordinates from ATOM lines
    if line.startswith("ATOM") or line.startswith("HETATM"):
      x_coord.append(float(line[30:38].strip()))
      y_coord.append(float(line[38:46].strip()))
      z_coord.append(float(line[46:54].strip()))
  pdb_file.close()
  # convert x, y, and z coordinate lists into single numpy matrix
  xyz_coord = np.column_stack((x_coord, y_coord, z_coord))
  # read translation and rotation matrices into separate numpy matrices
  n_lines = 0
  transl = list()
  rot_x = list()
  rot_y = list()
  rot_z = list()
  transform_file = open(transformin, "r")
  for line in transform_file:
    if n_lines >= 2:
      # split data lines based on whitespace
      split_line = line.split()
      transl.append(float(split_line[1]))
      rot_x.append(float(split_line[2]))
      rot_y.append(float(split_line[3]))
      rot_z.append(float(split_line[4]))
    n_lines += 1
  transform_file.close()
  tra_mat = np.asarray(transl)
  rot_mat = np.column_stack((rot_x, rot_y, rot_z))
  # apply translation and rotation matrices to original coordinates
  xyz_coord_rot = np.matmul(xyz_coord, np.transpose(rot_mat))  
  xyz_coord_rot_tra = xyz_coord_rot + np.transpose(tra_mat)
  
  # make new pdb file with updated coordinates
  n_lines = 0
  pdb_file = open(pdbin, "r")
  outfilename = file_path + mobile_name + "_" + station_name +suffix
  os.makedirs(os.path.dirname(outfilename), exist_ok=True)
  fileout = open(outfilename, "a")
  for line in pdb_file:
    if line.startswith("ATOM") or line.startswith("HETATM"):
      new_x = '{:8.3f}'.format(xyz_coord_rot_tra[n_lines,0])
      new_y = '{:8.3f}'.format(xyz_coord_rot_tra[n_lines,1])
      new_z = '{:8.3f}'.format(xyz_coord_rot_tra[n_lines,2])
      line_new_coor = line[:30]+new_x+new_y+new_z+line[54:]
      fileout.write(line_new_coor)
      n_lines += 1
    else:
      fileout.write(line)
  fileout.close()
  pdb_file.close()
  timing_logger.info("Completed transform_pdb")
    
@log_execution_time
def frtmalign_2_list(directory_in):
  timing_logger.info(f"Starting frtmalign_2_list for {directory_in}")
  all_list = list()
  frtmalign_text_files = glob.glob(directory_in + "/**/*.frtxt")
  for filename in frtmalign_text_files:
    #print(filename)
    file_data = open(filename, "r")
    raw_aln_file = open(os.path.splitext(filename)[0] + ".aln", "w")
    fasta_file = open(os.path.splitext(filename)[0] + ".fasta", "w")
    chain = 1
    for line in file_data:
      if line.startswith("Chain 1"):
        # extract chain name and length
        mobile_name = line.split(":")[1][:4]
        mobile_length = line.split("=")[1][:4].strip()
      elif line.startswith("Chain 2"):
        # extract chain name and length
        station_name = line.split(":")[1][:4]
        station_length = line.split("=")[1][:4].strip()
      elif line.startswith("Aligned"):
        # extract aligned length, RMSD, TM-score, and sequence identity
        values = line.split("=")
        align_length = values[1][:4].strip()
        RMSD = values[2][:6].strip()
        TM_score = values[3][:7].strip()
        seq_id = values[4][:5].strip()
      elif (line.strip()) and (not "**" in line) and (not "(" in line):#line doesn't contain ** or ) and line is not empty:
        # print to file
        raw_aln_file.write(line)
        if ":" in line:
          chain += 1
        elif chain == 1:
          fasta_file.write(">" + mobile_name + "\n" + line)
        elif chain == 2:
          fasta_file.write(">" + station_name + "\n" + line)
    data_list = [mobile_name, mobile_length, station_name, station_length, align_length, RMSD, TM_score, seq_id]
    all_list.append(data_list)
    file_data.close()
    raw_aln_file.close()
    fasta_file.close()
  #print(all_list)
  timing_logger.info("Completed frtmalign_2_list")
  return all_list
@log_execution_time
def frtmalign_2_tables(frtmalign_list, names_df, output_folder):
    timing_logger.info("Starting frtmalign_2_tables")
    names_list = names_df['PDB ID'].values.tolist()  # List of PDB IDs
    names_index_df = names_df.set_index('PDB ID')

    # Initialize DataFrames
    align_length = pd.DataFrame(index=names_list, columns=names_list)
    RMSD = pd.DataFrame(index=names_list, columns=names_list)
    TM_score = pd.DataFrame(index=names_list, columns=names_list)
    seq_id = pd.DataFrame(index=names_list, columns=names_list)

    # Populate DataFrames
    for data_list in frtmalign_list:
        align_length.loc[data_list[0], data_list[2]] = data_list[4]
        RMSD.loc[data_list[0], data_list[2]] = data_list[5]
        TM_score.loc[data_list[0], data_list[2]] = data_list[6]
        seq_id.loc[data_list[0], data_list[2]] = data_list[7]

    # Save DataFrames to CSV
    frtmalign_df = pd.DataFrame(frtmalign_list, columns=[
        "mobile", "mobile_length", "stationary", "stationary_length", "aligned_length", "RMSD", "TM-score", "sequence_identity"])
    frtmalign_df.to_csv(output_folder + "frtmalign_output.csv")
    align_length.to_csv(output_folder + "aligned_length.csv")
    RMSD.to_csv(output_folder + "RMSD.csv")
    TM_score.to_csv(output_folder + "TM_score.csv")
    seq_id.to_csv(output_folder + "sequence_identity.csv")

    # Handle NaN values in TM_score DataFrame
    TM_score = TM_score.astype('float')
    TM_score.replace('None', np.nan, inplace=True)  # Replace any 'None' strings with NaN
    TM_score.fillna(0, inplace=True)  # Fill NaNs with zeros (or another appropriate value)

    # Convert TM-score to pseudo-distance
    TM_to_dist = lambda x: 1 - x
    distance = TM_score.applymap(TM_to_dist)

    # Check for finite values
    if not np.isfinite(distance.values).all():
        raise ValueError("Distance matrix contains non-finite values after filling NaNs.")

    # Perform hierarchical clustering
    sns.set(font_scale=1.5, rc={'figure.figsize': (11, 8.5)})
    cluster = linkage(distance.T, method='single', metric='euclidean', optimal_ordering=True)
    cluster_order = distance.index[leaves_list(cluster)]
    distance = distance.reindex(index=cluster_order)

    # Use categories to colorcode clustermap
    group_df = names_index_df.iloc[:,1:]

    # For clustermap  
    group_colors = None
    col_num = 0
    palettes = ['Greys', 'Greens', 'Blues', 'Purples', 'Oranges', 'Reds']
    colordict = {'TRPA':'#a6cee3', 'TRPV':'#1f78b4', 'P2X':'#b2df8a', 'ELIC':'#33a02c', 'TRPC':'#fb9a99', 'TRPM':'#e31a1c', 'TRP':'#fdbf6f', 'TRPN':'#ff7f00', 'TRPP':'#cab2d6', 'TRPML':'#6a3d9a', 'Kv':'#ffff99'}
    patch_list = []
    for column in group_df:
        column_values = group_df[column]
        if col_num == 0:
            colorlist = [colordict[x] for x in column_values.unique()]
            column_pal = sns.color_palette(colorlist, n_colors=column_values.unique().size)
        else:
            column_pal = sns.color_palette(palettes[col_num-1], n_colors=column_values.unique().size)
        column_lut = dict(zip(map(str, column_values.unique()), column_pal))
        for key, item in column_lut.items():
            patch = mpatches.Patch(color=item, label=key)
            patch_list.append(patch)
        column_colors = pd.Series(column_values, index=group_df.index).map(column_lut)
        col_num += 1
        if group_colors is None:
            group_colors = pd.DataFrame(column_colors)
        else:
            group_colors = group_colors.join(pd.DataFrame(column_colors))
    clust_out = sns.clustermap(
        distance,
        row_cluster=False,
        col_linkage=cluster,
        row_colors=group_colors,
        col_colors=group_colors,
        vmin=0,
        vmax=1,
        cmap='gnuplot2_r',
        center=0.4,
        cbar_kws={'label': 'pseudo-distance'}
    )
    plt.savefig(output_folder + "clustermap.png", dpi=200)
    plt.savefig(output_folder + "clustermap.pdf", dpi=200)
    plt.close()

    # Save the clustering order
    names_df = pd.DataFrame(names_list)
    col_order = names_df.reindex(clust_out.dendrogram_col.reordered_ind)
    col_order.to_csv(output_folder + 'stationary_clustering_order.csv', header=['PDB ID'], index=False)
    timing_logger.info("Completed frtmalign_2_tables")




def raise_seq(infile, outfile, seqn):
  aligns = []
  name = ""
  seq = ""
  if '>' not in seqn:
    seqn = '>' + seqn
  f = open(infile, 'r')
  for line in f:
    if line.startswith(">"):
      if seq != "":
        aligns.append((name, seq))
      name = line.strip()
      seq = ""
    elif not line.startswith("#"):
      seq = seq + line
  aligns.append((name, seq))
  f.close()
  
  index = [x for x, y in enumerate(aligns) if seqn in y[0]] # locate the top sequence
  if len(index)==0:
    raise SystemExit(seqn + " cannot be located in " + infile)
  else:
    index = index[0]
  out = open(outfile, 'w')
  out.write(aligns[index][0] + '\n' + aligns[index][1].strip())
  for a in aligns:
    if a[0]!=seqn:
      out.write('\n' + a[0] + '\n' + a[1].strip())
  out.close()

def align_merger(file_list, outname, width, reference_seq):
    if not file_list:
      raise ValueError("No alignment files provided to align_merger.")
    refs, alis, alis_names = [], [], []
    for f in file_list:
        if reference_seq != '':
            raise_seq(f, f + '.tmp', reference_seq)
            alignment = open(f + '.tmp', 'r')
        else:
            alignment = open(f, 'r')
        flag = 0
        sequence1 = ""
        sequence2 = ""
        alis_element = []
        for line in alignment:
            if line.startswith(">") and flag == 2:
                alis_element.append(sequence2)
                alis_names.append(name2)
                sequence2 = ""
                name2 = line.strip()
            if line.startswith(">") and flag == 1:
                flag = 2
                alis_element.append(sequence1)
                name2 = line.strip()
            if line.startswith(">") and flag == 0:
                flag = 1
                name1 = line.strip()
            if not line.startswith(">") and flag == 1:
                sequence1 = sequence1 + line.strip().upper()
            if not line.startswith(">") and flag == 2:
                sequence2 = sequence2 + line.strip().upper()
        alignment.close()
        if reference_seq != '':
            os.remove(f + '.tmp')
        alis_element.append(sequence2)
        alis.append(alis_element)
        alis_names.append(name2)

    refs = [''.join([s for s in seqs[0] if s != '-']) for seqs in alis]
    if refs.count(refs[0]) != len(refs):
        print("The reference sequences in all the provided alignments are not identical.")
        for i, r in enumerate(refs[1:]):
            for j, s in enumerate(refs[0]):
                if s!=refs[i+1][j]:
                    print(file_list[0] +": (" + str(j) + "," + s + "), " + file_list[i+1] + ": " + r[j])
        raise SystemExit("References need to be the same to proceed.")

    a = Alignment.from_reference(refs)
    for i in range(len(alis)):
        a.merge(i, alis[i])

    flds = str(a).split('\n')

    aligned_list = []
    out = open(outname, 'w')
    for i, ln in enumerate(flds):
        if i == 0:
            s = ln[ln.index(':') + 2:]
            out.write(name1 + '\n')
            aligned_list.append((name1, s))
            while len(s) > 0:
                out.write(s[:width] + '\n')
                s = s[width:]
        if i >= len(refs):
            s = ln[ln.index(':') + 2:]
            out.write(alis_names[i - len(refs)] + '\n')
            aligned_list.append((alis_names[i - len(refs)], s))
            while len(s) > 0:
                out.write(s[:width] + '\n')
                s = s[width:]
    out.close()
    return aligned_list
    
@log_execution_time
def batch_align_merger(input_directory, order_file):
    timing_logger.info(f"Starting batch_align_merger with input directory: {input_directory}")
    order_list = pd.read_csv(order_file, header=0, usecols=['PDB ID'])
    order_dict = {k: v for v, k in order_list.itertuples()}
    
    for dirname in os.listdir(input_directory):
        sta_pdbid = dirname[-4:]
        dir_path = os.path.join(input_directory, dirname)
        if os.path.isdir(dir_path):
            start_time = time.time()
            filenames = glob.glob(os.path.join(dir_path, "*.fasta"))
            sorted_filenames = [''] * len(order_dict)
            
            for filename in filenames:
                # Extract mobile PDB ID from filename
                base_filename = os.path.basename(filename)
                parts = base_filename.split('_')
                if len(parts) >= 2:
                    mob_pdbid = parts[0]
                    if sta_pdbid != mob_pdbid and mob_pdbid in order_dict:
                        sorted_filenames[order_dict[mob_pdbid]] = filename
            
            ordered_filenames = [x for x in sorted_filenames if x]
            if not ordered_filenames:
                timing_logger.warning(f"No alignment files found for stationary PDB ID {sta_pdbid}.")
                continue
            
            outname = os.path.join(input_directory, f"{sta_pdbid}_full.ali")
            width = 72
            ref_seq = sta_pdbid
            alignment = align_merger(ordered_filenames, outname, width, ref_seq)
            
            # Process the multiple sequence alignment to remove gaps
            test = list(zip(*alignment))
            test_seq = [list(str) for str in test[1]]
            full_alignment = pd.DataFrame(test_seq, index=test[0]).transpose()
            mask = full_alignment[f">{sta_pdbid}"] != '-'
            nogap_alignment = full_alignment[mask]
            with open(os.path.join(input_directory, f"{sta_pdbid}_nogap.ali"), 'w') as outfile:
                for column in nogap_alignment:
                    outfile.write(column + '\n' + ''.join(nogap_alignment[column]) + '\n')
            end_time = time.time()
            timing_logger.info(f"Processed {sta_pdbid} in {end_time - start_time:.2f} seconds")
    timing_logger.info("Completed batch_align_merger")

def from_name_to_vdwr(pdb_atom_name):
  #from simple radius file in hole2, with extra 0.10A added as buffer
  fntv = {'C' : 1.95,
    'O' : 1.75,
    'S' : 2.10,
    'N' : 1.85,
    'H' : 1.10,
    'P' : 2.20}
  if pdb_atom_name[0] in list(fntv.keys()):
    return fntv[pdb_atom_name[0]]
  else:
    return 0.00

def calc_dist(coord1, coord2):
  return np.sqrt(((coord1[0]-coord2[0])**2)+((coord1[1]-coord2[1])**2)+((coord1[2]-coord2[2])**2))

@log_execution_time
def batch_hole(directory_in, category_df, hole_path, ref_struct, vdw_file, pore_point):
  timing_logger.info("Starting batch_hole")
  input_df = category_df.set_index('PDB ID')
  arg_list = []
  for filename in glob.glob(directory_in+"stationary_%s/*ls" %(ref_struct)):
    short_filename = filename[-24:-15]
    pdb_id = short_filename[0:4]
    out_dir = os.path.split(filename)[0] # places the output files into the same directory as the original coordinate file
    arg_list.append((filename, short_filename, pdb_id, out_dir, hole_path, input_df))
    single_hole(filename, short_filename, pdb_id, out_dir, hole_path, input_df, vdw_file, pore_point)
  timing_logger.info("Completed batch_hole")
    #print(out_dir+"/"+short_filename+"_hole_out.txt", out_dir+"/"+short_filename+"_hole.pdb")
  #n_cpus = mp.cpu_count()
  #pool = mp.Pool(n_cpus)
  #results = [pool.apply(single_hole, args=arg_tup) for arg_tup in arg_list]
   
def single_hole(filename, short_filename, pdb_id, out_dir, hole_path, input_df, vdw_file, pore_point):
    pore_point_str = ' '.join(map(str, pore_point))
    
    H = hole(pdbfile=filename, 
             outfile=out_dir+"/"+short_filename+"_hole_out.txt",
             sphpdb_file=out_dir+"/"+short_filename+"_hole.pdb", 
             cpoint=pore_point_str,  
             ignore_residues=['SOL', 'WAT', 'TIP', 'HOH', 'K  ', 'NA ', 'CL ', 'CA ', 'MG ', 'GD ', 'DUM', 'TRS'], 
             vdwradii_file=vdw_file, 
             executable=hole_path)
    
    # # Check if H is a dictionary (which it seems to be based on the error)
    # if isinstance(H, dict):
    #     # If it's a dictionary, we need to handle it differently
    #     # You might need to adjust this part based on what's actually in the dictionary
    #     profiles = H.get('profiles', {})
    #     if profiles:
    #         profile = list(profiles.values())[0]
    #         x = profile.radius
    #         y = profile.rxncoord
    #     else:
    #         logging.error(f"No profiles found in HOLE output for {pdb_id}")
    #         return
    # else:
    #     # If it's not a dictionary, assume it's the expected HOLE object
    #      H.run()
    #      H.collect()
    #     # If it's not a dictionary, assume it's the expected HOLE object
    #     if hasattr(H, 'results') and len(H.results.profiles) > 0:
    #         profile = H.results.profiles[0]
    #         if 'radius' in profile.dtype.names and 'rxn_coord' in profile.dtype.names:
    #             x = profile['radius']
    #             y = profile['rxn_coord']
    #         else:
    #             logging.error(f"Expected 'radius' and 'rxn_coord' not found in profile for {pdb_id}. Available fields: {profile.dtype.names}")
    #             return
    #     else:
    #         x = list(H.profiles.items())[0][1].radius
    #         y = list(H.profiles.items())[0][1].rxncoord
    #         logging.error(f"No profiles found in HOLE output for {pdb_id}")
    #         return

    H.run()
    H.collect()

    print(short_filename)
        # save HOLE profile plot
    x = list(H.profiles.items())[0][1].radius
    y = list(H.profiles.items())[0][1].rxncoord

    # Save HOLE profile plot
    fig, ax = plt.subplots()
    ax.axvline(4.1, linestyle='--', color='silver', label='') #hydrated Ca
    ax.axvline(3.6, linestyle=':', color='silver', label='') #hydrated Na
    ax.axvline(1.0, linestyle='-', color='silver', label='') #dehydrated Ca/Na
    plt.plot(x,y)
    plt.title(pdb_id)
    plt.xlabel('Radius (\u00C5)')
    plt.ylabel('Pore coordinate (\u00C5')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(0, 10)
    plt.ylim(-40,30)
    fig.savefig(out_dir+"/"+pdb_id+"_hole_plot.png", bbox_inches="tight")
    fig.savefig(out_dir+"/"+pdb_id+"_hole_plot.pdf", bbox_inches="tight")
    plt.close(fig)

    #Only perform further analysis on structures with four chains 
    if len(input_df.loc[pdb_id, 'TM chains'])!=4: 
      print(pdb_id+' does not have four chains, and was not included in multiple sequence alignment annotations of HOLE data.')
      return
    
    chain1 = input_df.loc[pdb_id,'TM chains'][0][0]
    chain2 = input_df.loc[pdb_id,'TM chains'][1][0]
    chain3 = input_df.loc[pdb_id,'TM chains'][2][0]
    chain4 = input_df.loc[pdb_id,'TM chains'][3][0]
    range1 = input_df.loc[pdb_id,'TM chains'][0][1]
    range2 = input_df.loc[pdb_id,'TM chains'][1][1]
    range3 = input_df.loc[pdb_id,'TM chains'][2][1]
    range4 = input_df.loc[pdb_id,'TM chains'][3][1]
    # Only perform further analysis on structures where ranges for all four chains are the same
    range2_same = range1 == range2
    range3_same = range1 == range3
    range4_same = range1 == range4
    if not all((range2_same, range3_same, range4_same)):
      print(pdb_id+' does not have the same residue range for all chains, and was not included in multiple sequence alignment annotations of HOLE data.')
      return

    pdb = PDBParser(QUIET=True)
    structure = pdb.get_structure(pdb_id, filename)
    hole_pdb = PDBParser(QUIET=True)
    hole_id = short_filename+'_hole'
    hole_struct = hole_pdb.get_structure(hole_id, out_dir+'/'+hole_id+'.pdb')

    hole_sph_list = []
    for atom in hole_struct.get_atoms():
      coord = atom.get_coord()
      radius = atom.get_bfactor()
      atom_list = [coord[0], coord[1], coord[2], radius]
      hole_sph_list.append(atom_list)
    hole_sph_df = pd.DataFrame(hole_sph_list, columns=['x','y','z','radius'])
    #print(hole_sph_df)

    residue_level_radius = {}
    for residue1 in structure[0][chain1]:
      # find matching residues
      try:
        residue2 = structure[0][chain2][residue1.get_full_id()[3]]
        residue3 = structure[0][chain3][residue1.get_full_id()[3]]
        residue4 = structure[0][chain4][residue1.get_full_id()[3]]
      except:
        #print("for "+filename+", "+residue1.get_resname()+" does not exist in other chain(s)")
        residue_level_radius[residue1.get_full_id()]  = [residue1.get_resname(), 'NaN']
        continue

      res2_same = residue1.get_resname()==residue2.get_resname()
      res3_same = residue1.get_resname()==residue3.get_resname()
      res4_same = residue1.get_resname()==residue4.get_resname()
      #print(residue1, residue2, residue3, residue4, res2_same, res3_same, res4_same)
      if all((res2_same, res3_same, res4_same)):
        atom_dict = {}
        for atom1 in residue1:
          try:
            atom2 = structure[0][chain2][residue1.get_full_id()[3]][atom1.get_name()]
            atom3 = structure[0][chain3][residue1.get_full_id()[3]][atom1.get_name()]
            atom4 = structure[0][chain4][residue1.get_full_id()[3]][atom1.get_name()]
          except:
            #print("for "+filename+", "+residue1.get_resname()+" "+atom1.get_name()+" does not exist in other chain(s)")
            continue
          coord1 = atom1.get_coord()
          coord2 = atom2.get_coord()
          coord3 = atom3.get_coord()
          coord4 = atom4.get_coord()
          name1 = atom1.get_name()
          name2 = atom2.get_name()
          name3 = atom3.get_name()
          name4 = atom4.get_name()
          # calculate distance between coord1 and the other three
          dist12 = calc_dist(coord1, coord2)
          dist13 = calc_dist(coord1, coord3)
          dist14 = calc_dist(coord1, coord4)
          # find maximum of dist12, dist13, and dist14, which represents the hypotenuse across the pore
          radius_distance = (max([dist12, dist13, dist14])/2)-from_name_to_vdwr(name1)
          avg_x = (coord1[0]+coord2[0]+coord3[0]+coord4[0])/4
          avg_y = (coord1[1]+coord2[1]+coord3[1]+coord4[1])/4
          avg_z = (coord1[2]+coord2[2]+coord3[2]+coord4[2])/4
          atom_dict[name1] = (avg_x, avg_y, avg_z, radius_distance)
          #print(name1, avg_x, avg_y, avg_z)
        if len(atom_dict) == 0:
          residue_level_radius[residue1.get_full_id()] = [residue1.get_resname(), 'NaN']
          continue
        contribute_to_pore = {}
        hole_sph_res = []
        for key, tup in atom_dict.items():
          index = abs(hole_sph_df['z'] - tup[2]).idxmin()
          #print('residue: %s, sphere: %s, struct: %s' %(residue1.get_full_id(), hole_sph_df.iloc[index, 3], tup[3]))
          if float(hole_sph_df.iloc[index, 3]) >= float(tup[3]):
            # add to list of atoms in residue where they are close enough to contribute to the pore diameter
            #print('%s contributes to pore' %(residue1.get_resname()))
            contribute_to_pore[key]=tup
            hole_sph_res.append(hole_sph_df.iloc[index].tolist())
        #print(hole_sph_res)
        # check whether dictionary is empty (if it is, no atoms in this residue contribute to the pore, so radius=NaN for this amino acid)
        if len(contribute_to_pore) == 0:
          residue_level_radius[residue1.get_full_id()]  = [residue1.get_resname(), 'NaN']
          #print(residue1.get_resname(), 'NaN')
        # else, convert dictionary into a dataframe, find min and max z value, then find sphere located between those z values that has the smallest radius
        else:
          # for every atom in avg_coords, find sphere closest to average coordinate point, and use compare its radius to the radius distance
          avg_coords = pd.DataFrame.from_dict(contribute_to_pore)
          #print(avg_coords)
          min_z = avg_coords.min(axis=1)[2]
          max_z = avg_coords.max(axis=1)[2]
          #print('min: %s, max: %s'%(min_z, max_z))
          # check which hole spheres are located between min and max z coordinates, and find one with smallest radius
          pore_level = hole_sph_df.loc[(hole_sph_df['z'] <= max_z) & (hole_sph_df['z'] >= min_z)]
          hole_sph_res_df = pd.DataFrame(hole_sph_res, columns=['x','y','z','radius'])
          pore_level_radius = pd.concat([pore_level, hole_sph_res_df])
          #print(pore_level_radius)
          residue_level_radius[residue1.get_full_id()] = [residue1.get_resname(), pore_level_radius.radius.min()]
          #print(residue1.get_resname(), pore_level.radius.min())
          
      else:
        #print("for "+filename+", "+residue1.get_resname()+" does not match residue(s) in other chain(s)")
        residue_level_radius[residue1.get_full_id()] = [residue1.get_resname(), 'NaN']

    residue_level_radius_df = pd.DataFrame.from_dict(residue_level_radius, orient='index', columns=['residue', 'radius'])
    #print(residue_level_radius_df)
    #print(residue_level_radius_df['radius'].notna())

    amino_acid_list = ['ALA','ARG','ASN','ASP','CYS','GLU','GLN','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','MSE']
    aa_level_radius_df = residue_level_radius_df.loc[residue_level_radius_df['residue'].isin(amino_acid_list)].copy()
    #print(aa_level_radius_df)
    
    aa_level_radius_df.loc[:,'res'] = aa_level_radius_df['residue'].map(lambda x: from3to1_general(x))
    aa_level_radius_df.loc[:,'res_num'] = aa_level_radius_df.index.map(lambda x: x[3][1])
    #print(aa_level_radius_df)

    tm_aa_level_radius_df = aa_level_radius_df[aa_level_radius_df.res_num.isin(list(itertools.chain.from_iterable(range1)))].copy()
    #print(tm_aa_level_radius_df)
    tm_aa_level_radius_df.to_csv(out_dir+'/'+short_filename+'_radius.csv')

@log_execution_time
def hole_annotation(msa_filename, radius_directory, norm_max_radius):
  timing_logger.info(f"Starting hole_annotation for {msa_filename}")
  # strategy:
  # read in nogap msa
  # read in all radius files
  # match up radius numbers to sequences, with correct gaps and everything
  # find maximum radius value
  # extract radius info (disregard gaps, and replace last two with 0 and max) into jalview file (for full.ali)
  # delete gaps from template sequence
  # find maximum radius value
  # extract radius info (disregard gaps, and replace last two with 0 and max) into jalview file (for nogap.ali)
  msa_location_prefix = msa_filename[:-8]
  full_annot_filename = msa_location_prefix+'full_annot.jlv'
  nogap_annot_filename = msa_location_prefix+'nogap_annot.jlv'
  full_msa_csv_filename = msa_location_prefix+'full_msa.csv'
  nogap_msa_csv_filename = msa_location_prefix+'nogap_msa.csv'
  full_radius_csv_filename = msa_location_prefix+'full_radius.csv'
  nogap_radius_csv_filename = msa_location_prefix+'nogap_radius.csv'
  with open(full_annot_filename, 'w') as full_annot_file:
    full_annot_file.write('JALVIEW_ANNOTATION\r\n')
  with open(nogap_annot_filename, 'w') as nogap_annot_file:
    nogap_annot_file.write('JALVIEW_ANNOTATION\r\n')
  radius_dict = {}
  radius_seq_dict = {}
  max_list = []
  for radius_filename in glob.glob(radius_directory+'/*_radius.csv'):
    pdb_id, radius_col, max_radius, radius_seq = read_one_radius_file(radius_filename)
    max_radius = float(max_radius)
    radius_dict[pdb_id] = radius_col
    radius_seq_dict[pdb_id] = radius_seq
    if not np.isnan(max_radius):
      max_list.append(max_radius)
  max_radius_full = np.nanmax(max_list)
  if max_radius_full <= norm_max_radius:
    max_radius_full = norm_max_radius
    logging.info('For normalization of '+full_annot_filename+', the maximum radius has been set to '+str(norm_max_radius)+' \u00c5, as specified in paths.txt. The minimum radius has been set to 0 \u00c5.')
  else:
    logging.info('For normalization of, '+full_annot_filename+', the maximum radius is '+str(max_radius_full)+' \u00c5, which exceeds than the norm_max_radius of '+str(norm_max_radius)+' \u00c5 set in paths.txt. Adjust scale bars or edit norm_max_radius in paths.txt and run again. The minimum radius has been set to 0 \u00c5.')
  full_radius_dict_norm = {}
  full_radius_dict_norm_match = {}
  align_df, template_id, msa_seq_dict = read_msa_file(msa_filename)
  template_mask = align_df.loc[:,msa_location_prefix[-5:-1]] != '-'
  nogap_align_df = align_df.loc[template_mask]
  #print(align_df)
  msa_to_csv(full_msa_csv_filename, align_df)
  #print(nogap_align_df)
  msa_to_csv(nogap_msa_csv_filename, nogap_align_df)
  matching_seqs_list = []
  for pdb_id, radius_list in radius_dict.items():
    radius_list_norm = list(radius_list)
    radius_list_norm[-2:] = [0,max_radius_full]
    full_radius_dict_norm[pdb_id] = radius_list_norm
    # need to check whether the sequences in the msa match the sequences in the radius files
    if radius_seq_dict[pdb_id] == msa_seq_dict[pdb_id]:
      matching_seqs_list.append(pdb_id)
      full_radius_dict_norm_match[pdb_id] = radius_list_norm
  #print(matching_seqs_list)
  #print(full_radius_dict_norm_match)
  radii_to_annotation_file(full_annot_filename, full_radius_dict_norm_match)

  align_radius_dict, max_radius_nogap = map_radius_to_msa(align_df, radius_dict, template_id, matching_seqs_list, full_radius_csv_filename, nogap_radius_csv_filename)
  if max_radius_nogap <= norm_max_radius:
    max_radius_nogap = norm_max_radius
    logging.info('For normalization of '+nogap_annot_filename+', the maximum radius has been set to '+str(norm_max_radius)+' \u00c5, as specified in paths.txt. The minimum radius has been set to 0 \u00c5.')
  else:
    logging.info('For normalization of, '+nogap_annot_filename+', the maximum radius is '+str(max_radius_nogap)+' \u00c5, which exceeds than the norm_max_radius of '+str(norm_max_radius)+' \u00c5 set in paths.txt. Adjust scale bars or edit norm_max_radius in paths.txt and run again. The minimum radius has been set to 0 \u00c5.')
  nogap_radius_dict_norm = {}
  nogap_radius_dict_norm_match = {}
  for pdb_id, radius_list in align_radius_dict.items():
    #print(radius_list)
    radius_list_norm = list(radius_list)
    radius_list_norm[-2:] = [0, max_radius_nogap]
    #print(radius_list_norm)
    nogap_radius_dict_norm[pdb_id] = radius_list_norm
    if pdb_id in matching_seqs_list:
      nogap_radius_dict_norm_match[pdb_id] = radius_list_norm
  radii_to_annotation_file(nogap_annot_filename, nogap_radius_dict_norm_match)
  timing_logger.info("Completed hole_annotation")

def read_one_radius_file(radius_filename):
  pdb_id = radius_filename[-20:-16]
  radius_df = pd.read_csv(radius_filename)
  #print(pdb_id, radius_df)
  max_radius = radius_df.radius.dropna().max()
  #print(pdb_id, max_radius)
  radius_df = pd.concat([radius_df]*4, ignore_index=True)
  sequence = ''.join(map(str,radius_df['res']))
  ### Need to remove NaNs at some point...
  return pdb_id, radius_df['radius'], max_radius, sequence

def read_msa_file(msa_filename):
  template_name = msa_filename[-13:-9]
  alignment = AlignIO.read(msa_filename, 'fasta')
  align_dict = {}
  msa_seq_dict = {}
  for record in alignment:
    align_dict[str(record.id)] = list(str(record.seq))
    msa_seq_dict[str(record.id)] = ''.join(c for c in str(record.seq) if c not in '-')
  #print(align_dict)
  align_df = pd.DataFrame.from_dict(align_dict) # makes dataframe with pdb_ids as column names
  #print(align_df)
  return align_df, template_name, msa_seq_dict

def map_radius_to_msa(align_df, radius_dict, template_name, match_list, radius_csv_filename, nogap_radius_csv_filename):
  # create copy of align_df
  align_radius_df = align_df
  #print(align_df)
  for pdb_id in radius_dict.keys():
  # replace every non-dash character with the corresponding radius
    dash_mask = align_radius_df.loc[:,pdb_id] != '-'
    same_size = align_radius_df.loc[dash_mask, pdb_id].size == len(radius_dict[pdb_id].tolist())
    size_diff = align_radius_df.loc[dash_mask, pdb_id].size - len(radius_dict[pdb_id].tolist())
    #print(pdb_id, same_size, size_diff)
    if pdb_id in match_list:
      align_radius_df.loc[dash_mask, pdb_id] = radius_dict[pdb_id].tolist()
  # make a mask corresponding to rows(?) that do NOT have a - dash in the template_name column
  radii_to_csv(radius_csv_filename, align_radius_df)
  template_mask = align_radius_df.loc[:,template_name] != '-'
  nogap_align_radius_df = align_radius_df.loc[template_mask]
  radii_to_csv(nogap_radius_csv_filename, nogap_align_radius_df)
  #print(nogap_align_radius_df)
  nogap_align_radius_dict = {}
  nogap_max_list = []
  for pdb_id in nogap_align_radius_df.columns:
    if pdb_id in match_list:
      radius_only_list = nogap_align_radius_df.loc[:,pdb_id] != '-'
      #print(radius_only_list)
      nogap_align_radius_dict[pdb_id] = nogap_align_radius_df.loc[radius_only_list,pdb_id]
      nogap_max_list.append(np.nanmax(nogap_align_radius_df.loc[radius_only_list, pdb_id].tolist()))
  #print(nogap_align_radius_dict)
  #print(np.nanmax(nogap_max_list))
  return nogap_align_radius_dict, np.nanmax(nogap_max_list)

def radii_to_annotation_file(annot_filename, radius_dict):
  with open(annot_filename, 'a') as annot_file:
    for pdb_id, radius_list in radius_dict.items():
      radius_list = ['' if np.isnan(x) else x for x in radius_list]
      radius_str = '|'.join(map(str, radius_list))
      annot_file.write('SEQUENCE_REF\t%s\r\n'%(pdb_id))
      annot_file.write('LINE_GRAPH\tradius\tHOLE radius\t%s\t\r\n'%(radius_str))

def radii_to_csv(radius_csv_filename, radius_df):
  # radius_dict to pandas dataframe
  # save dataframe as csv
  #radius_df = pd.DataFrame.from_dict(radius_dict)
  #print(radius_df)
  radius_df.to_csv(radius_csv_filename)

def msa_to_csv(msa_csv_filename, msa_df):
  #print(msa_df)
  msa_df.to_csv(msa_csv_filename)

def one_pdb_to_dssp(pdbid, pdb_file):
  dssp_dict = dssp_dict_from_pdb_file(pdb_file, "mkdssp")
  # print(dssp_dict)
  chain_list = []
  resid_list = []
  aa_list = []
  ss_list = []
  for key, value in dssp_dict[0].items():
    chain = key[0]
    resid = key[1][1]
    amino_acid = value[0]
    sec_struct = value[1]
    chain_list.append(chain)
    resid_list.append(resid)
    aa_list.append(amino_acid)
    ss_list.append(sec_struct)
    # check whether I'm going to need to sort the dataframe
  ss_df=pd.DataFrame(list(zip(chain_list, resid_list, aa_list, ss_list)), columns = ['chain', 'resid', 'aa', 'ss'])
  # print(pdbid, ss_df)
  return (pdbid, ss_df)

def insert_msa_gaps(ss_df_dict, msa_filename):
  #print(ss_df_dict)
  alignment = AlignIO.read(open(msa_filename), "fasta")
  ss_dict = {}
  for record in alignment:
    # print(str(record.id))
    ss_df_seq = ''.join(ss_df_dict[record.id]['aa'])
    msa_seq = str(record.seq)
    # print(msa_seq)
    for char in '-':
      msa_seq = msa_seq.replace(char, '')
    if ss_df_seq == msa_seq:
      # print(ss_df_seq)
      # print(msa_seq)
      properties = ['-']*len(str(record.seq))# list with same length as MSA, full of '-' gaps
      match_loc_list = [match.start() for match in re.finditer('[^-]', str(record.seq))]
      for i, idx in enumerate(match_loc_list):
        # print(i, idx)
        properties[idx]=ss_df_dict[str(record.id)]['ss'][i]
    ss_dict[str(record.id)] = ''.join(properties)
  # print(ss_dict)
  return ss_dict

def create_pseudo_fasta_dssp(seq_name, dssp_str, msa_file):
  fileloc = os.path.split(msa_file)
  templ_name = fileloc[1][:4]
  filename = fileloc[0] + '/stationary_' + templ_name + '/' + seq_name + '_' + templ_name + '_dssp.fa'
  with open(filename, 'w') as dssp_file:
    dssp_file.write('>' + seq_name + '\n')
    dssp_file.write(dssp_str + '\n')

def dssp_to_csv(dssp_csv_filename, dssp_df):
  dssp_df.to_csv(dssp_csv_filename)
@log_execution_time
def batch_dssp(msa_file, pdb_dir):
    timing_logger.info(f"Starting batch_dssp with MSA file: {msa_file}")
    dssp_df_dict = {}
    for filename in glob.glob(pdb_dir+'*_clean.pdb'):
        pdbid = filename[-14:-10]
        dssp_tuple = one_pdb_to_dssp(pdbid, filename)
        dssp_df_dict[dssp_tuple[0]] = dssp_tuple[1]
    dssp_dict = insert_msa_gaps(dssp_df_dict, msa_file)
    fileloc = os.path.split(msa_file)
    templ_name = fileloc[1][:4]
    dssp_msa_filename = fileloc[0] + '/' + templ_name + '_full_dssp.fa'
    dssp_csv_filename = fileloc[0] + '/' + templ_name + '_full_dssp.csv'
    alignment = AlignIO.read(open(msa_file), "fasta")
    test = list(zip(*dssp_dict.items()))
    test_seq = [list(str) for str in test[1]]
    templ_aa = pd.DataFrame(list(alignment[0].seq), columns=[templ_name+'_aa'])
    full_alignment = pd.DataFrame(test_seq, index=test[0])
    #full_alignment = pd.concat[templ_aa, full_alignment]
    full_alignment = full_alignment.transpose()
    full_alignment = pd.concat([templ_aa, full_alignment], axis=1, sort=False)
    #print(full_alignment)
    dssp_to_csv(dssp_csv_filename, full_alignment)
    mask = full_alignment[templ_name+'_aa'] != '-'
    nogap_alignment = full_alignment[mask]
    # print(nogap_alignment)
    dssp_csv_nogap_filename = fileloc[0] + '/' + templ_name + '_nogap_dssp.csv'
    dssp_msa_nogap_filename = fileloc[0] + '/' + templ_name + '_nogap_dssp.fa'
    nogap_msa_csv_filename = fileloc[0] + '/' + templ_name + '_nogap_msa.csv'
    dssp_to_csv(dssp_csv_nogap_filename, nogap_alignment) 
    with open(dssp_msa_nogap_filename, 'w') as dssp_msa_nogap:
        for column in nogap_alignment:
            dssp_msa_nogap.write('>' + column + '\n' + ''.join(nogap_alignment[column]) + '\n')
    for key, value in dssp_dict.items():
        # create_jalview_annot(key, value, msa_file)
        create_pseudo_fasta_dssp(key, value, msa_file)
        with open(dssp_msa_filename, 'a') as dssp_msa:
            dssp_msa.write('>' + key + '\n')
            dssp_msa.write(value + '\n')
    timing_logger.info("Completed batch_dssp")


def batch_annotation(input_directory, hole_ref_pdb, norm_max_radius, clean_dir):
    logging.info(f"Starting batch annotation. Reference PDB: {hole_ref_pdb}")
    try:
        norm_max_radius_float = float(norm_max_radius)
        msa_filename = f"{input_directory}{hole_ref_pdb}_full.ali"
        radius_directory = f"{input_directory}stationary_{hole_ref_pdb}"
        
        logging.info("Running HOLE annotation")
        hole_annotation(msa_filename, radius_directory, norm_max_radius_float)
        
        logging.info("Running DSSP batch processing")
        batch_dssp(msa_filename, clean_dir)
        
        logging.info("Batch annotation completed successfully")
    except Exception as e:
        logging.error(f"Error in batch annotation: {str(e)}")
        raise

@log_execution_time
def ident_sim_calc(input_directory, parent_pdbid):
  timing_logger.info(f"Starting ident_sim_calc for {parent_pdbid}")
  # Analyzes a multiple sequence alignment of the motifs of interest for identity and similarity relative to a single reference or parent sequence.
  blosum62_list = [[4.0, -1.0, -2.0, -2.0, 0.0, -1.0, -1.0, 0.0, -2.0, -1.0, -1.0, -1.0, -1.0, -2.0, -1.0, 1.0, 0.0, -3.0, -2.0, 0.0, -2.0, -1.0, 0.0, -4.0, 0.0], 
    [-1.0, 5.0, 0.0, -2.0, -3.0, 1.0, 0.0, -2.0, 0.0, -3.0, -2.0, 2.0, -1.0, -3.0, -2.0, -1.0, -1.0, -3.0, -2.0, -3.0, -1.0, 0.0, -1.0, -4.0, 0.0], 
    [-2.0, 0.0, 6.0, 1.0, -3.0, 0.0, 0.0, 0.0, 1.0, -3.0, -3.0, 0.0, -2.0, -3.0, -2.0, 1.0, 0.0, -4.0, -2.0, -3.0, 3.0, 0.0, -1.0, -4.0, 0.0], 
    [-2.0, -2.0, 1.0, 6.0, -3.0, 0.0, 2.0, -1.0, -1.0, -3.0, -4.0, -1.0, -3.0, -3.0, -1.0, 0.0, -1.0, -4.0, -3.0, -3.0, 4.0, 1.0, -1.0, -4.0, 0.0], 
    [0.0, -3.0, -3.0, -3.0, 9.0, -3.0, -4.0, -3.0, -3.0, -1.0, -1.0, -3.0, -1.0, -2.0, -3.0, -1.0, -1.0, -2.0, -2.0, -1.0, -3.0, -3.0, -2.0, -4.0, 0.0], 
    [-1.0, 1.0, 0.0, 0.0, -3.0, 5.0, 2.0, -2.0, 0.0, -3.0, -2.0, 1.0, 0.0, -3.0, -1.0, 0.0, -1.0, -2.0, -1.0, -2.0, 0.0, 3.0, -1.0, -4.0, 0.0], 
    [-1.0, 0.0, 0.0, 2.0, -4.0, 2.0, 5.0, -2.0, 0.0, -3.0, -3.0, 1.0, -2.0, -3.0, -1.0, 0.0, -1.0, -3.0, -2.0, -2.0, 1.0, 4.0, -1.0, -4.0, 0.0], 
    [0.0, -2.0, 0.0, -1.0, -3.0, -2.0, -2.0, 6.0, -2.0, -4.0, -4.0, -2.0, -3.0, -3.0, -2.0, 0.0, -2.0, -2.0, -3.0, -3.0, -1.0, -2.0, -1.0, -4.0, 0.0], 
    [-2.0, 0.0, 1.0, -1.0, -3.0, 0.0, 0.0, -2.0, 8.0, -3.0, -3.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0, -2.0, 2.0, -3.0, 0.0, 0.0, -1.0, -4.0, 0.0], 
    [-1.0, -3.0, -3.0, -3.0, -1.0, -3.0, -3.0, -4.0, -3.0, 4.0, 2.0, -3.0, 1.0, 0.0, -3.0, -2.0, -1.0, -3.0, -1.0, 3.0, -3.0, -3.0, -1.0, -4.0, 0.0], 
    [-1.0, -2.0, -3.0, -4.0, -1.0, -2.0, -3.0, -4.0, -3.0, 2.0, 4.0, -2.0, 2.0, 0.0, -3.0, -2.0, -1.0, -2.0, -1.0, 1.0, -4.0, -3.0, -1.0, -4.0, 0.0], 
    [-1.0, 2.0, 0.0, -1.0, -3.0, 1.0, 1.0, -2.0, -1.0, -3.0, -2.0, 5.0, -1.0, -3.0, -1.0, 0.0, -1.0, -3.0, -2.0, -2.0, 0.0, 1.0, -1.0, -4.0, 0.0], 
    [-1.0, -1.0, -2.0, -3.0, -1.0, 0.0, -2.0, -3.0, -2.0, 1.0, 2.0, -1.0, 5.0, 0.0, -2.0, -1.0, -1.0, -1.0, -1.0, 1.0, -3.0, -1.0, -1.0, -4.0, 0.0], 
    [-2.0, -3.0, -3.0, -3.0, -2.0, -3.0, -3.0, -3.0, -1.0, 0.0, 0.0, -3.0, 0.0, 6.0, -4.0, -2.0, -2.0, 1.0, 3.0, -1.0, -3.0, -3.0, -1.0, -4.0, 0.0], 
    [-1.0, -2.0, -2.0, -1.0, -3.0, -1.0, -1.0, -2.0, -2.0, -3.0, -3.0, -1.0, -2.0, -4.0, 7.0, -1.0, -1.0, -4.0, -3.0, -2.0, -2.0, -1.0, -2.0, -4.0, 0.0], 
    [1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -2.0, 0.0, -1.0, -2.0, -1.0, 4.0, 1.0, -3.0, -2.0, -2.0, 0.0, 0.0, 0.0, -4.0, 0.0], 
    [0.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -1.0, -1.0, -1.0, -2.0, -1.0, 1.0, 5.0, -2.0, -2.0, 0.0, -1.0, -1.0, 0.0, -4.0, 0.0], 
    [-3.0, -3.0, -4.0, -4.0, -2.0, -2.0, -3.0, -2.0, -2.0, -3.0, -2.0, -3.0, -1.0, 1.0, -4.0, -3.0, -2.0, 11.0, 2.0, -3.0, -4.0, -3.0, -2.0, -4.0, 0.0], 
    [-2.0, -2.0, -2.0, -3.0, -2.0, -1.0, -2.0, -3.0, 2.0, -1.0, -1.0, -2.0, -1.0, 3.0, -3.0, -2.0, -2.0, 2.0, 7.0, -1.0, -3.0, -2.0, -1.0, -4.0, 0.0], 
    [0.0, -3.0, -3.0, -3.0, -1.0, -2.0, -2.0, -3.0, -3.0, 3.0, 1.0, -2.0, 1.0, -1.0, -2.0, -2.0, 0.0, -3.0, -1.0, 4.0, -3.0, -2.0, -1.0, -4.0, 0.0], 
    [-2.0, -1.0, 3.0, 4.0, -3.0, 0.0, 1.0, -1.0, 0.0, -3.0, -4.0, 0.0, -3.0, -3.0, -2.0, 0.0, -1.0, -4.0, -3.0, -3.0, 4.0, 1.0, -1.0, -4.0, 0.0], 
    [-1.0, 0.0, 0.0, 1.0, -3.0, 3.0, 4.0, -2.0, 0.0, -3.0, -3.0, 1.0, -1.0, -3.0, -1.0, 0.0, -1.0, -3.0, -2.0, -2.0, 1.0, 4.0, -1.0, -4.0, 0.0], 
    [0.0, -1.0, -1.0, -1.0, -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0, 0.0, 0.0, -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -4.0, 0.0], 
    [-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, 1.0, 0.0], 
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
  blosum62_labels = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B', 'Z', 'X', '*', '-']
  blosum62 = pd.DataFrame(blosum62_list, columns=blosum62_labels, index=blosum62_labels)

  motif_file = input_directory+parent_pdbid+'_nogap.ali'
  with open(motif_file) as msa_file:
    msa = list(AlignIO.read(msa_file, 'fasta'))

  msa_dict = {}
  for record in msa:
    msa_dict[record.id[0:4]] = str(record.seq)
  #print(msa_dict)
  identity_data = []
  similarity_data = []
  pdbid_order = []
  identity_data_dict = {}
  similarity_data_dict = {}
  # compare every motif sequence to the parent and calculate identity and similarity (positives using BLOSUM62 matrix)
  for pdbid, motif in msa_dict.items():  
    ident = 0
    similar = 0
    for i_char, char in enumerate(motif):
      if char == msa_dict[parent_pdbid][i_char]:
        ident += 1
      if positive_match(char, msa_dict[parent_pdbid][i_char], blosum62):
        similar += 1
    ident = 100*(ident/len(msa_dict[parent_pdbid]))
    similar = 100*(similar/len(msa_dict[parent_pdbid]))
    identity_data.append(ident)
    similarity_data.append(similar)
    pdbid_order.append(pdbid)
    #print(parent_pdbid, pdbid, ident, similar)
  identity_data_dict['identity'] = identity_data
  identity_data_dict['PDB'] = pdbid_order
  similarity_data_dict['similarity'] = similarity_data
  similarity_data_dict['PDB'] = pdbid_order
  identity_data_df = pd.DataFrame.from_dict(identity_data_dict)
  identity_data_df.set_index('PDB', inplace=True)
  similarity_data_df = pd.DataFrame.from_dict(similarity_data_dict)
  similarity_data_df.set_index('PDB', inplace=True)
  #print(identity_data_df)
  #print(similarity_data_df)
  make_heatmap(identity_data_df, input_directory+parent_pdbid+'_nogap_identity', 'Blues', 'Identity')
  make_heatmap(similarity_data_df, input_directory+parent_pdbid+'_nogap_similarity', 'Oranges', 'Similarity')
  identity_data_df.to_csv(input_directory+parent_pdbid+'_nogap_identity.csv')
  similarity_data_df.to_csv(input)
  timing_logger.info("Completed ident_sim_calc")

def make_heatmap(data, output_filename, colormap, plot_type):
  fig, ax = plt.subplots(figsize=(10,30))
  sns.heatmap(data, vmin=0, vmax=100, cmap=colormap, cbar_kws={'label': plot_type+' (%)'})
  fig.savefig(output_filename+'.png', bbox_inches='tight')
  fig.savefig(output_filename+'.pdf', bbox_inches='tight')
  plt.close(fig)

def positive_match(char1, char2, matrix):
  return(matrix.loc[char1, char2] > 0) # similarity match defined as one with positive value in BLOSUM62 matrix
