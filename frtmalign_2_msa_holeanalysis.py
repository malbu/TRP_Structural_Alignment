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
#from MDAnalysis.analysis.hole2 import HoleAnalysis
#from hole2_mdakit import HoleAnalysis
#from mdahole2 import HoleAnalysis
from mdahole2.analysis import HoleAnalysis
from MDAnalysis.topology.guessers import guess_atom_element
from MDAnalysis import Universe
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
    #if not os.path.exists(paths['hole']):
    #    raise SystemExit(paths['hole']+' path to HOLE executable specified in paths.txt does not exist')
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
def batch_hole(directory_in, category_df, ref_struct, vdw_file, pore_point):
    timing_logger.info("Starting batch_hole")
    input_df = category_df.set_index('PDB ID')
    arg_list = []
    for filename in glob.glob(directory_in+"stationary_%s/*_full_align.pdb" %(ref_struct)):
        short_filename = filename[-24:-15]
        pdb_id = short_filename[0:4]
        #out_dir = os.path.split(filename)[0]
        out_dir = os.path.dirname(filename)
        arg_list.append((filename, short_filename, pdb_id, out_dir, input_df))
        single_hole(filename, short_filename, pdb_id, out_dir, input_df, vdw_file, pore_point)
    timing_logger.info("Completed batch_hole")
    #print(out_dir+"/"+short_filename+"_hole_out.txt", out_dir+"/"+short_filename+"_hole.pdb")
  #n_cpus = mp.cpu_count()
  #pool = mp.Pool(n_cpus)
  #results = [pool.apply(single_hole, args=arg_tup) for arg_tup in arg_list]
   

def assign_elements(u):
    """
    Assigns element symbols to atoms based on their names.
    
    """
    # Define a mapping from atom name prefixes to element symbols
    atom_element_mapping = {
        'C': 'C',
        'N': 'N',
        'O': 'O',
        'S': 'S',
        'H': 'H',
        'P': 'P',

    }
    
    for atom in u.atoms:
        # If the element is already assigned, skip
        if atom.element.strip():
            continue
        
        # Extract the first character of the atom name to infer the element
        atom_name = atom.name.strip()
        if not atom_name:
            atom.element = 'C'  # Default to Carbon if atom name is empty
            continue
        
        first_char = atom_name[0].upper()
        
        # Assign element based on the mapping
        if first_char in atom_element_mapping:
            atom.element = atom_element_mapping[first_char]
        else:
            # Default or handle unknown elements
            atom.element = 'C'  # You can choose a different default or implement more logic

@log_execution_time
def single_hole(filename, short_filename, pdb_id, out_dir, input_df, vdw_file, pore_point):
    """
    Runs HOLE analysis on a given PDB file using HoleAnalysis from MDAnalysis,
    attempting to automatically determine the pore axis.
    """
    start_time = time.time()
    timing_logger.info(f"Running HOLE: mobile={short_filename}, stationary={pdb_id}")
    print(f'm: {short_filename}, s: {pdb_id}')

    try:
        # Initialize Universe
        u = Universe(filename)
        logging.debug(f"Initialized Universe for {filename}")

        # Guess elements and masses
        from MDAnalysis.topology import guessers
        for atom in u.atoms:
            if not atom.element:
                atom.element = guessers.guess_atom_element(atom.name)
            if atom.mass == 0:
                atom.mass = guessers.guess_atom_mass(atom.element)

        # Use the correct HOLE executable path
        hole_executable = "/miniconda/envs/myenv/bin/hole"

        # Process pore_point
        if pore_point:
            if isinstance(pore_point, str):
                pore_point = [float(x) for x in pore_point.strip('[]').split(',')]
            elif isinstance(pore_point, (list, tuple)) and len(pore_point) == 3:
                pore_point = [float(x) for x in pore_point]
            else:
                logging.warning(f"Invalid pore_point format: {pore_point}. Using None.")
                pore_point = None

        # Initialize HoleAnalysis
        sphpdb_file = os.path.join(out_dir, f"{short_filename}_hole.sph")
        outfile = os.path.join(out_dir, f"{short_filename}_hole.out")
        H = HoleAnalysis(
            u,
            executable=hole_executable,
            vdwradii_file=vdw_file,
            ignore_residues=['SOL', 'WAT', 'TIP', 'HOH', 'K', 'NA', 'CL', 'CA', 'MG', 'GD', 'DUM', 'TRS'],
            cpoint=pore_point
        )
        logging.debug(f"Initialized HoleAnalysis for {short_filename}")

        # Set the output files
        H.sphpdb = sphpdb_file
        H.outfile = outfile

        # Run HOLE analysis
        H.run()
        logging.info(f"HOLE analysis completed for {short_filename}")

        # Copy the generated .sph file to the correct location
        source_sph = "/app/R1/hole000.sph"
        if os.path.exists(source_sph):
            shutil.copy2(source_sph, sphpdb_file)
            logging.info(f"Copied {source_sph} to {sphpdb_file}")
        else:
            logging.error(f"Source .sph file not found: {source_sph}")


        # Check if the results attribute exists and has data
        if hasattr(H, 'results') and len(H.results.profiles) > 0:
            profile = H.results.profiles[0]
        
            # Print available attributes for debugging
            logging.debug(f"Available attributes in profile: {profile.dtype.names}")
        
            # Use 'radius' and 'rxn_coord' attributes
            if 'rxn_coord' in profile.dtype.names and 'radius' in profile.dtype.names:
                x = profile['radius']
                y = profile['rxn_coord']

                fig, ax = plt.subplots()
                ax.axvline(4.1, linestyle='--', color='silver', label='Hydrated Ca')
                ax.axvline(3.6, linestyle=':', color='silver', label='Hydrated Na')
                ax.axvline(1.0, linestyle='-', color='silver', label='Dehydrated Ca/Na')
                ax.plot(x, y, label='HOLE Profile')
                plt.title(pdb_id)
                plt.xlabel('Radius (Å)')
                plt.ylabel('Pore coordinate (Å)')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.xlim(0, 10)
                plt.ylim(-40, 30)
                fig.legend(loc='upper right')
                fig.savefig(os.path.join(out_dir, f"{pdb_id}_hole_plot.png"), dpi=200, bbox_inches="tight")
                fig.savefig(os.path.join(out_dir, f"{pdb_id}_hole_plot.pdf"), dpi=200, bbox_inches="tight")
                plt.close(fig)
                logging.info(f"HOLE profile plot saved for {pdb_id}")

                # Save radius data to CSV
                radius_df = pd.DataFrame({'res': range(len(x)), 'radius': x})
                radius_csv_path = os.path.join(out_dir, f"{short_filename}_radius.csv")
                radius_df.to_csv(radius_csv_path, index=False)
                logging.info(f"Radius data saved to {radius_csv_path}")
            else:
                logging.error(f"Expected 'rxn_coord' and 'radius' not found in profile. Available fields: {profile.dtype.names}")
        else:
            logging.warning(f"No profiles found for {short_filename}. Skipping plot generation.")

        # Residue-level analysis
        if len(input_df.loc[pdb_id, 'TM chains']) != 4:
            logging.info(f"{pdb_id} does not have four chains, skipping residue-level analysis.")
            return

        chain_info = input_df.loc[pdb_id, 'TM chains']
        chain_ids = [chain[0] for chain in chain_info]
        residue_ranges = [chain[1] for chain in chain_info]

        if not all(r == residue_ranges[0] for r in residue_ranges):
            logging.info(f"{pdb_id} does not have the same residue range for all chains, skipping residue-level analysis.")
            return

        # Parse original PDB structure
        pdb_parser = PDBParser(QUIET=True)
        structure = pdb_parser.get_structure(pdb_id, filename)

        # Parse HOLE output SPH file
        hole_sph_list = []
        with open(sphpdb_file, 'r') as sph_file:
            for line in sph_file:
                if line.startswith('ATOM'):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    radius = float(line[54:60])
                    hole_sph_list.append([x, y, z, radius])
        hole_sph_df = pd.DataFrame(hole_sph_list, columns=['x', 'y', 'z', 'radius'])

        residue_level_radius = {}
        for residue in structure[0][chain_ids[0]]:
            # Fetch corresponding residues from other chains
            try:
                residues_other = [structure[0][chain][residue.get_id()[1]] for chain in chain_ids[1:]]
            except KeyError:
                # Missing residues in other chains
                residue_level_radius[residue.get_full_id()] = [residue.get_resname(), 'NaN']
                continue

            # Check if residues have the same name
            if not all(residue.get_resname() == res.get_resname() for res in residues_other):
                residue_level_radius[residue.get_full_id()] = [residue.get_resname(), 'NaN']
                continue

            # Gather atom coordinates from all chains
            atom_dict = {}
            for atom in residue:
                atom_name = atom.get_name()
                coords = [atom.get_coord()]
                for res in residues_other:
                    if atom_name in res:
                        coords.append(res[atom_name].get_coord())
                    else:
                        coords.append(None)  # Missing atom
                # Filter out None values
                coords = [coord for coord in coords if coord is not None]
                if len(coords) != len(chain_ids):
                    continue  # Incomplete atom data

                # Calculate average coordinates
                avg_coord = np.mean(coords, axis=0)

                # Find the closest sphere in HOLE output based on z-coordinate
                closest_idx = (np.abs(hole_sph_df['z'] - avg_coord[2])).idxmin()
                closest_radius = hole_sph_df.loc[closest_idx, 'radius']

                # Adjust for van der Waals radius
                vdwr = from_name_to_vdwr(atom_name)
                radius_distance = (closest_radius / 2) - vdwr

                if radius_distance >= 0:
                    atom_dict[atom_name] = (avg_coord[0], avg_coord[1], avg_coord[2], radius_distance)

            if not atom_dict:
                residue_level_radius[residue.get_full_id()] = [residue.get_resname(), 'NaN']
                continue

            # Determine the minimal radius_distance among contributing atoms
            min_radius = min([tup[3] for tup in atom_dict.values()])

            residue_level_radius[residue.get_full_id()] = [residue.get_resname(), min_radius]

        # Create DataFrame from residue-level radius data
        residue_level_radius_df = pd.DataFrame.from_dict(residue_level_radius, orient='index', columns=['residue', 'radius'])

        # Filter for standard amino acids
        amino_acid_list = ['ALA','ARG','ASN','ASP','CYS','GLU','GLN','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','MSE']
        aa_level_radius_df = residue_level_radius_df[residue_level_radius_df['residue'].isin(amino_acid_list)].copy()

        # Map three-letter codes to one-letter codes
        aa_level_radius_df['res'] = aa_level_radius_df['residue'].apply(from3to1_general)
        aa_level_radius_df['res_num'] = aa_level_radius_df.index.map(lambda x: x[3][1])

        # Filter for transmembrane region residues
        tm_aa_level_radius_df = aa_level_radius_df[aa_level_radius_df['res_num'].isin(list(itertools.chain.from_iterable(residue_ranges)))].copy()

        # Save the radius data
        tm_aa_level_radius_df.to_csv(os.path.join(out_dir, f"{short_filename}_radius.csv"), index=False)
        logging.info(f"Residue-level radius data saved for {pdb_id}")

    except Exception as e:
        logging.error(f"Error during residue-level analysis for {pdb_id}: {e}")
        raise

    end_time = time.time()
    timing_logger.info(f"Single HOLE analysis for {short_filename} completed in {end_time - start_time:.2f} seconds")


@log_execution_time
def hole_annotation(msa_filename, radius_directory, norm_max_radius):
    """
    Annotates the MSA with HOLE radius data.

    """
    timing_logger.info(f"Starting hole_annotation for {msa_filename}")
    logging.info(f"Starting hole_annotation for {msa_filename}")

    try:
        # Read all radius CSV files
        radius_files = glob.glob(os.path.join(radius_directory, '*_radius.csv'))
        if not radius_files:
            logging.error(f"No radius CSV files found in {radius_directory}.")
            logging.info(f"Contents of {radius_directory}:")
            for item in os.listdir(radius_directory):
                logging.info(f"  {item}")
            raise FileNotFoundError(f"No radius CSV files found in {radius_directory}.")

        radius_dict = {}
        max_list = []

        for radius_file in radius_files:
            pdb_id, radius_col, max_radius, sequence = read_one_radius_file(radius_file)
            if pd.isna(max_radius):
                logging.warning(f"Max radius for {pdb_id} is NaN. Skipping.")
                continue
            radius_dict[pdb_id] = radius_col
            max_list.append(max_radius)

        if not max_list:
            logging.error("No valid max_radius values found. Exiting hole_annotation.")
            raise ValueError("No valid max_radius values found.")

        max_radius_full = np.nanmax(max_list)

        # Compare with norm_max_radius and log accordingly
        if max_radius_full <= norm_max_radius:
            max_radius_full = norm_max_radius
            logging.info(f"Normalization: max_radius set to norm_max_radius={norm_max_radius} Å.")
        else:
            logging.info(f"Normalization: max_radius={max_radius_full} Å exceeds norm_max_radius={norm_max_radius} Å.")

        # Read MSA file
        alignment = AlignIO.read(msa_filename, 'fasta')
        msa_dict = {record.id: str(record.seq) for record in alignment}

        # Initialize dictionary for annotations
        annotated_radius = {}

        for pdb_id, radius_list in radius_dict.items():
            if pdb_id not in msa_dict:
                logging.warning(f"PDB ID {pdb_id} from radius data not found in MSA. Skipping.")
                continue
            msa_seq = msa_dict[pdb_id]
            # Remove gaps for comparison
            msa_seq_nogap = msa_seq.replace('-', '')
            if len(msa_seq_nogap) != len(radius_list):
                logging.warning(f"Length mismatch for {pdb_id}: MSA sequence (nogap) length {len(msa_seq_nogap)} vs radius list length {len(radius_list)}.")
                # Handle mismatch appropriately, e.g., skip or adjust
                continue
            # Normalize radius values
            radius_norm = [min(max(r, 0), max_radius_full) / max_radius_full for r in radius_list]
            annotated_radius[pdb_id] = radius_norm

        # Create DataFrame for annotations
        annotated_radius_df = pd.DataFrame.from_dict(annotated_radius, orient='index')
        annotated_radius_df.columns = [f"pos_{i+1}" for i in range(annotated_radius_df.shape[1])]

        # Save annotations to a CSV file
        annotated_radius_csv = os.path.splitext(msa_filename)[0] + '_hole_annotation.csv'
        annotated_radius_df.to_csv(annotated_radius_csv)
        logging.info(f"HOLE annotations saved to {annotated_radius_csv}")

        # Generate Jalview annotation file (JALVIEW expects specific formats)
        jalview_annotation = os.path.splitext(msa_filename)[0] + '_hole_annotation.jlv'
        with open(jalview_annotation, 'w') as jlv_file:
            jlv_file.write('JALVIEW_ANNOTATION\r\n')
            for pdb_id, radius_norm in annotated_radius.items():
                radius_str = '|'.join(map(str, radius_norm))
                jlv_file.write(f'SEQUENCE_REF\t{pdb_id}\r\n')
                jlv_file.write(f'LINE_GRAPH\tradius\tHOLE radius\t{radius_str}\r\n')
        logging.info(f"Jalview annotation file saved to {jalview_annotation}")

    except Exception as e:
        logging.error(f"Error in hole_annotation: {e}")
        raise

    timing_logger.info("Completed hole_annotation")
    logging.info("Completed hole_annotation")

def read_one_radius_file(radius_filename):
    pdb_id = os.path.basename(radius_filename).split('_')[0]
    radius_df = pd.read_csv(radius_filename)

    if 'res' not in radius_df.columns or 'radius' not in radius_df.columns:
        logging.error(f"Radius file {radius_filename} missing required columns.")
        raise ValueError(f"Radius file {radius_filename} missing required columns.")

    max_radius = radius_df['radius'].dropna().max()
    sequence = ''.join(radius_df['res'].astype(str).tolist())

    return pdb_id, radius_df['radius'].tolist(), max_radius, sequence


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
        msa_filename = os.path.join(input_directory, f"{hole_ref_pdb}_full.ali")
        radius_directory = os.path.join(input_directory, f"stationary_{hole_ref_pdb}")
        
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
  #similarity_data_df.to_csv(input) this seesm like a mistake
  similarity_data_df.to_csv(input_directory + parent_pdbid + '_nogap_similarity.csv') #instead of above
  timing_logger.info("Completed ident_sim_calc")

def make_heatmap(data, output_filename, colormap, plot_type):
  fig, ax = plt.subplots(figsize=(10,30))
  sns.heatmap(data, vmin=0, vmax=100, cmap=colormap, cbar_kws={'label': plot_type+' (%)'})
  fig.savefig(output_filename+'.png', bbox_inches='tight')
  fig.savefig(output_filename+'.pdf', bbox_inches='tight')
  plt.close(fig)

def positive_match(char1, char2, matrix):
  return(matrix.loc[char1, char2] > 0) # similarity match defined as one with positive value in BLOSUM62 matrix
