import os
import sys
import importlib

def check_hole_executable():
    hole_path = "/miniconda/envs/myenv/bin/hole"
    if not os.path.exists(hole_path):
        print(f"HOLE executable not found at {hole_path}. Please ensure HOLE is installed correctly.")
        sys.exit(1)
    return hole_path



if len(sys.argv) < 2:
    raise SystemExit("Usage: python %s <path to paths.txt> [--logging {original,detailed}] [--run-hole-analysis]" % sys.argv[0])
else:
    paths_file = sys.argv[1].strip()

# Check for additional arguments
logging_mode = 'original'
run_hole_analysis = False
use_spear = False
if len(sys.argv) > 2:
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == '--logging':
            if i + 1 < len(sys.argv) and sys.argv[i + 1] in ['original', 'detailed']:
                logging_mode = sys.argv[i + 1]
            else:
                raise SystemExit("Invalid logging option. Use 'original' or 'detailed'.")
        elif sys.argv[i] == '--run-hole-analysis':
            run_hole_analysis = True
        elif sys.argv[i] == '--use-spear':
            use_spear = True

# Import the appropriate module based on logging_mode and run_hole_analysis
if logging_mode == 'detailed':
    frtmalign_module = importlib.import_module('frtmalign_2_msa_additional_logging')
elif run_hole_analysis:
    frtmalign_module = importlib.import_module('frtmalign_2_msa_holeanalysis')
else:
    frtmalign_module = importlib.import_module('frtmalign_2_msa')

# Use the imported module's functions instead of the original ones
paths_dic = frtmalign_module.paths_dic
xml_parser = frtmalign_module.xml_parser
get_struct = frtmalign_module.get_struct
strip_tm_chains = frtmalign_module.strip_tm_chains
batch_frtmalign = frtmalign_module.batch_frtmalign
frtmalign_2_list = frtmalign_module.frtmalign_2_list
frtmalign_2_tables = frtmalign_module.frtmalign_2_tables
batch_align_merger = frtmalign_module.batch_align_merger
batch_hole = frtmalign_module.batch_hole
batch_annotation = frtmalign_module.batch_annotation
ident_sim_calc = frtmalign_module.ident_sim_calc

if use_spear:
    print('Info: Running SPEAR pipeline analysis')
    paths = paths_dic(paths_file)
    struct_info_dic, struct_info_df = xml_parser(paths['structs_info'])
    
    # Do preprocessing steps first (same as non-SPEAR path)
    print('Info: Acquiring and preprocessing structures.')
    for pdb_id, value in struct_info_dic.items():
        success = get_struct(pdb_id, paths['pdb_dir'], paths['provided_struct'])
        if success:
            num = strip_tm_chains(paths['clean_dir'], pdb_id, 
                                paths['pdb_dir'] + pdb_id + '.pdb', 
                                struct_info_dic[pdb_id]['tm_chains'])
    
    # Now run SPEAR analysis
    try:
        from spear import run_spear_pipeline
        
        results, spear_analysis = run_spear_pipeline(
            clean_dir=paths['clean_dir'], 
            frtmalign_dir=paths['frtmalign_dir'], 
            frtmalign_path=paths['frtmalign'],
            original_dir=paths['pdb_dir'],
            clean_dir_path=paths['clean_dir'],
            category_df=struct_info_df,
            vdw_file=paths['vdw_radius_file'],
            pore_point=paths['hole_reference_pore_point'],
            paths=paths,
            config=None
        )
        print('Info: SPEAR analysis completed successfully')
    except Exception as e:
        print(f'Error: SPEAR analysis failed: {str(e)}')
        raise

else:

    # set working directory, output directories
    # identify files containing necessary information, including paths to program files (frtmalign, HOLE, pyali), information for each structure
    print('Info: Reading paths.txt.')
    paths = paths_dic(paths_file)
    if not os.path.isfile(paths['structs_info']):
        raise SystemExit('Error: File for structure info ' + paths['structs_info'] + ' does not exist.')

    # establish structures to include in the analysis, as well as chain order and residues to be kept when cleaning files
    # Note that paths['structs_info'] is an xml file containing information for each structure, including: pdbid, chain order, starting and ending residue numbers for transmembrane domain, and categories (subfamily, environment, method, and ligand)
    print('Info: Reading xml file with structure info.')
    struct_info_dic, struct_info_df = xml_parser(paths['structs_info'])

    # acquire structures from OPM based on list of pdb codes and save to paths['pdb_dir']
    # perform cleaning based on residue ranges specified in paths['structs_info'] xml file
    # save cleaned structures to paths['clean_dir']
    print('Info: Acquiring and preprocessing structures.')
    for pdb_id, value in struct_info_dic.items():
        success = get_struct(pdb_id, paths['pdb_dir'], paths['provided_struct'])
        if success:
            num = strip_tm_chains(paths['clean_dir'], pdb_id, paths['pdb_dir'] + pdb_id + '.pdb', struct_info_dic[pdb_id]['tm_chains'])

    # perform all-vs-all pairwise alignments using Fr-TM-Align on all structures in the paths['clean_dir']
    # save Fr-TM-Align output, including aligned structures, transformation matrices, and text files, to paths['frtmalign_dir']
    # create separate folders for each stationary structure within paths['frtmalign_dir']
    # file names include the mobile PDB ID first and the stationary PDB ID second
    print('Info: Running Fr-TM-Align to align all structures pairwise.')
    batch_frtmalign(paths['clean_dir'], paths['frtmalign_dir'], paths['frtmalign'], paths['pdb_dir'], paths['clean_dir'])

    # process Fr-TM-Align text files to extract data for all pairwise alignments (TM-score, RMSD, aligned length, sequence identity)
    # save all outputs to paths['frtmalign_dir']
    # output data to .csv files for further analysis
    # create clustermap as in Figure 2, with TM scores clustered along the stationary axis and with color codes for qualitative categories (subfamily, environment, method, ligand)
    # saves post-clustering list of pdbids to a .csv file for later use
    print('Info: Processing, organizing, and saving Fr-TM-Align output.')
    frtmalign_list = frtmalign_2_list(paths['frtmalign_dir'])
    frtmalign_2_tables(frtmalign_list, struct_info_df, paths['frtmalign_dir'])

    # create multiple sequence alignments, using each stationary structure as reference
    # save alignment files in paths['frtmalign_dir']
    # name alignment files based on reference (i.e. common stationary) structure
    # full alignment includes all residues from all structures
    # nogap alignment removes all gaps/insertions from reference structure
    print('Info: Constructing multiple sequence alignments from pairwise structural alignments.')
    batch_align_merger(paths['frtmalign_dir'], paths['frtmalign_dir']+'stationary_clustering_order.csv')

    # perform HOLE pore radius analysis for each structure and determine minimum radius of each residue
    # save HOLE output files in paths['frtmalign_dir']
    #print('Info: Running permeation pathway HOLE profile analysis on structures aligned to reference structure '+paths['hole_reference_struct'])
    #batch_hole(paths['frtmalign_dir'], struct_info_df, paths['hole'], paths['hole_reference_struct'], paths['vdw_radius_file'], paths['hole_reference_pore_point'])
    # Check for HOLE executable
    check_hole_executable()

    print('Info: Running permeation pathway HOLE profile analysis on structures aligned to reference structure '+paths['hole_reference_struct'])
    #this batch_hole runs for for frtmalign_2_msa_holeanalysis.py
    batch_hole(paths['frtmalign_dir'], struct_info_df, paths['hole_reference_struct'], paths['vdw_radius_file'], paths['hole_reference_pore_point'])

    #this batch_hole runs for frtmalign_2_msa_additiontal_logging.py
    #batch_hole(paths['frtmalign_dir'], struct_info_df, paths['hole'], paths['hole_reference_struct'], paths['vdw_radius_file'], paths['hole_reference_pore_point'])

    # create Jalview annotation file for minimum radius of each residue in full and nogap multiple sequence alignments
    # perform DSSP secondary structure analysis for each structure and determine secondary structure of each residue
    # create pseudo-fasta file containing secondary structure of each residue in full and nogap multiple sequence alignments
    # save output files in paths['frtmalign_dir']
    #print('Info: Running secondary structure dssp analysis on structures aligned to reference structure '+paths['hole_reference_struct']+' and making HOLE radius annotation files and dssp pseudo-FASTA files.')
    #batch_annotation(paths['frtmalign_dir'], paths['hole_reference_struct'], paths['norm_max_radius'], paths['clean_dir'])
    print('Info: Running secondary structure dssp analysis on structures aligned to reference structure '+paths['hole_reference_struct']+' and making HOLE radius annotation files and dssp pseudo-FASTA files.')
    batch_annotation(paths['frtmalign_dir'], paths['hole_reference_struct'], paths['norm_max_radius'], paths['clean_dir'])

    print('Info: Running identity and similarity calculations on multiple sequence alignment aligned to reference structure '+paths['hole_reference_struct']+'.')
    ident_sim_calc(paths['frtmalign_dir'], paths['hole_reference_struct'])

    print('Info: Structural alignment is complete.')