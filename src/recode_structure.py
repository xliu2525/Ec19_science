import datetime
from os import path
import yaml
from typing import Dict, List

import numpy as np
import pandas as pd
import atomium
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import substitution_matrices
import torch

from colabdesign.af.alphafold.common import residue_constants

from common import memory
from mpnn import load_data_from_pdb_path, score_seq_mpnn, init_mpnn_model, compute_mpnn_seqs, \
    generate_seq, alphabet as mpnn_alphabet, unpack_probs_by_name
from af2rank import af2rank, score_seqs
from pdb_utils import extract_fixed_chains, extract_chains_ids
from log import dlog

def load_esm(model_name="esm2_t33_650M_UR50D", device_index=0):
    esm_model, esm_alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
    esm_model.eval().to(f'cuda:' + str(device_index))
    print("ESM initialized")
    return esm_model, esm_alphabet


def blosum_distance(seq, wt, matrix=substitution_matrices.load('BLOSUM62')):
    distance = 0
    for i in range(len(wt)):
        distance += matrix.get((seq[i], wt[i]), matrix.get((wt[i], seq[i])))
    return distance

@memory.cache(ignore=['model','alphabet','device'])
def compute_esm_seq_logprobs(model, model_name: str, alphabet, seq: str, device: str):
    model = model.eval()
    torch.set_grad_enabled(False)
    batch_converter = alphabet.get_batch_converter()
    data = [('protein', seq)]
    _, _, toks = batch_converter(data)
    # run model just once
    logits = model(toks.to(device))['logits'].cpu()
    logprobs = torch.log_softmax(logits[0], dim=-1)
    return logprobs


@memory.cache(ignore=['model','alphabet','device'])
def compute_esm_seq_pseudo_log_likelihood(model, model_name: str, alphabet, seq: str, device: str,
                                         include_full_logprobs=False):
    model = model.eval()
    torch.set_grad_enabled(False)
    batch_converter = alphabet.get_batch_converter()
    masked_logprobs = []
    data = [('protein', seq)]
    _, _, toks = batch_converter(data)
    full_logprobs = []

    for i in range(toks.size(-1) - 2):  # exclude start and end tokens
        j = i + 1
        tok_idx = toks[0][j]

        # [1, N_res]
        masked_toks = toks.clone()
        masked_toks[0][j] = alphabet.mask_idx
        logits = model(masked_toks.to(device))['logits'].cpu()

        # [1, N_res, N_vocab]
        logprobs = torch.log_softmax(logits[0, j], dim=-1)
        full_logprobs.append(logprobs)
        logprob = logprobs[tok_idx]
        masked_logprobs.append(logprob)
        del masked_toks

    if include_full_logprobs:
        return np.mean(masked_logprobs), torch.stack(full_logprobs)
    else:
        return np.mean(masked_logprobs)

def replace_letter(seq, letter_to_change):
    if letter_to_change == 'I':
        return seq.replace('I', 'V')
    elif letter_to_change == 'V':
        return seq.replace('V', 'I')
    elif letter_to_change == 'X':
        return seq
    else:
        raise ValueError(f"Unexpected letter to change {letter_to_change}")

def splice_reference_seq(seq, initial_seq, reference_seq, letter_to_redesign):
    index = reference_seq.find(initial_seq)
    assert (index != -1), f"Can't find {initial_seq} in reference {reference_seq}"
    prefix = reference_seq[:index]
    suffix = reference_seq[index+len(initial_seq):]
    
    return replace_letter(prefix, letter_to_redesign) + seq + replace_letter(suffix, letter_to_redesign)

def compute_protein_scores(data_root: str, seqs_by_lib: Dict[str,str], methods: List[str], reference_seq: str, 
                           uniprot_id: str, letter_to_redesign: str, 
                           trim_seqs=False, recycles=1, standardize_scores=True,
                           esm_model=None, esm_alphabet=None, esm_model_name=None,
                           mpnn_model=None, mpnn_dataset=None, mpnn_chain_id_dict=None,
                           multimer=True, single_chain=False, per_residue_data=False):
    """ 
    Args:
        - seqs_by_lib: mapping between names of multiple designs and the actual designed sequences for a given protein.
        - methods: models used to scores the designed sequences.
        - reference_seq: wild type sequence. 
    Returns dataframe with normalized scores for each method.
    """
    pdb_name = f"{uniprot_id}_nearby_protein_{letter_to_redesign}"
    pdb_path = path.join(data_root, f"{pdb_name}.pdb")
    target_chain = "X"
    fixed_chains_list, initial_seq = extract_fixed_chains(pdb_path, target_chain)    
    start_index = reference_seq.find(initial_seq)

    if single_chain:
        fixed_chains_list = None

    # will hold results for df:
    # seq, method, score, metric, model
    results = []

    # eliminate duplicates in seqs_by_lib
    for lib, seqs in seqs_by_lib.items():
        seqs_by_lib[lib] = list(set(seqs))

    if trim_seqs:
        # trim seqs to only include the region equivalent to initial seq
        # find offset of the initial seq in the reference seq, then extract the same region from each seq
        for lib, seqs in seqs_by_lib.items():
            seqs_by_lib[lib] = [seq[start_index:start_index+len(initial_seq)] for seq in seqs]

    if 'esm' in methods:
        for lib_name, seqs in seqs_by_lib.items():
            for i, seq in enumerate(seqs):
                print(f"Computing ESM score, {i}/{len(seqs)}")
                spliced_seq = splice_reference_seq(seq, initial_seq, reference_seq, letter_to_redesign)
                if per_residue_data:
                    score, full_logprobs = compute_esm_seq_pseudo_log_likelihood(esm_model, esm_model_name, esm_alphabet, spliced_seq, 
                                                                                 'cuda:0', include_full_logprobs=True)
                    results.append((seq, 'esm', full_logprobs, 'full_log_probs', 'esm', lib_name))                
                else:
                    score = compute_esm_seq_pseudo_log_likelihood(esm_model, esm_model_name, esm_alphabet, spliced_seq, 'cuda:0')
                results.append((seq, 'esm', score, 'pseudo_log_likelihood', 'esm', lib_name))

    if 'af2rank' in methods:
        if fixed_chains_list:
            chains = f"{target_chain},{','.join(fixed_chains_list)}"
        else:
            chains = f"{target_chain}"

        if multimer:
            af_model_names = [f"model_{model_num}_multimer_v3" for model_num in range(1,6)]
        else:
            af_model_names = [f"model_{model_num}_ptm" for model_num in range(1,3)]
        af = af2rank(pdb_path, chains, model_names=af_model_names)
        dlog(af, s=1)
        wt_multi_seq = "".join([residue_constants.restypes[aa] for aa in af.model._wt_aatype])
        for lib_name, seqs in seqs_by_lib.items():
            print(f"Scoring {lib_name}")
            seq_metric_scores = score_seqs(af, seqs, initial_seq, wt_multi_seq, af_model_names, recycles=recycles, per_res=per_residue_data)
            for seq, model_scores in seq_metric_scores.items():
                for model, model_score in zip(af_model_names, model_scores):
                    for metric, score in model_score.items():
                        if isinstance(score, float) or isinstance(score, np.ndarray):
                            results.append((seq, 'af2rank', score, metric, model, lib_name))

    if 'blosum' in methods:
        for lib_name, seqs in seqs_by_lib.items():
            for seq in seqs:
                score = blosum_distance(seq, initial_seq)
                results.append((seq, 'blosum', score, 'blosum', 'blosum62', lib_name))

    if 'mpnn' in methods:
        for lib_name, seqs in seqs_by_lib.items():
            for i, seq in enumerate(seqs):
                print(f"Computing MPNN score, {i}/{len(seqs)}")
                score = score_seq_mpnn(mpnn_model, seq, mpnn_dataset, mpnn_chain_id_dict)
                results.append((seq, 'mpnn', score, 'mpnn_score', 'mpnn', lib_name))

    # turn results into dataframe
    df = pd.DataFrame(results, columns=['seq', 'method', 'score', 'metric', 'model', 'lib'])
    if standardize_scores:
        # standardize scores by method and model
        df['score'] = df.groupby(['method', 'model', 'metric'])['score'].transform(lambda x: (x - x.mean()) / (x.std()+1e-8))
    # drop rows with- nans in score column
    df = df.dropna(subset=['score'])

    if 'af2rank' in methods:
        # make "avg model" with average between af2rank models
        df_avg = df[df['method'] == 'af2rank'].groupby(['seq', 'metric', 'lib'])['score'].mean().reset_index()
        df_avg['model'] = 'avg'
        df_avg['method'] = 'af2rank'
        df = pd.concat([df, df_avg])

    return df

def find_nearby_res(pdb_path, target_chain, recode_positions, redesign_radius=5, 
                    top_to_take=2, include_neighbors=True):
    pdb = atomium.open(pdb_path)
    pdb.model.optimise_distances()

    target_c = pdb.model.chain(target_chain)
  
    all_residues = list(target_c.residues())
    chain_neighbor_indices = set()
    for res in target_c.residues():      
        res_index = all_residues.index(res)+1
        if res_index not in recode_positions:
            continue
        if include_neighbors:
            if res_index != len(all_residues):
                chain_neighbor_indices.add(res_index+1)
            if res_index > 1:
                chain_neighbor_indices.add(res_index-1)
        
        nearby_res_by_distance = {}
        atoms = res.nearby_atoms(redesign_radius)
        for atom in atoms:
            if atom.chain.id != target_c.id:
                continue
            nearby_res = atom.het
            nearby_res_index = all_residues.index(nearby_res)+1
            if abs(nearby_res_index - res_index) < 6:
                continue
                
            distance = min([atom.distance_to(r_atom) for r_atom in res.atoms()])
            if distance < nearby_res_by_distance.get(nearby_res_index, np.inf):
                nearby_res_by_distance[nearby_res_index] = distance
                
        # Find closest residues
        sorted_by_distance = sorted(list(nearby_res_by_distance.items()), key=lambda kv: kv[1])
        top = [k for k, v in sorted_by_distance[:top_to_take]]
        chain_neighbor_indices.update(top)                

    # if first residue has code 'M', remove it from the chain_neighbor_indices
    if all_residues[0].code == 'M':      
        print("Removing first M")
        chain_neighbor_indices.discard(1)
    dlog(chain_neighbor_indices)

    return sorted(chain_neighbor_indices)

def prepare_multichain_dataset(pdb_path, letter_to_redesign, target_chain, fixed_chains_list, include_contacts, 
                               redesign_radius = None, top_to_take=None, include_neighbors=True, recode_positions=None,
                               starting_seq=None):
    if include_contacts:
        fixed_chains = ",".join(fixed_chains_list)
    else:
        fixed_chains = ""
    dataset, chain_id_dict, _ = load_data_from_pdb_path(pdb_path, target_chain, fixed_chains)
    if starting_seq == None:
        starting_seq = dataset.data[0][f'seq_chain_{target_chain}']

    dlog(recode_positions,v=1)
    if recode_positions == None:
        recode_positions = [i+1 for i, c in enumerate(list(starting_seq)) if c == letter_to_redesign]
    if redesign_radius != None:
        nearby_positions = find_nearby_res(pdb_path, target_chain, recode_positions, 
                                           redesign_radius, top_to_take, include_neighbors)
        recode_positions = sorted(list(set(recode_positions) | set(nearby_positions)))

    fixed_positions = [i+1 for i, c in enumerate(list(starting_seq)) if (i+1) not in recode_positions]
    dlog(recode_positions, fixed_positions,v=1)
    
    pdb_name = path.basename(pdb_path).split('.')[0]
    fixed_positions_dict = { pdb_name: { target_chain: fixed_positions } }
    if letter_to_redesign:
        omit_AA_dict = { pdb_name: { target_chain: [[recode_positions, letter_to_redesign]] } }
    else:
        omit_AA_dict = {}
    dlog(omit_AA_dict, v=1)
    return dataset, chain_id_dict, fixed_positions_dict, omit_AA_dict, recode_positions


def generate_mpnn_designs(data_root, code_root, gene_name, uniprot_id, reference_seq, letter_to_redesign,
    include_neighbors, temp, mpnn_designs_num, redesign_radius, top_to_take, config_version=None):
    pdb_name = f"{uniprot_id}_nearby_protein_{letter_to_redesign}"
    pdb_path = path.join(data_root, f"{pdb_name}.pdb")
    target_chain = "X"
    fixed_chains_list, initial_seq = extract_chains_ids(pdb_path, target_chain)
    fixed_chains_list = [chain for chain in fixed_chains_list if chain != target_chain]

    mpnn_model = init_mpnn_model(code_root)

    if config_version:
        config_settings = load_config_settings(data_root, uniprot_id, config_version)
        starting_seq = config_settings.get('starting_seq', reference_seq)
        starting_seq = trim_seqs([starting_seq], initial_seq, reference_seq)[0]
        recode_positions = config_settings['redesign_positions_1_based']
        start_index = reference_seq.find(initial_seq)
        recode_positions = sorted([i - start_index for i in recode_positions])
    else:
        starting_seq = initial_seq
        recode_positions = None

    dataset, chain_id_dict, fixed_positions, omit_AA, positions_to_redesign = prepare_multichain_dataset(
        pdb_path, letter_to_redesign, target_chain, fixed_chains_list, include_contacts=True,
        redesign_radius=redesign_radius, top_to_take=top_to_take, include_neighbors=include_neighbors,
        starting_seq=starting_seq, recode_positions=recode_positions)

    mpnn_seqs, mpnn_scores = compute_mpnn_seqs(pdb_path, mpnn_model, mpnn_designs_num, dataset, chain_id_dict,
        fixed_positions, omit_AA, temp=temp, override_seq=starting_seq)

    return [{'uniprot_id': uniprot_id, 'gene': gene_name, 'seq': seq, 'include_neighbors': include_neighbors, 'temp': temp}
            for seq in mpnn_seqs]

def load_target_genes(kind, data_root):
    if kind == 'fig2_50':
        fig2_genes = pd.read_excel(f'{data_root}/designs need for figure2D.xlsx')
        target_genes = pd.DataFrame().assign(
            gene=fig2_genes['Name'],
            uniprot_id=fig2_genes['Uniprot'],
            full_seq=fig2_genes['WT']
        )
    if kind == 'ribo_59':
        hazel_genes = pd.read_csv(f'{data_root}/59 genes WT vs itov (HZ 10-24-22).csv')
        target_genes = pd.DataFrame().assign(gene = hazel_genes.gene_name, uniprot_id = hazel_genes.Uniprot_ID, full_seq = hazel_genes.WT_aa)
    if kind == 'tf_35':
        tf_genes = pd.read_excel(f'{data_root}/translation_transcription factor_AA ligase genes.xlsx')
        target_genes = pd.DataFrame().assign(
            gene=tf_genes['Name'],
            uniprot_id=tf_genes['Uniprot'],
            full_seq=tf_genes['WT']
        )
    if kind == 'essential_163':
        essential_genes = pd.read_csv(f'{data_root}/20200823_WT_ESS_HEG_mini.csv')
        target_genes = pd.DataFrame().assign(gene = essential_genes.gene.str[1:], uniprot_id = essential_genes.UNIPROT_ID, full_seq = essential_genes.aa_seq)
    if kind == 'ht_44':
        reference_genes = pd.concat([load_target_genes('ribo_59', data_root), load_target_genes('essential_163', data_root)])
        reference_genes.drop_duplicates(subset=['gene'], inplace=True)
        ht_genes = pd.read_csv(f'{data_root}/Ht_gene_list.csv') # only one column: gene
        target_genes = reference_genes[reference_genes.gene.isin(ht_genes.gene)]
        assert len(ht_genes) == len(target_genes)
    if kind == 'calin_126':
        # read from fasta file: calin_library_wt_genes.fasta
        # format is >uniprotid_genename
        target_genes = pd.DataFrame().assign(gene = [], uniprot_id = [], full_seq = [])
        for record in SeqIO.parse(f'{data_root}/calin_library_wt_genes.fasta', "fasta"):
            gene = record.id.split('_')[1]
            uniprot_id = record.id.split('_')[0]
            target_genes = target_genes.append({'gene': gene, 'uniprot_id': uniprot_id, 'full_seq': str(record.seq)}, ignore_index=True)

    return target_genes

def load_designs(fasta_file, lib):
    designs = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        gene = record.id.split('_')[0]
        designs.append([gene, lib, str(record.seq)])
    return pd.DataFrame(designs, columns=['gene', 'lib', 'seq'])

def trim_seqs(seqs, initial_seq, reference_seq):
    ofsset = reference_seq.find(initial_seq)
    return [seq[ofsset:ofsset+len(initial_seq)] for seq in seqs]

def load_config_settings(data_root, uniprot_id, version):
    # load config.yaml from data_root/Recode_AF_MPNN_designs/<version> folder
    # yaml format:
    # designs:
    #- uniprot_id: P0AG48
    #  starting_seq: MYAVFQSGGKQHRVSEGQTVRLEKLDAATGETVEFAEVLMVANGEEVKVGVPFVDGGVVKAEVVAHGRGEKVKVVKFRRRKHYRKQQGHRQWFTDVKVTGVSA
    #  redesign_positions_1_based: [27, 49, 59]
    #  other setings

    # return a dict with settings for the given uniprot_id
    config_path = path.join(data_root, f"output/Recode_AF_MPNN_designs/{version}/config.yaml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for d in config['designs']:
        if d['uniprot_id'] == uniprot_id:
            return d
    return None
    
def design_dir(data_root, version):
    if version:
        return path.join(data_root, f"Recode_AF_MPNN_designs/{version}")
    else:
        return path.join(data_root, f"Recode_AF_MPNN_designs")

def score_designs(data_root, code_root, gene_name, uniprot_id, reference_seq, letter_to_redesign, redesign_radius, top_to_take, mpnn_designs_num,
    multimer=False, single_chain=False, config_version=None, llm_designs=True, method='mpnn'):
    pdb_name = f"{uniprot_id}_nearby_protein_{letter_to_redesign}"
    pdb_path = path.join(data_root, f"{pdb_name}.pdb")
    target_chain = "X"
    fixed_chains_list, initial_seq = extract_chains_ids(pdb_path, target_chain)
    # remove 'X' from fixed chains
    fixed_chains_list = [chain for chain in fixed_chains_list if chain != target_chain]

    mpnn_model = init_mpnn_model(code_root)
    esm_model_name = "esm2_t33_650M_UR50D"
    esm_model, esm_alphabet = load_esm(esm_model_name, 0)

    if config_version:
        config_settings = load_config_settings(data_root, uniprot_id, config_version)
        af_design_path = path.join(data_root, f"output/Recode_AF_MPNN_designs/{config_version}")

        starting_seq = config_settings['starting_seq'] if 'starting_seq' in config_settings else reference_seq    
        starting_seq = trim_seqs([starting_seq], initial_seq, reference_seq)[0]
        dlog(starting_seq)
        recode_positions = config_settings['redesign_positions_1_based']
        
        start_index = reference_seq.find(initial_seq)
        recode_positions = sorted([i - start_index for i in recode_positions])
    else:
        af_design_path = path.join(data_root, f"output/Recode_AF_MPNN_designs")
        starting_seq = initial_seq
        recode_positions = None
        config_settings = None
    
    if method == 'afdesign_mpnn_bias':
        try:
            af_mpnn_bias_designs = pd.read_csv(path.join(af_design_path, 'designs_af_mpnn_bias.csv'))
        except FileNotFoundError:
            af_mpnn_bias_designs = pd.DataFrame(columns=['uniprot_id', 'redesign_radius', 'seq', 'include_neighbors', 'recode_positions'])
    else:
        af_mpnn_bias_designs = pd.DataFrame(columns=['uniprot_id', 'redesign_radius', 'seq', 'include_neighbors', 'recode_positions'])

    if method == 'mpnn' and llm_designs:
        lib_designs = pd.concat([load_designs(path.join(data_root, "output/recode_designs/ecoli_ribosomal_ESM1b_autoregressive_I_to_else.fasta"), 'EAI'),
                            load_designs(path.join(data_root, "output/recode_designs/ecoli_ribosomal_ESM1b_simultaneous_I_to_else.fasta"), 'ESI')])

    if method == 'mpnn':
        try:
            mpnn_designs_df = pd.read_csv(path.join(data_root, 'output/recode_designs/designs_mpnn.csv'))
            mpnn_designs_df = mpnn_designs_df[mpnn_designs_df.uniprot_id == uniprot_id]
            mpnn_seqs = mpnn_designs_df[mpnn_designs_df.include_neighbors == True].seq.values
            mpnn_seqs_no_n = mpnn_designs_df[mpnn_designs_df.include_neighbors == False].seq.values
            mpnn_seqs_no_n_high_temp = []
        except FileNotFoundError:
            mpnn_seqs, mpnn_seqs_no_n, mpnn_seqs_no_n_high_temp = [], [], []
    else:
        mpnn_seqs, mpnn_seqs_no_n, mpnn_seqs_no_n_high_temp = [], [], []

    # Setup dataset for MPNN scoring
    dataset, chain_id_dict, fixed_positions, omit_AA, positions_to_redesign = prepare_multichain_dataset(
        pdb_path, letter_to_redesign, target_chain, fixed_chains_list, include_contacts=True,
        redesign_radius=redesign_radius, top_to_take=top_to_take, include_neighbors=False,
        starting_seq=starting_seq, recode_positions=recode_positions)

    if method == 'afdesign_mpnn_bias':
        filtered_af_mpnn_bias_designs = af_mpnn_bias_designs[(af_mpnn_bias_designs.uniprot_id == uniprot_id)]
        if recode_positions:
            filtered_af_mpnn_bias_designs = filtered_af_mpnn_bias_designs[
                filtered_af_mpnn_bias_designs.recode_positions == ",".join([str(i) for i in recode_positions])]

        af_design_mpnn_bias_seqs = filtered_af_mpnn_bias_designs.seq.values
        af_design_mpnn_bias_seqs = [seq for seq in af_design_mpnn_bias_seqs if len(seq) == len(initial_seq)]

        filtered_af_mpnn_bias_no_n_designs = af_mpnn_bias_designs[(af_mpnn_bias_designs.uniprot_id == uniprot_id) &
                                    (af_mpnn_bias_designs.include_neighbors == False)]
        if recode_positions:
            filtered_af_mpnn_bias_no_n_designs = filtered_af_mpnn_bias_no_n_designs[
                filtered_af_mpnn_bias_no_n_designs.recode_positions == ",".join([str(i) for i in recode_positions])]

        af_design_mpnn_bias_no_n_seqs = filtered_af_mpnn_bias_no_n_designs.seq.values
        af_design_mpnn_bias_no_n_seqs = [seq for seq in af_design_mpnn_bias_no_n_seqs if len(seq) == len(initial_seq)]
    else:
        af_design_mpnn_bias_seqs = []
        af_design_mpnn_bias_no_n_seqs = []

    seqs_by_lib = {'WT': [initial_seq]}

    if method == 'mpnn':
        seqs_by_lib.update({
            'MPNN': mpnn_seqs,
            'MPNN_no_n': mpnn_seqs_no_n,
            'MPNN_no_n_high_temp': mpnn_seqs_no_n_high_temp
        })
        if llm_designs:
            seqs_by_lib['llm_designs'] = trim_seqs(lib_designs[lib_designs.gene == gene_name].seq.values, initial_seq, reference_seq)
    elif method == 'afdesign_mpnn_bias':
        seqs_by_lib.update({
            'afdesign_mpnn_bias': af_design_mpnn_bias_seqs,
            'afdesign_mpnn_bias_no_n': af_design_mpnn_bias_no_n_seqs
        })

    score_methods = ['af2rank', 'blosum', 'mpnn', 'esm']

    protein_scores = compute_protein_scores(data_root, seqs_by_lib, score_methods, reference_seq, uniprot_id,
        letter_to_redesign=letter_to_redesign,
        esm_model=esm_model, esm_alphabet=esm_alphabet, esm_model_name=esm_model_name,
        mpnn_model=mpnn_model, mpnn_dataset=dataset, mpnn_chain_id_dict=chain_id_dict, multimer=multimer, single_chain=single_chain)
    protein_scores['initial_seq'] = initial_seq
    return protein_scores

def save_designs_fasta(final_designs_df, file_prefix, lib_name, letter_to_redesign, target_genes):
    # finally, write out to fasta file "ht_mpnn_{letter_to_redesign}_{date}.fasta"
    # seq description is "{gene}_{uniprot_id}_mpnn_{letter_to_redesign}_{index}"
    # index is 1, 2, etc depending on number of designs for that uniprot_id
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    records = []
    for uniprot_id in final_designs_df.uniprot_id.unique():
        gene = target_genes[target_genes.uniprot_id == uniprot_id].gene.values[0]
        reference_seq = target_genes[target_genes.uniprot_id == uniprot_id].full_seq.values[0]
        designs = final_designs_df[final_designs_df.uniprot_id == uniprot_id]
        for i, (_, design) in enumerate(designs.iterrows()):
            # splice design seq with reference seq
            spliced_seq = splice_reference_seq(design.seq, design.initial_seq, reference_seq, letter_to_redesign)
            seq = Seq(spliced_seq)
            description = f"{gene}_{uniprot_id}_{lib_name}_{letter_to_redesign}_{i+1}"        
            record = SeqRecord(seq, id=description, description='')
            records.append(record)
    SeqIO.write(records, f"{file_prefix}_{letter_to_redesign}_{date}.fasta", "fasta")

@memory.cache(ignore=['mpnn_model'])
def mpnn_probs_bits(mpnn_model, pdb_path, ref_seq, conditional_probs_only, temperature=0.1, exclude_letter=None, include_aa_column=False):
    target_chain = 'X'
    fixed_chains_list, initial_seq = extract_chains_ids(pdb_path, target_chain)

    # remove 'X' from fixed_chains_list
    fixed_chains_list = [x for x in fixed_chains_list if x != target_chain]
    dataset, chain_id_dict, fixed_positions, omit_AA, positions_to_redesign = prepare_multichain_dataset(pdb_path, exclude_letter, 
                                                                                target_chain, fixed_chains_list,
                                                                                include_contacts=True, 
                                                                                redesign_radius=0,
                                                                                top_to_take=0,
                                                                                include_neighbors=False,
                                                                                starting_seq=initial_seq,
                                                                                recode_positions=None
                                                                                )
    if conditional_probs_only:
        score, probs = generate_seq(mpnn_model, dataset, chain_id_dict, fixed_positions_dict=fixed_positions, conditional_probs_only=True)
        # apply softmax to probs
        softmax_probs = torch.nn.functional.softmax(probs, dim=2)

    else:
        score, probs, _ = generate_seq(mpnn_model, dataset, chain_id_dict, fixed_positions_dict=fixed_positions, temperature=temperature)
        # turn probs back from numpy into torch
        softmax_probs = torch.tensor(probs)
    # probs shape is [1, <num_residues>, 21]
    # chop to the len of the first chain
        
    if exclude_letter is not None:
        exclude_idx = mpnn_alphabet.index(exclude_letter)
        softmax_probs[:, :, exclude_idx] = 0
        # renormalize
        softmax_probs = softmax_probs / (softmax_probs.sum(dim=-1, keepdim=True) + 1e-6)
        
    softmax_probs = softmax_probs[:, :len(initial_seq), :].cpu()

    offset = ref_seq.find(initial_seq)
    # pad probs with zeros if offset > 0
    if offset > 0:
        softmax_probs = torch.cat([torch.zeros(1, offset, 21), softmax_probs], dim=1)

    # pad from the back too
    if len(ref_seq) > softmax_probs.shape[1]:
        softmax_probs = torch.cat([softmax_probs, torch.zeros(1, len(ref_seq) - softmax_probs.shape[1], 21)], dim=1)

    conditional_probs_by_name = unpack_probs_by_name(softmax_probs, mpnn_alphabet)

    # conditional probs by name is a dict { AA_name: torch tensor of [1, <num_residues>] }
    # transform this to a dataframe with <num_residues> rows and 21 columns
    # with columns being AA names and values being the probabilities

    bits = pd.DataFrame({k: v[0].numpy() for k, v in conditional_probs_by_name.items()})

    if include_aa_column:
        # add column named "AA" and fill it with the reference sequence
        bits['AA'] = list(ref_seq)

    assert len(bits) == len(ref_seq), f"len(bits) = {len(bits)}, len(ref_seq) = {len(ref_seq)}"
    return bits

@memory.cache(ignore=['esm_model','esm_alphabet'])
def esm_probs_bits_2(esm_model, esm_model_name, esm_alphabet, seq, letter_to_replace='I', exclude_letter= None, include_aa_column=False):
    aa_alphabet="ACDEFGHIKLMNPQRSTVWY"
    # replace I with "<MASK>" in wt_seq
    masked_seq = seq.replace(letter_to_replace, '<mask>')
    logprobs = compute_esm_seq_logprobs(esm_model, esm_model_name, esm_alphabet, masked_seq, "cuda:0")
    logprobs_filtered = logprobs.cpu().detach().clone()[1:-1]
    # fill all values with non-AA tokens with -10000
    for i, tok in enumerate(esm_alphabet.all_toks):
        if tok in aa_alphabet:
            continue
        else:
            logprobs_filtered[:, i] = -10000

    # softmax to get the probs
    probs = torch.nn.functional.softmax(logprobs_filtered, dim=-1)

    if exclude_letter is not None:
       exclude_idx = esm_alphabet.tok_to_idx[exclude_letter]
       probs[:, exclude_idx] = 0
       # renormalize
       probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-6)
 
    probs_dict = {}
    for i, tok in enumerate(esm_alphabet.all_toks):
        if tok in aa_alphabet:
            probs_dict[tok] = probs[:, i].numpy()

    bits = pd.DataFrame({k: v for k, v in sorted(probs_dict.items())})

    if include_aa_column:
        # add column named "AA" and fill it with the reference sequence
        bits['AA'] = list(seq)

    return bits