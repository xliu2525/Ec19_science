import re
import os
from os import path

import numpy as np

from pdb import extract_all_chains, positions_hash

from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af import mk_af_model, clear_mem

from mpnn import generate_seq, unpack_probs_by_name, pack_probs_by_name, alphabet as mpnn_alphabet, init_mpnn_model
from recode_structure import prepare_multichain_dataset, load_config_settings, trim_seqs
from log import dlog

from multiprocessing import get_logger

logger = get_logger()


def prepare_af_bias_start_seq(unconditional_probs, af_model, mpnn_bias_temp, recode_positions_1_based, starting_seq):
    max_seq = np.argmax(unconditional_probs, axis=1)

    designed_chain_len = af_model._lengths[0]

    bias = np.zeros((len(af_model._wt_aatype), 20), dtype=np.float32)

    # convert starting seq to aa constaints using residue_constants.restype_order
    starting_seq_indices = af_model._wt_aatype.copy()
    starting_seq_indices[0:len(starting_seq)] = [residue_constants.restype_order[aa] for aa in starting_seq]

    bias[np.arange(len(bias)), starting_seq_indices] = 1e8

    recode_positions_0_based = recode_positions_1_based - 1
    bias[recode_positions_0_based] = unconditional_probs[recode_positions_0_based] / mpnn_bias_temp
    return bias, max_seq

def check_completed_exists(dirname, uniprot_id, letter_to_redesign, mpnn_bias_temp, seed, recode_positions, initial_seq):
    # Example filename: P76156_I_mpnn_bias_0.001_seed_1_positions_27_dgram_cce_1.988.pdb
    fname_desc_bit = construct_filename_desc(uniprot_id, letter_to_redesign, mpnn_bias_temp, seed, recode_positions)
    regex = re.compile(f"{fname_desc_bit}_dgram_cce_(.*).pdb")

    print(f"Checking for completed designs in {dirname}, with fname_desc_bit={fname_desc_bit}")

    for filename in os.listdir(dirname):
        if filename.startswith(uniprot_id) and filename.endswith(".pdb"):
            # check if filename matches regex and extract dgram_cce score
            match = regex.match(filename)
            if match:
                dgram_cce = float(match.group(1))
                print("Found completed design with dgram_cce score of", dgram_cce)
                # load pdb file and check if length of sequence matches
                pdb_path = path.join(dirname, filename)
                chains = extract_all_chains(pdb_path)
                designed_seq = chains['A']
                if len(designed_seq) != len(initial_seq):
                    print(f"Length of designed sequence ({len(designed_seq)}) does not match length of initial sequence ({len(initial_seq)})")
                    continue

                # find positions that are different between initial and designed seq
                diff_letters = [i+1 for i in range(len(designed_seq)) if designed_seq[i] != initial_seq[i]]
                # if diff_letters is not a subset of recode_positions, then continue
                if not set(diff_letters).issubset(set(recode_positions)):
                    print(f"Positions that differ between initial and designed seq ({diff_letters}) are not a subset of recode_positions ({recode_positions})")
                    continue

                # if all conditions are met, return dgram_cce score
                print(f"Returning dgram_cce score of {dgram_cce} for completed design")
                return designed_seq, dgram_cce

    print("No completed design found")
    return None, None

def construct_filename_desc(uniprot_id, letter_to_redesign, mpnn_bias_temp, seed, recode_positions):
    recode_positions_hash = positions_hash(recode_positions)
    return f"{uniprot_id}_{letter_to_redesign}_mpnn_bias_{mpnn_bias_temp}_seed_{seed}_positions_{len(recode_positions)}_{recode_positions_hash}"

def design_af_with_mpnn_bias(code_root, data_root, uniprot_id, letter_to_redesign, reference_seq, mpnn_bias_temp, seed, include_neighbors, redesign_radius, top_to_take,
    quick=False, skip_completed=True, config_version=None):
    logger.critical(f"Designing {uniprot_id}, letter_to_redesign={letter_to_redesign}, mpnn_bias_temp={mpnn_bias_temp}, seed={seed}, quick={quick}")
    logger.critical(f"include_neighbors={include_neighbors}, redesign_radius={redesign_radius}, top_to_take={top_to_take}")
    logger.critical(f"Config: {config_version}")
    model_mpnn = init_mpnn_model(code_root)

    pdb_path = path.join(data_root, f"ribo/{uniprot_id}_nearby_protein_{letter_to_redesign}.pdb")
    use_templates = True
    designed_chain = 'X'

    chains = extract_all_chains(pdb_path) # returns dict of chain_id -> chain
    fixed_chains = [c for c in chains if c != designed_chain]

    chain_len_sum = sum([len(chains[c]) for c in chains])
    if chain_len_sum > 500:
        fixed_chains = []

    initial_seq = chains[designed_chain]

    if config_version:
        config_settings = load_config_settings(data_root, uniprot_id, config_version)
        dlog(config_settings,v=1)
        af_design_path = path.join(data_root, f"Recode_AF_MPNN_designs/{config_version}")    
        starting_seq = config_settings['starting_seq'] if 'starting_seq' in config_settings else reference_seq    
        starting_seq = trim_seqs([starting_seq], initial_seq, reference_seq)[0]
        recode_positions = config_settings['redesign_positions_1_based']

        start_index = reference_seq.find(initial_seq)
        recode_positions = sorted([i - start_index for i in recode_positions])
    else:
        af_design_path = path.join(data_root, f"Recode_AF_MPNN_designs")
        starting_seq = initial_seq
        recode_positions = None

    letter_to_redesign = 'I'
    dataset, chain_id_dict, fixed_positions_dict, omit_AA_dict, recode_positions = \
        prepare_multichain_dataset(pdb_path, letter_to_redesign, designed_chain, fixed_chains, include_contacts=True,
        include_neighbors=include_neighbors, redesign_radius=redesign_radius, top_to_take=top_to_take,
        starting_seq=starting_seq, recode_positions=recode_positions)

    if skip_completed:
        completed_seq, completed_dgram_cce = check_completed_exists(af_design_path, uniprot_id, letter_to_redesign, 
                        mpnn_bias_temp, seed, recode_positions, starting_seq)
        if completed_seq is not None:
            return completed_seq, completed_dgram_cce, recode_positions

    _, probs = generate_seq(model_mpnn, dataset, chain_id_dict, fixed_positions_dict=fixed_positions_dict, 
                  omit_AA_dict=omit_AA_dict, conditional_probs_only=True)
    conditional_probs_by_name = unpack_probs_by_name(probs.cpu(), mpnn_alphabet)

    clear_mem()
    best_metric = "dgram_cce"
    af_design_model = mk_af_model(protocol="fixbb", best_metric=best_metric, use_templates=use_templates)

    chain_list = [designed_chain]
    if fixed_chains:
        chain_list += fixed_chains
    af_design_model.prep_inputs(pdb_filename=pdb_path, chain=",".join(chain_list))

    seq_logits = pack_probs_by_name(conditional_probs_by_name, residue_constants.restypes)[0]

    bias, start_seq = prepare_af_bias_start_seq(seq_logits, af_design_model, mpnn_bias_temp, np.array(recode_positions), starting_seq)

    bias[:, residue_constants.restype_order[letter_to_redesign]] = -1e8

    af_design_model.restart(seed=seed)
    af_design_model.set_seq(seq=start_seq, bias=bias)
    if use_templates:
        af_design_model.set_opt("template",dropout=0.15)
    af_design_model.set_weights(pae=0.01,plddt=0.01)
    if quick:
        af_design_model.design_3stage(2,2,2)
    else:
        af_design_model.design_3stage()

    best = af_design_model._tmp['best']
    designed_chain_len = af_design_model._lengths[0]
    seqid = np.mean(best['aux']['aatype'][:designed_chain_len] == af_design_model._wt_aatype[:designed_chain_len])
    logger.critical(f"{best_metric}: {best['metric']:.3f}, designed chain seqid: {seqid:.3f}")
    final_seq = "".join([residue_constants.restypes[i] for i in best['aux']['aatype'][:designed_chain_len]])

    # save pdb
    if not quick:
        os.makedirs(af_design_path, exist_ok=True)
        fname_desc_bit = construct_filename_desc(uniprot_id, letter_to_redesign, mpnn_bias_temp, seed, recode_positions)
        fname_metric_bit = f"{best_metric}_{best['metric']:.3f}"
        fname = f"{fname_desc_bit}_{fname_metric_bit}.pdb"
        logger.critical(f"Saving {fname}")
        af_design_model.save_pdb(path.join(af_design_path, fname))
    return final_seq, best['metric'], recode_positions

