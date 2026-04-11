import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
from os import path
import sys
import numpy as np
import re
import time 

from log import dlog
from common import memory

import os
# Get path to project root (EC19_Science)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Add ProteinMPNN to Python path
sys.path.append(os.path.join(ROOT, "ProteinMPNN"))

from protein_mpnn_utils import _scores, _S_to_seq, tied_featurize, parse_PDB, StructureDatasetPDB, ProteinMPNN

def init_mpnn_model(root, gpu_id=0):
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available()) else "cpu")
    #v_48_010=version with 48 edges 0.10A noise
    model_name = "v_48_020" #@param ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]
    backbone_noise=0.00               # Standard deviation of Gaussian noise to add to backbone atoms

    path_to_model_weights=path.join(root, 'ProteinMPNN/vanilla_model_weights')          
    hidden_dim = 128
    num_layers = 3 
    model_folder_path = path_to_model_weights
    if model_folder_path[-1] != '/':
        model_folder_path = model_folder_path + '/'
    checkpoint_path = model_folder_path + f'{model_name}.pt'

    checkpoint = torch.load(checkpoint_path, map_location=device) 
    print('Number of edges:', checkpoint['num_edges'])
    noise_level_print = checkpoint['noise_level']
    print(f'Training noise level: {noise_level_print}A')
    model_mpnn = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=backbone_noise, k_neighbors=checkpoint['num_edges'])
    model_mpnn.to(device)
    model_mpnn.load_state_dict(checkpoint['model_state_dict'])
    model_mpnn.eval()
    print("Model loaded")
    return model_mpnn

def make_tied_positions_for_homomers(pdb_dict_list):
    my_dict = {}
    for result in pdb_dict_list:
        all_chain_list = sorted([item[-1:] for item in list(result) if item[:9]=='seq_chain']) #A, B, C, ...
        tied_positions_list = []
        chain_length = len(result[f"seq_chain_{all_chain_list[0]}"])
        for i in range(1,chain_length+1):
            temp_dict = {}
            for j, chain in enumerate(all_chain_list):
                temp_dict[chain] = [i] #needs to be a list
            tied_positions_list.append(temp_dict)
        my_dict[result['name']] = tied_positions_list
    return my_dict



homomer = False #@param {type:"boolean"}
designed_chain = "A" #@param {type:"string"}
fixed_chain = "" #@param {type:"string"}

#@markdown - specified which chain(s) to design and which chain(s) to keep fixed. 
#@markdown   Use comma:`A,B` to specifiy more than one chain

#chain = "A" #@param {type:"string"}
#pdb_path_chains = chain
##@markdown - Define which chain to redesign

#@markdown ### Design Options
num_seqs = 1 #@param ["1", "2", "4", "8", "16", "32", "64"] {type:"raw"}
num_seq_per_target = num_seqs

#@markdown - Sampling temperature for amino acids, T=0.0 means taking argmax, T>>1.0 means sample randomly.
sampling_temp = "0.1" #@param ["0.0001", "0.1", "0.15", "0.2", "0.25", "0.3", "0.5"]


base_folder="."
save_score=0                      # 0 for False, 1 for True; save score=-log_prob to npy files
save_probs=0                      # 0 for False, 1 for True; save MPNN predicted probabilites per position
score_only=0                      # 0 for False, 1 for True; score input backbone-sequence pairs
conditional_probs_only=0          # 0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)
conditional_probs_only_backbone=0 # 0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)
unconditional_probs_only=0

batch_size=1                      # Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory
max_length=20000                  # Max sequence length
    
out_folder='.'                    # Path to a folder to output sequences, e.g. /home/out/
jsonl_path=''                     # Path to a folder with parsed pdb into jsonl
omit_AAs='X'                      # Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.
   
pssm_multi=0.0                    # A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions
pssm_threshold=0.0                # A value between -inf + inf to restric per position AAs
pssm_log_odds_flag=0               # 0 for False, 1 for True
pssm_bias_flag=0                   # 0 for False, 1 for True


##############################################################

folder_for_outputs = out_folder

NUM_BATCHES = num_seq_per_target//batch_size
dlog(NUM_BATCHES, num_seq_per_target, batch_size)
BATCH_COPIES = batch_size
temperatures = [float(item) for item in sampling_temp.split()]
omit_AAs_list = omit_AAs
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)

chain_id_dict = None
fixed_positions_dict = None
pssm_dict = None
omit_AA_dict = None
bias_AA_dict = None
tied_positions_dict = None
bias_by_res_dict = None
bias_AAs_np = np.zeros(len(alphabet))

def load_data_from_pdb_path(pdb_path, designed_chain="A", fixed_chain=""):
    if designed_chain == "":
        designed_chain_list = []
    else:
        designed_chain_list = re.sub("[^A-Za-z]+",",", designed_chain).split(",")

    if fixed_chain == "":
        fixed_chain_list = []
    else:
        fixed_chain_list = re.sub("[^A-Za-z]+",",", fixed_chain).split(",")
    chain_list = list(set(designed_chain_list + fixed_chain_list))
    dlog(pdb_path, chain_list)
    pdb_dict_list = parse_PDB(pdb_path, input_chain_list=chain_list)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=max_length)

    chain_id_dict = {}
    chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)

    for chain in chain_list:
      l = len(pdb_dict_list[0][f"seq_chain_{chain}"])
      print(f"Length of chain {chain} is {l}")
      
    if homomer:
        tied_positions_dict = make_tied_positions_for_homomers(pdb_dict_list)
    else:
        tied_positions_dict = None

    return dataset_valid, chain_id_dict, chain_list

omit_AA_mask = None
X = None
chain_M = None
chain_enconding_all = None

def generate_seq(model_mpnn, dataset_valid, chain_id_dict, override_seqs=None, tied_positions_dict=None,
                 score_only=False, conditional_probs_only=False, conditional_probs_only_backbone=False,
                 unconditional_probs_only=False, omit_AA_dict=None, fixed_positions_dict=None, temperature=0.1):
    temperatures = [temperature]
    with torch.no_grad():
        # extract device from model
        device = next(model_mpnn.parameters()).device
        #print('Generating sequences...')
        for ix, protein in enumerate(dataset_valid):
            score_list = []
            all_probs_list = []
            all_log_probs_list = []
            S_sample_list = []
            protein = copy.deepcopy(protein)

            if override_seqs:
                for override_chain_key, override_seq in override_seqs.items():
                    full_chain_key = f"seq_chain_{override_chain_key}"
                    old_protein_seq = protein[full_chain_key]
                    protein[full_chain_key] = override_seq
                    assert(old_protein_seq in protein['seq'])
                    protein['seq'] = protein['seq'].replace(old_protein_seq, override_seq)
                    
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]

            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, \
            masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
            tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, \
            tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, 
                                    tied_positions_dict, pssm_dict, bias_by_res_dict)
            
            pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float() #1.0 for true, 0.0 for false
            name_ = batch_clones[0]['name']
            gen = torch.Generator(device=X.device)
            gen.manual_seed(42)
            if score_only:
                structure_sequence_score_file = base_folder + '/score_only/' + batch_clones[0]['name'] + '.npy'
                native_score_list = []
                for j in range(NUM_BATCHES):
                    randn_1 = torch.randn(chain_M.shape, device=X.device, generator=gen)
                    log_probs = model_mpnn(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                    mask_for_loss = mask*chain_M*chain_M_pos
                    scores = _scores(S, log_probs, mask_for_loss)
                    native_score = scores.cpu().data.numpy()
                    native_score_list.append(native_score)
                native_score = np.concatenate(native_score_list, 0)
                ns_mean = native_score.mean()
                ns_mean_print = np.format_float_positional(np.float32(ns_mean), unique=False, precision=4)
                ns_std = native_score.std()
                ns_std_print = np.format_float_positional(np.float32(ns_std), unique=False, precision=4)
                ns_sample_size = native_score.shape[0]
                #np.save(structure_sequence_score_file, native_score)
                print(f'Score for {name_}, mean: {ns_mean_print}, std: {ns_std_print}, sample size: {ns_sample_size}')
                return ns_mean
            elif conditional_probs_only:
                print(f'Calculating conditional probabilities for {name_}')
                conditional_probs_only_file = base_folder + '/conditional_probs_only/' + batch_clones[0]['name']
                log_conditional_probs_list = []
                for j in range(NUM_BATCHES):
                    randn_1 = torch.randn(chain_M.shape, device=X.device, generator=gen)
                    log_conditional_probs = model_mpnn.conditional_probs(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1, conditional_probs_only_backbone)
                    log_conditional_probs_list.append(log_conditional_probs.cpu().numpy())
                if NUM_BATCHES != 1:
                    raise ValueError(f"Expecting NUM_BATCHES to be 1, got {NUM_BATCHES} instead")
                concat_log_p = np.concatenate(log_conditional_probs_list, 0) #[B, L, 21]
                mask = chain_M*chain_M_pos*mask
                score = _scores(S, log_conditional_probs, mask)
                return score, log_conditional_probs
            elif unconditional_probs_only:
                print(f'Calculating sequence unconditional probabilities for {name_}')
                unconditional_probs_only_file = base_folder + '/unconditional_probs_only/' + batch_clones[0]['name']
                log_unconditional_probs_list = []
                for j in range(NUM_BATCHES):
                    log_unconditional_probs = model_mpnn.unconditional_probs(X, mask, residue_idx, chain_encoding_all)
                    log_unconditional_probs_list.append(log_unconditional_probs.cpu().numpy())
                concat_log_p = np.concatenate(log_unconditional_probs_list, 0) #[B, L, 21]
                mask_out = (chain_M*chain_M_pos*mask)[0,].cpu().numpy()
                return concat_log_p
                #np.savez(unconditional_probs_only_file, log_p=concat_log_p, S=S[0,].cpu().numpy(), mask=mask[0,].cpu().numpy(), design_mask=mask_out)
            else:
                randn_1 = torch.randn(chain_M.shape, device=X.device)
                log_probs = model_mpnn(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                mask_for_loss = mask*chain_M*chain_M_pos
                scores = _scores(S, log_probs, mask_for_loss)
                native_score = scores.cpu().data.numpy()
                # Generate some sequences
                ali_file = base_folder + '/seqs/' + batch_clones[0]['name'] + '.fa'
                print(f'Generating sequences for: {name_}')
                t0 = time.time()
                for temp in temperatures:
                    for j in range(NUM_BATCHES):
                        randn_2 = torch.randn(chain_M.shape, device=X.device)
                        if tied_positions_dict == None:
                            sample_dict = model_mpnn.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag), bias_by_res=bias_by_res_all)
                            S_sample = sample_dict["S"] 
                        else:
                            sample_dict = model_mpnn.tied_sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag), tied_pos=tied_pos_list_of_lists_list[0], tied_beta=tied_beta, bias_by_res=bias_by_res_all)
                        # Compute scores
                            S_sample = sample_dict["S"]
                        log_probs = model_mpnn(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, 
                                        use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])
                        mask_for_loss = mask*chain_M*chain_M_pos
                        scores = _scores(S_sample, log_probs, mask_for_loss)
                        scores = scores.cpu().data.numpy()
                        all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                        all_log_probs_list.append(log_probs.cpu().data.numpy())
                        S_sample_list.append(S_sample.cpu().data.numpy())
                        for b_ix in range(BATCH_COPIES):
                            masked_chain_length_list = masked_chain_length_list_list[b_ix]
                            masked_list = masked_list_list[b_ix]
                            seq_recovery_rate = torch.sum(torch.sum(torch.nn.functional.one_hot(S[b_ix], 21)*torch.nn.functional.one_hot(S_sample[b_ix], 21),axis=-1)*mask_for_loss[b_ix])/torch.sum(mask_for_loss[b_ix])
                            seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                            score = scores[b_ix]
                            score_list.append(score)
                            native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                            if b_ix == 0 and j==0 and temp==temperatures[0]:
                                start = 0
                                end = 0
                                list_of_AAs = []
                                for mask_l in masked_chain_length_list:
                                    end += mask_l
                                    list_of_AAs.append(native_seq[start:end])
                                    start = end
                                native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                                l0 = 0
                                for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                    l0 += mc_length
                                    native_seq = native_seq[:l0] + '/' + native_seq[l0:]
                                    l0 += 1
                                sorted_masked_chain_letters = np.argsort(masked_list_list[0])
                                print_masked_chains = [masked_list_list[0][i] for i in sorted_masked_chain_letters]
                                sorted_visible_chain_letters = np.argsort(visible_list_list[0])
                                print_visible_chains = [visible_list_list[0][i] for i in sorted_visible_chain_letters]
                                native_score_print = np.format_float_positional(np.float32(native_score.mean()), unique=False, precision=4)
                                print('>{}, score={}, fixed_chains={}, designed_chains={}\n{}\n'.format(name_, native_score_print, print_visible_chains, print_masked_chains, native_seq)) #write the native sequence
                            start = 0
                            end = 0
                            list_of_AAs = []
                            for mask_l in masked_chain_length_list:
                                end += mask_l
                                list_of_AAs.append(seq[start:end])
                                start = end

                            seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                            l0 = 0
                            for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                l0 += mc_length
                                seq = seq[:l0] + '/' + seq[l0:]
                                l0 += 1
                            score_print = np.format_float_positional(np.float32(score), unique=False, precision=4)
                            seq_rec_print = np.format_float_positional(np.float32(seq_recovery_rate.detach().cpu().numpy()), unique=False, precision=4)

                            print('>T={}, sample={}, score={}, seq_recovery={}\n{}\n'.format(temp,b_ix,score_print,seq_rec_print,seq)) #write generated sequence
                if save_score:
                    score_file = base_folder + '/scores/' + batch_clones[0]['name'] + '.npy'
                    np.save(score_file, np.array(score_list, np.float32))
                if save_probs:
                    probs_file = base_folder + '/probs/' + batch_clones[0]['name'] + '.npz'
                    all_probs_concat = np.concatenate(all_probs_list)
                    all_log_probs_concat = np.concatenate(all_log_probs_list)
                    S_sample_concat = np.concatenate(S_sample_list)
                    np.savez(probs_file, probs=np.array(all_probs_concat, np.float32), log_probs=np.array(all_log_probs_concat, np.float32), S=np.array(S_sample_concat, np.int32), mask=mask_for_loss.cpu().data.numpy(), chain_order=chain_list_list)
                t1 = time.time()
                dt = round(float(t1-t0), 4)
                num_seqs = len(temperatures)*NUM_BATCHES*BATCH_COPIES
                total_length = X.shape[1]
                print(f'{num_seqs} sequences of length {total_length} generated in {dt} seconds')
                return seq, np.concatenate(all_probs_list), np.float32(score)

def unpack_probs_by_name(probs, alphabet):
    result = {}
    for i in range(20):
        #print(f"Putting {i} into letter {alphabet[i]}")
        result[alphabet[i]] = probs[:, :, i]
    return result

def pack_probs_by_name(probs_by_name, alphabet):
    result = None
    for name, v in probs_by_name.items():
        if result is None:
            #print(f"Creating result to be {v.shape + (20,)}")
            result = np.zeros(v.shape + (20,))
            
        #print(f"Unpacking {name} to {alphabet.index(name)}")
        result[:, :, alphabet.index(name)] = v
    return result

@memory.cache(ignore=['dataset','model_mpnn'])
def score_seq_mpnn(model_mpnn, seq, dataset, chain_id_dict, conditional_score=True):
    if conditional_score:       
        score, _ = generate_seq(model_mpnn, dataset, chain_id_dict, override_seqs={'X': seq}, conditional_probs_only=True)
        score = score.cpu().numpy()[0]
    else:
        score = generate_seq(model_mpnn, dataset, chain_id_dict, override_seqs={'X': seq}, score_only=True)
    return score

def score_pdb(pdb_path, dataset, conditional_score=True):
    pdb_dataset, _, _ = load_data_from_pdb_path(pdb_path, "A")
    seq = pdb_dataset[0]['seq']
    return score_seq_mpnn(seq, dataset, chain_id_dict, conditional_score=conditional_score)
                
@memory.cache(ignore=['dataset', 'model_mpnn'])
def compute_mpnn_seqs(protein_name, model_mpnn, seq_num, dataset, chain_id_dict, fixed_positions, omit_AA, 
        override_seq = None, seed=42, temp=0.1):
    torch.manual_seed(seed)
    mpnn_seqs = []
    mpnn_scores = []
    override_seqs = {'X': override_seq} if override_seq is not None else None


    for i in range(seq_num):
        print(f"Generating {i}/{seq_num-1}")
        mpnn_seq, _, mpnn_score = generate_seq(model_mpnn, dataset, chain_id_dict, omit_AA_dict=omit_AA, fixed_positions_dict=fixed_positions,
            temperature=temp, override_seqs=override_seqs)
        mpnn_seqs.append(mpnn_seq)
        mpnn_scores.append(mpnn_score)

    return mpnn_seqs, mpnn_scores