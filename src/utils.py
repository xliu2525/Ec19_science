import glob
import os
import pathlib
import itertools
import string

import numpy as np
import torch
import esm
from typing import Dict, List, Tuple, Set
from Bio import SeqIO, pairwise2

from mpnn import load_data_from_pdb_path


deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)


def read_fasta(path: str) -> Tuple[List[str], List[str]]:
    with open(path) as handle:
        records = [record for record in SeqIO.parse(handle, "fasta")]
        labels = [record.id for record in records]
        seqs = [str(record.seq) for record in records]
    return labels, seqs

def read_designs_fasta(path: str) -> Tuple[List[str], List[str]]:
    with open(path) as handle:
        records = [record for record in SeqIO.parse(handle, "fasta")]
        pdb_ids = [record.id.split('_')[0] for record in records]
        full_names = [record.id for record in records]
        seqs = [str(record.seq) for record in records]
    return full_names, pdb_ids, seqs

def load_sequences(file: str, alphabet):
    dataset = esm.FastaBatchedDataset.from_file(file)
    batches = dataset.get_batch_indices(toks_per_batch=4096, extra_toks_per_seq=1)
    dataloader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches)
    print(f"Read fasta file with {len(dataset)} sequences")
    return dataloader

def get_msa_filenames(msa_dir: str) -> List[str]:
    return glob.glob(os.path.join(msa_dir, '*.a3m'))

def map_uniprot_to_name_and_seq_from_fasta(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, 'r') as f:
        lines = f.readlines()
    names, uniprots, seqs = [], [], []

    for line in lines:
        if line.startswith('>'):
            name = line[1:].strip().split('_')[0]
            uniprot = line[1:].strip().split('_')[1]
            uniprots.append(uniprot)
            names.append(name)
        else:
            seqs.append(line.strip())

    return {i: {'name': j, 'seq': k} for (i, j, k) in zip(uniprots, names, seqs)}

def map_name_to_seq_from_fasta(path: str) -> Dict[str, str]:
    names, seqs = [], []
    with open(path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            gene_name = (record.id).split(" ")[0]
            names.append(gene_name)
            seqs.append(str(record.seq))
    return dict(zip(names, seqs))

def extract_name_and_uniprot_from_msa_filename(filename: str, mapping: Dict[str, Dict[str, str]])\
        -> Tuple[str, str]:
    uniprot = pathlib.Path(filename).stem.split('.')[0]
    name = mapping[uniprot]['name']
    return uniprot, name

def extract_name_from_msa_filename(filename: str) -> str:
    return pathlib.Path(filename).stem

def set_msa_nseq(seq: str) -> int:
    if len(seq) > 500:
        if len(seq) > 800:
            nseq = 128
        else:
            nseq = 256
    else:
        nseq = 384
    return nseq

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(infile: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions.
    The input file must be in a3m format (although we use the SeqIO fasta parser)
    for remove_insertions to work properly."""
    msa = [
        (record.description, remove_insertions(str(record.seq)))
        for record in itertools.islice(SeqIO.parse(infile, "fasta"), nseq)
    ]
    return msa

def check_recoded_sequences(designs: dict, residue: str):
    for label, seq in designs.items():
        try:
            assert residue not in seq
        except AssertionError:
            print(f"Residue {residue} present in {label} sequence.")

def save_designs_to_fasta(designs: dict, outpath: str):
    with open(outpath, 'w') as f:
        for label, seq in designs.items():
            f.write('>' + label + '\n')
            f.write(seq + '\n')

def compute_esm_seq_pseudo_log_likelihood(model: esm.model, alphabet: esm.data.Alphabet, seq: str, device: str):
    model = model.eval()
    torch.set_grad_enabled(False)
    batch_converter = alphabet.get_batch_converter()
    masked_logprobs = []
    data = [('protein', seq)]
    _, _, toks = batch_converter(data)

    for i in range(toks.size(-1) - 2):  # exclude start and end tokens
        j = i + 1
        tok_idx = toks[0][j]

        # [1, N_res]
        masked_toks = toks.clone()
        masked_toks[0][j] = alphabet.mask_idx
        logits = model(masked_toks.to(device))['logits'].cpu()

        # [1, N_res, N_vocab]
        logprobs = torch.log_softmax(logits[0, j], dim=-1)

        logprob = logprobs[tok_idx]
        masked_logprobs.append(logprob)
        del masked_toks

    return np.mean(masked_logprobs)

def compute_msa_seq_pseudo_log_likelihood(model: esm.model, alphabet: esm.data.Alphabet, seq: str, device: str,
                                          name: str, msa_dir: str):
    model = model.eval()
    torch.set_grad_enabled(False)
    use_uniprot_id = True

    batch_converter = alphabet.get_batch_converter()
    masked_logprobs = []

    if use_uniprot_id:
        msa_name = name.split('_')[1]
    else:
        msa_name = name.split('_')[0]
    filename = glob.glob(os.path.join(msa_dir, f'{msa_name}*.a3m'))[0]
    nseq = set_msa_nseq(seq)
    data = [read_msa(filename, nseq=nseq)]
    recoded_data = [[('protein', seq)] + data[0][1:]]
    _, _, msa_toks = batch_converter(recoded_data)

    for i in range(msa_toks.size(-1) - 1):  # exclude start token
        j = i + 1
        tok_idx = msa_toks[0, 0, j]

        # [1, N_seq, N_res]
        masked_toks = msa_toks.clone()
        masked_toks[0, 0, j] = alphabet.mask_idx

        # [1, N_seq, N_res, N_vocab]
        logits = model(masked_toks.to(device))['logits'].cpu()

        # [1, N_seq, N_vocab]
        logprobs = torch.log_softmax(logits[0, 0, j], dim=-1)

        logprob = logprobs[tok_idx]
        masked_logprobs.append(logprob)
        del masked_toks

    return np.mean(masked_logprobs)

def load_structure_data(structures_dir: str, uniprot: str):
    pdb_path = os.path.join(structures_dir, f"{uniprot}_nearby_protein_I.pdb")
    structure_data, _, _ = load_data_from_pdb_path(pdb_path, "X", "")
    return structure_data

def align_seq_from_structure_with_ref_seq(ref_seq: str, seq_to_align: str) -> np.ndarray:
    alignment = pairwise2.align.localxx(ref_seq, seq_to_align)
    return np.array(list(alignment[0][1]))

def get_aligned_residue_coords(aligned_seq: np.ndarray, structure_coords: List[List[float]]):
    """Extract coords align target seq with seq from structure to find missing residue coords."""
    print("aligned seq:", aligned_seq)
    aligned_res = aligned_seq != '-'  # [N_seq]
    aligned_coords = []
    start_idx = -1
    
    for bool_val in aligned_res:
        if bool_val:
            start_idx += 1
            aligned_coords.append(structure_coords[start_idx])
        else:
            # set missing residue coords to inf (won't be interacting)
            aligned_coords.append([float('inf'), float('inf'), float('inf')])
    # [N_seq, 3]
    return torch.tensor(aligned_coords)

def get_indices_of_interacting_residues(
        seq: str, coords: List[List[float]], mask: torch.tensor, min_dist_thresh: float, max_dist_thresh: float
        ) -> Set[int]:
        """Finds indices of residues that are within a distance <= dist_thres to target_residue."""

        # pairwise distances, [N_seq, N_seq] (symmetric tensor)
        pdist = torch.cdist(coords, coords)
        # [N_seq, N_seq]
        interaction_mask = ((pdist >= min_dist_thresh) & (pdist <= max_dist_thresh))

        # find residues that interact with residue to recode
        # [N_seq]
        # TODO: adapt not to use seq but only worst positions
        target_res_indices = set(torch.where(mask == 1)[0].tolist())

        # [N_seq, N_seq]
        row_mask = mask.repeat(len(seq), 1)
        # [N_seq, N_seq]
        col_mask = row_mask.transpose(0,1)

        # [N_seq, N_seq], mask for positions of residue to recode (symmetric)
        target_mask = torch.clip(row_mask + col_mask, max=1)

        interacting_res_indices = set(torch.where((interaction_mask.long() +  target_mask) == 2)[0].tolist())
        return interacting_res_indices - target_res_indices

def get_interacting_residues_from_structure(
        structures_dir: str, uniprot: str, seq: str, mask: torch.tensor, min_dist_thresh: float, max_dist_thresh: float
        ) -> Set[float]:
    structure_data = load_structure_data(structures_dir, uniprot)
    seq_from_structure = structure_data.data[0]['seq']
    unaligned_coords = structure_data.data[0]['coords_chain_X']['CA_chain_X']
    aligned_af_seq = align_seq_from_structure_with_ref_seq(seq, seq_from_structure)
    coords = get_aligned_residue_coords(aligned_af_seq, unaligned_coords)
    return get_indices_of_interacting_residues(seq, coords, mask, min_dist_thresh, max_dist_thresh)

def find_seqs_differences(seq1: str, seq2: str) -> str:
    seq1 = np.array(list(seq1)) 
    seq2 = np.array(list(seq2))
    diff_mask = seq1 != seq2
    diff_pos = np.where(diff_mask)[0]
    seq1_letters = seq1[diff_mask]
    seq2_letters = seq2[diff_mask]
    return "_".join([f"{i}{j}{k}" for i, j, k in zip (seq1_letters, diff_pos, seq2_letters)])
    
