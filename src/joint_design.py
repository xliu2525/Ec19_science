"""Autoregressive recoding with ProteinMPNN and ESM2 combined.

Both models work in ONE coordinate system: 0-based indices into the full UniProt
sequence. `pad_protein_to_full_seq` aligns the PDB chain onto that sequence once
and scatters its backbone coordinates into a full-length array, leaving NaN
wherever the structure has nothing.

For each position i in recode_full_positions, in order:
  * mask i in the ProteinMPNN input -> log p_mpnn(s_i | s_-i, structure)
  * mask i in the ESM2 input        -> log p_esm (s_i | s_-i)
  * add the two log-probs, take the argmax, write it back into the sequence
    before moving to the next position.

Positions with mask=0 have no backbone, so ProteinMPNN cannot score them at all
(conditional_probs does chain_M*mask at protein_mpnn_utils.py:1315). Those fall
back to ESM2 alone.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(ROOT, "ProteinMPNN"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from protein_mpnn_utils import tied_featurize

MPNN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
AA20 = 'ACDEFGHIKLMNPQRSTVWY'
BACKBONE_ATOMS = ('N', 'CA', 'C', 'O')


class NoBackboneError(ValueError):
    """Position has no backbone coordinates, so ProteinMPNN cannot score it."""


def esm_to_aa20_index(esm_alphabet):
    """ESM2 has 33 tokens in its own order; ProteinMPNN has 21 in another.
    Returns the ESM token id for each of the 20 standard amino acids."""
    return [esm_alphabet.get_idx(a) for a in AA20]


def mpnn_to_aa20_index():
    return [MPNN_ALPHABET.index(a) for a in AA20]


def _check_target_chain_first(protein, chain_id_dict, target_chain):
    """tied_featurize concatenates designable chains first, each group sorted
    (protein_mpnn_utils.py:223-225). Indices into target_chain are therefore
    valid tensor indices only when target_chain sorts first among the designable
    chains -- everything here assumes that, so fail loudly if it does not hold."""
    designed, _fixed = chain_id_dict[protein['name']]
    if not designed or sorted(designed)[0] != target_chain:
        raise ValueError(
            f"chain {target_chain} must be the first designable chain for its "
            f"residue indices to match tensor indices; got designed={designed}")


def _rebuild_concat_seq(protein):
    """Glue all chains tgt"""
    letters = sorted(k[len('seq_chain_'):] for k in protein
                     if k.startswith('seq_chain_')) # All chain ID
    return ''.join(protein[f'seq_chain_{c}'] for c in letters)


def pad_protein_to_full_seq(protein, full_seq, structure_dir, uniprot_ID,
                            target_chain='X'):
    """Re-index the designable chain onto the full UniProt sequence.

    Returns a copy of `protein` whose `seq_chain_<target_chain>` is `full_seq`
    and whose backbone coordinates are based on full_seq, NaN
    for residues don't have structure. After this, a 0-based index into full_seq is also a valid index
    into every tensor tied_featurize returns.
    """
    from utils import map_pdb_to_full_seq_idx   # lazy: utils pulls in gemmi

    pdb_seq = protein[f'seq_chain_{target_chain}']
    L0 = len(pdb_seq)
    full_idx = map_pdb_to_full_seq_idx(list(range(1, L0 + 1)), structure_dir,
                                       uniprot_ID, full_seq)
    print(full_idx)
    if len(full_idx) == L0:
        pdb_to_full = dict(enumerate(full_idx))
    else:
        pdb_to_full = {}
        for i in range(L0):
            hit = map_pdb_to_full_seq_idx([i + 1], structure_dir, uniprot_ID, full_seq)
            if hit:
                pdb_to_full[i] = hit[0]
    if not pdb_to_full:
        raise ValueError("no residue of the PDB chain aligned to full_seq")

    LF = len(full_seq)
    src = protein[f'coords_chain_{target_chain}']
    padded = {}
    for a in BACKBONE_ATOMS:
        key = f'{a}_chain_{target_chain}'
        xyz = np.asarray(src[key], dtype=float)
        dst = np.full((LF, 3), np.nan)
        for i_pdb, i_full in pdb_to_full.items():
            dst[i_full] = xyz[i_pdb]
        padded[key] = dst.tolist()

    out = dict(protein)
    out[f'seq_chain_{target_chain}'] = full_seq
    out[f'coords_chain_{target_chain}'] = padded
    out['seq'] = _rebuild_concat_seq(out)
    return out


@torch.no_grad()
def structure_mask(mpnn_model, protein, chain_id_dict, target_chain='X'):
    """Boolean array over the designable chain: True where backbone coords exist.
    """
    _check_target_chain_first(protein, chain_id_dict, target_chain)
    device = next(mpnn_model.parameters()).device
    _X, _S, mask, *_rest = tied_featurize(
        [protein], device, chain_id_dict, None, None, None, None, None)
    L = len(protein[f'seq_chain_{target_chain}'])
    return mask[0, :L].cpu().numpy().astype(bool)


def combine(logp_mpnn, logp_esm, w_mpnn=1.0, w_esm=1.0, omit='I'):
    """Weighted sum of logprob over the 20 standard amino acids.
    Setting w_mpnn=0 drops ProteinMPNN out entirely while keeping the `omit` ban,
    which is how the no-structure positions are scored.
    """
    score = w_mpnn * np.asarray(logp_mpnn) + w_esm * np.asarray(logp_esm)
    score[AA20.index(omit)] = -np.inf
    return score


def _argmax_aa(logp20, omit):
    """Best amino acid under a 20-vector, with `omit` banned."""
    v = np.array(logp20, dtype=float)
    v[AA20.index(omit)] = -np.inf
    return AA20[int(np.argmax(v))]


@torch.no_grad()
def mpnn_conditional_logprobs(mpnn_model, protein, chain_id_dict, seq, position0,
                              target_chain='X', seed=0):
    """log p(s_i | s_-i, structure) from stock ProteinMPNN.conditional_probs.

    The trick is chain_M: conditional_probs loops over positions where chain_M==1,
    so a one-hot chain_M gives exactly one decoder pass at position0.
    """
    device = next(mpnn_model.parameters()).device
    p = dict(protein)
    p[f'seq_chain_{target_chain}'] = seq
    p['seq'] = _rebuild_concat_seq(p)

    # mask: Which coordinate has structure coordinates
    # chain_M: 1=designable, 0=fixed
    (X, S, mask, _lengths, chain_M, chain_encoding_all,
     _chain_list, _visible_list, _masked_list, _masked_chain_lengths,
     _chain_M_pos, _omit_AA_mask, residue_idx, *_unused) = tied_featurize(
        [p], device, chain_id_dict, None, None, None, None, None)
    # print(S)
    # print(mask)
    # print(chain_M)
    # print(chain_encoding_all)
    # conditional_probs does chain_M*mask internally, so a position missing from the
    # structure would silently yield the zero-initialised row instead of a distribution.
    if mask[0, position0] == 0:
        raise NoBackboneError(f"position {position0} (0-based) is missing from the structure; "
                              "ProteinMPNN has no backbone there")
    one_hot = torch.zeros_like(chain_M).float()
    one_hot[0, position0] = 1.0

    # Other positions are randomly ordered
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    randn = torch.randn(chain_M.shape, device=device, generator=gen)

    log_cond = mpnn_model.conditional_probs(
        X, S, mask, one_hot, residue_idx, chain_encoding_all, randn)
    return log_cond[0, position0].cpu().numpy()      # [21], log-softmax


@torch.no_grad()
def esm_conditional_logprobs(esm_model, esm_alphabet, seq, position0, device,
                             start_offset=1):
    """log p(s_i | s_-i) from ESM2 with position0 replaced by <mask>."""
    bc = esm_alphabet.get_batch_converter()
    _, _, toks = bc([("seq", seq)])
    toks = toks.to(device)
    toks[0, position0 + start_offset] = esm_alphabet.mask_idx
    logits = esm_model(toks)['logits'][0, position0 + start_offset]
    return F.log_softmax(logits.float(), dim=-1).cpu().numpy()   # [33]


def joint_recode(mpnn_model, protein, chain_id_dict, esm_model, esm_alphabet,
                 full_seq, recode_full_positions,
                 w_mpnn=1.0, w_esm=1.0,
                 omit='I', target_chain='X', device=None,
                 seed=0, verbose=True):
    """Decode recode_full_positions autoregressively under both models.

    protein               padded by pad_protein_to_full_seq, so its designable
                          chain IS full_seq
    full_seq              UniProt sequence; both models are indexed by it
    recode_full_positions positions to redesign, as 0-based indices into full_seq
    omit                  amino acids banned at every recoded position

    Positions without backbone coordinates are scored by ESM2 alone.

    Returns (design dict, DataFrame of per-step detail).
    """
    designs={}
    if protein[f'seq_chain_{target_chain}'] != full_seq:
        raise ValueError("protein is not padded to full_seq; "
                         "call pad_protein_to_full_seq first")
    device = device or next(esm_model.parameters()).device
    esm_idx = esm_to_aa20_index(esm_alphabet)
    mpnn_idx = mpnn_to_aa20_index()
    has_structure = structure_mask(mpnn_model, protein, chain_id_dict, target_chain)

    cur = list(full_seq)
    rows = []
    for i in recode_full_positions:
        lp_e = esm_conditional_logprobs(esm_model, esm_alphabet, ''.join(cur), i, device)
        e20 = lp_e[esm_idx]

        if has_structure[i]:
            lp_m = mpnn_conditional_logprobs(mpnn_model, protein, chain_id_dict,
                                             ''.join(cur), i,
                                             target_chain=target_chain, seed=seed)
            m20 = lp_m[mpnn_idx]
            score = combine(m20, e20, w_mpnn, w_esm, omit)
            source, mpnn_pick = 'joint', _argmax_aa(m20, omit)
        else:
            m20 = None
            score = combine(np.zeros(20), e20, 0.0, w_esm, omit)
            source, mpnn_pick = 'esm_only', None

        pick = AA20[int(np.argmax(score))]
        rows.append({
            'full_idx': i, 'wt': full_seq[i], 'picked': pick, 'source': source,
            'mpnn_argmax': mpnn_pick,
            'esm_argmax': _argmax_aa(e20, omit),
            'logp_mpnn': np.nan if m20 is None else float(m20[AA20.index(pick)]),
            'logp_esm': float(e20[AA20.index(pick)]),
            'joint': float(score[int(np.argmax(score))]),
        })
        if verbose:
            r = rows[-1]
            alone = (f"mpnn alone {r['mpnn_argmax']}, " if m20 is not None else "")
            print(f"  {i:>4} ({source:>9}): {r['wt']}->{pick}   "
                  f"({alone}esm alone {r['esm_argmax']})")

        cur[i] = pick                 # autoregressive: commit before the next position

    designs["ProteinMPNN+ESM2"]=''.join(cur)
    
    return designs, pd.DataFrame(rows)
