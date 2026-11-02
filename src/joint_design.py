"""Autoregressive recoding with ProteinMPNN and ESM2 combined.

For each position i in recode_full_positions, in order:
  * mask i in the ProteinMPNN input -> log p_mpnn(s_i | s_-i, structure)
  * mask i in the ESM2 input        -> log p_esm (s_i | s_-i)
  * add the two log-probs, take the argmax, write it back into BOTH inputs
    before moving to the next position.

Positions with no backbone in the PDB fall back to ESM2 alone.

Two coordinate systems are in play and they are NOT interchangeable:
  * PDB resi       1-based, indexes pdb_seq (chain `target_chain`)
  * full_seq index 0-based, indexes the UniProt sequence
`build_full_to_pdb` is the only bridge between them.

Combining in LOG-PROB space, not raw-logit space, is deliberate: see `combine`.
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


class NoBackboneError(ValueError):
    """Position is in the PDB sequence but has no backbone coordinates."""


def esm_to_aa20_index(esm_alphabet):
    """ESM2 has 33 tokens in its own order; ProteinMPNN has 21 in another.
    Returns the ESM token id for each of the 20 standard amino acids."""
    return [esm_alphabet.get_idx(a) for a in AA20]


def mpnn_to_aa20_index():
    return [MPNN_ALPHABET.index(a) for a in AA20]


def build_full_to_pdb(recode_PDB_positions, structure_dir, uniprot_ID, full_seq):
    """{full_seq index (0-based) -> PDB resi (1-based)} via utils.map_pdb_to_full_seq_idx.

    That helper aligns the PDB chain against the full sequence and returns only
    the positions that aligned, dropping the rest, so zipping its output back
    onto the input list would pair the wrong residues whenever anything was
    dropped. When nothing is dropped the order is preserved and one call is
    enough; otherwise fall back to one call per position -- that costs one
    realignment each, but the pairing is then exact by construction.

    PDB positions absent from the returned dict are the ones with no structural
    counterpart; joint_recode handles those with ESM2 alone.
    """
    from utils import map_pdb_to_full_seq_idx   # lazy: utils pulls in gemmi

    pdb_positions = list(recode_PDB_positions)
    full_idx = map_pdb_to_full_seq_idx(pdb_positions, structure_dir, uniprot_ID, full_seq)
    if len(full_idx) == len(pdb_positions):
        return {f: p for p, f in zip(pdb_positions, full_idx)}

    out = {}
    for p in pdb_positions:
        hit = map_pdb_to_full_seq_idx([p], structure_dir, uniprot_ID, full_seq)
        if hit:
            out[hit[0]] = p
    return out


def combine(logp_mpnn, logp_esm, w_mpnn=1.0, w_esm=1.0, omit=()):
    """Product of experts over the 20 standard amino acids.

    Takes LOG-PROBABILITIES -- but note this does NOT change the designs relative
    to summing raw logits. log_softmax(l) = l - logsumexp(l), and logsumexp is one
    scalar per model, identical for every amino acid, so

        w1*logp_A + w2*logp_B  =  w1*l_A + w2*l_B  -  (w1*LSE_A + w2*LSE_B)

    differs from the raw-logit sum by a constant. The argmax is the same at every
    weight. The reason to work in log-prob space is that the numbers are readable
    and comparable: the sum is log[p_mpnn^w1 * p_esm^w2], a weighted product of
    experts, and the per-model terms tell you whether w1/w2 actually balance the
    two models (they did not at 1:1 -- ESM2's distributions are much sharper).
    Raw logits carry an arbitrary per-position offset and mean nothing alone.

    Setting w_mpnn=0 drops ProteinMPNN out entirely while keeping the `omit` ban,
    which is how the no-structure positions are scored.
    """
    score = w_mpnn * np.asarray(logp_mpnn) + w_esm * np.asarray(logp_esm)
    for aa in omit:
        score[AA20.index(aa)] = -np.inf
    return score


def _argmax_aa(logp20, omit=()):
    """Best amino acid under a 20-vector, with `omit` banned."""
    v = np.array(logp20, dtype=float)
    for aa in omit:
        v[AA20.index(aa)] = -np.inf
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
    old = p[f'seq_chain_{target_chain}']
    p[f'seq_chain_{target_chain}'] = seq
    p['seq'] = p['seq'].replace(old, seq)

    (X, S, mask, _lengths, chain_M, chain_encoding_all,
     _chain_list, _visible_list, _masked_list, _masked_chain_lengths,
     _chain_M_pos, _omit_AA_mask, residue_idx, *_unused) = tied_featurize(
        [p], device, chain_id_dict, None, None, None, None, None)
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
                 pdb_seq, full_seq, recode_full_positions, full_to_pdb,
                 w_mpnn=1.0, w_esm=1.0,
                 omit=('I',), target_chain='X', device=None,
                 seed=0, verbose=True):
    """Decode recode_full_positions autoregressively under both models.

    pdb_seq               chain `target_chain` sequence from the PDB -- the MPNN input
    full_seq              UniProt sequence -- the ESM2 input, so ESM sees the whole protein
    recode_full_positions positions to redesign, as 0-based indices into full_seq
    full_to_pdb           {full index (0-based) -> PDB resi (1-based)}, from
                          build_full_to_pdb. A position absent from this dict has
                          no structure and is scored by ESM2 alone.
    omit                  amino acids banned at every recoded position

    Returns (new_pdb_seq, new_full_seq, DataFrame of per-step detail).
    """
    device = device or next(esm_model.parameters()).device
    esm_idx = esm_to_aa20_index(esm_alphabet)
    mpnn_idx = mpnn_to_aa20_index()

    cur_pdb = list(pdb_seq)
    cur_full = list(full_seq)

    rows = []
    for i_full in recode_full_positions:
        pdb_resi = full_to_pdb.get(i_full)
        i_pdb = None if pdb_resi is None else pdb_resi - 1

        if i_pdb is not None and full_seq[i_full] != pdb_seq[i_pdb]:
            # local alignment tolerates mismatches, so this is possible without the
            # mapping being wrong -- but it usually means it is.
            print(f"  warning: full_seq[{i_full}]={full_seq[i_full]} vs "
                  f"pdb_seq resi {pdb_resi}={pdb_seq[i_pdb]}; check the alignment")

        lp_e = esm_conditional_logprobs(esm_model, esm_alphabet, ''.join(cur_full),
                                        i_full, device)
        e20 = lp_e[esm_idx]

        m20 = None
        if i_pdb is not None:
            try:
                lp_m = mpnn_conditional_logprobs(mpnn_model, protein, chain_id_dict,
                                                 ''.join(cur_pdb), i_pdb,
                                                 target_chain=target_chain, seed=seed)
                m20 = lp_m[mpnn_idx]
            except NoBackboneError:
                # in the PDB sequence but no coordinates -- same as no structure
                m20 = None

        if m20 is None:
            score = combine(np.zeros(20), e20, 0.0, w_esm, omit)
            source, mpnn_pick, logp_m = 'esm_only', None, np.nan
        else:
            score = combine(m20, e20, w_mpnn, w_esm, omit)
            source, mpnn_pick = 'joint', _argmax_aa(m20, omit)
        pick = AA20[int(np.argmax(score))]
        if m20 is not None:
            logp_m = float(m20[AA20.index(pick)])

        rows.append({
            'full_idx': i_full, 'pdb_resi': pdb_resi if m20 is not None else None,
            'wt': full_seq[i_full], 'picked': pick, 'source': source,
            'mpnn_argmax': mpnn_pick,
            'esm_argmax': _argmax_aa(e20, omit),
            'logp_mpnn': logp_m,
            'logp_esm': float(e20[AA20.index(pick)]),
            'joint': float(score[int(np.argmax(score))]),
        })
        if verbose:
            r = rows[-1]
            where = f"pdb {pdb_resi}" if m20 is not None else "no structure"
            alone = (f"mpnn alone {r['mpnn_argmax']}, " if m20 is not None else "")
            print(f"  full {i_full:>4} ({where:>13}): {r['wt']}->{pick}   "
                  f"({alone}esm alone {r['esm_argmax']})")

        # autoregressive: commit the pick to BOTH inputs before the next position
        cur_full[i_full] = pick
        if i_pdb is not None:
            cur_pdb[i_pdb] = pick

    return ''.join(cur_pdb), ''.join(cur_full), pd.DataFrame(rows)
