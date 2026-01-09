import os
from typing import List, Union

import esm
import torch

from utils import get_interacting_residues_from_structure


class Masking:
    def __init__(
        self,
        structure_dir: Union[os.PathLike, None],
        id_to_recode: str,
        residue_to_recode: str,
        mask_id: str,
        eos_id: str,
        incl_seq_neighbors: bool,
        incl_3d_neighbors: bool,
        incl_evo_neighbors: bool,
        min_3d_dist: float,
        max_3d_dist: float
    ):
        """
        Implements masking strategies for sampling designs from protein language models.

        Masking strategies:
        - **Default masking**: Only the residue to recode is masked.
        - **Sequence neighbors masking** (`incl_seq_neighbors=True`): The residues immediately adjacent
          (left and right) to the residue to recode are also masked.
        - **3D neighbors masking** (`incl_3d_neighbors=True`): Residues within the PDB structure of the
          wild-type gene that are within `min_3d_dist` to `max_3d_dist` Angstroms of the residue to recode
          are masked.
        - **Evolutionary neighbors masking** (`incl_evo_neighbors=True`): Residues that show evolutionary
          covariance with the residue to recode are masked.

           Args:
            structure_dir (Union[os.PathLike, None]):
                Path to the directory containing PDB structures of wild-type genes.
            id_to_recode (str):
                Token ID of the residue to recode in the language model's alphabet.
            residue_to_recode (str):
                One-letter code of the residue to recode.
            mask_id (str):
                Token ID for the mask symbol in the language model's alphabet.
            eos_id (str):
                Token ID for the end-of-sequence symbol in the language model's alphabet.
            incl_seq_neighbors (bool):
                If `True`, masks the primary sequence neighbors (adjacent residues) of the residue to recode.
            incl_3d_neighbors (bool):
                If `True`, masks spatial neighbors of the residue to recode based on 3D structure.
            incl_evo_neighbors (bool):  
                If `True`, masks residues that evolutionarily covary with the residue to recode.
            min_3d_dist (float):
                Minimum spatial distance (in Angstroms) to define 3D neighbors.
            max_3d_dist (float):
                Maximum spatial distance (in Angstroms) to define 3D neighbors.
        """
        self.structure_dir = structure_dir
        self.id_to_recode = id_to_recode
        self.residue_to_recode = residue_to_recode
        self.mask_id = mask_id
        self.eos_id = eos_id
        self.incl_seq_neighbors = incl_seq_neighbors
        self.incl_3d_neighbors = incl_3d_neighbors
        self.incl_evo_neighbors = incl_evo_neighbors
        self.min_3d_dist = min_3d_dist
        self.max_3d_dist = max_3d_dist
    
    def mask_token_id(self, toks: torch.Tensor, must_mask: torch.Tensor) -> torch.Tensor:
        return toks * (1 - must_mask) + self.mask_id * must_mask
    
    def mask_seq_neighbors(self, must_mask: torch.Tensor):
        new_col = torch.zeros((must_mask.shape[0],1))
        right_neighbors = torch.cat((new_col, must_mask), 1)[:,:-1]
        left_neighbors = torch.cat((must_mask, new_col), 1)[:,1:]
        new_mask = left_neighbors + must_mask + right_neighbors 
        return torch.clip(new_mask, max=1).long()
    
    def mask_3d_neighbors_in_seq_batch(self, must_mask: torch.Tensor, target_res_mask: torch.Tensor, labels: List[str], seqs: List[str], start_offset: int):
        for i, (label, seq) in enumerate(zip(labels, seqs)):
            uniprot = label.split('|')[0].split("_")[1]
            interacting_pos = get_interacting_residues_from_structure(
                self.structure_dir, uniprot, seq, target_res_mask[i][start_offset:len(seq)+1], self.min_3d_dist, self.max_3d_dist
                )
            # add selected interacting residues to original mask
            print(f"3D neighbors: {interacting_pos}")
            for res_idx in interacting_pos:
                must_mask[i][res_idx + start_offset] = 1
        return must_mask
    
    def mask_3d_neighbors_in_msa_seq(self, must_mask: torch.Tensor, target_res_mask: torch.Tensor, label: str, seq: str, start_offset: int):
        uniprot = label.split('|')[0].split("_")[1]
        interacting_pos = get_interacting_residues_from_structure(
            self.structure_dir, uniprot, seq, target_res_mask[0][start_offset:len(seq)+1], self.min_3d_dist, self.max_3d_dist
            )
        # add selected interacting residues to original mask
        print(f"3D neighbors: {interacting_pos}")
        for res_idx in interacting_pos:
            must_mask[0][res_idx + start_offset] = 1
        return must_mask
    
    def fix_mask(self, toks: torch.Tensor, must_mask: torch.Tensor) -> torch.Tensor:
        """Make sure cls and eos tokens are set to zero."""
        must_mask[:,0] = 0   # cls 
        must_mask[toks == self.eos_id] = 0   # eos 
        must_mask = torch.clip(must_mask, max=1)
        # TODO: replace Ile positions with best positions found by previous designs
        return must_mask
    
    def do_masking_in_batch_seqs(
            self, toks: torch.Tensor, labels: List[str], seqs: List[str], recode_from_prev_design: bool, alphabet: esm.data.Alphabet, start_offset: int
        ) -> torch.Tensor:

        if recode_from_prev_design:
            # recode from best previous design
            must_mask = torch.zeros(size=toks.size()).long()
            for i, label in enumerate(labels):
                # mutate seq tokens using previous design
                muts = label.split('|')[1].split("muts_")[1].split('-')
                for mut in muts:
                    mut_idx = int(mut[1:-1])
                    new_tok = alphabet.tok_to_idx[mut[-1]]
                    toks[i][mut_idx + start_offset] = new_tok

                # create mask only for worst positions
                worst_pos = label.split('|')[2].split("worst_")[1].split('-')
                worst_pos = list(map(int, worst_pos))
                for pos in worst_pos:
                    must_mask[i][pos + start_offset] = 1
        else:
            # recode from wild type
            must_mask = (toks == self.id_to_recode).long()
        target_res_must_mask = must_mask.clone().detach()
        
        if self.incl_seq_neighbors:
            must_mask = self.mask_seq_neighbors(must_mask)

        if self.incl_3d_neighbors:
            must_mask = self.mask_3d_neighbors_in_seq_batch(must_mask, target_res_must_mask, labels, seqs, start_offset)

        must_mask = self.fix_mask(toks, must_mask)
        return self.mask_token_id(toks, must_mask)
    
    def do_masking_in_msa_seq(
            self, toks: torch.Tensor, label: str, seq: str, recode_from_prev_design: bool, alphabet: esm.data.Alphabet, start_offset: int=0
        ) -> torch.Tensor:

        if recode_from_prev_design:
            # recode from best previous design
            must_mask = torch.zeros(size=toks.size()).long()

            # mutate seq tokens using previous design
            muts = label.split('|')[1].split("muts_")[1].split('-')
            for mut in muts:
                mut_idx = int(mut[1:-1])
                new_tok = alphabet.tok_to_idx[mut[-1]]
                toks[0][mut_idx + start_offset] = new_tok

            # create mask only for worst positions
            worst_pos = label.split('|')[2].split("worst_")[1].split('-')
            worst_pos = list(map(int, worst_pos))
            for pos in worst_pos:
                must_mask[0][pos + start_offset] = 1
        else:
            # recode from wild type
            must_mask = (toks == self.id_to_recode).long()
        target_res_must_mask = must_mask.clone().detach()
        
        if self.incl_seq_neighbors:
            must_mask = self.mask_seq_neighbors(must_mask)

        if self.incl_3d_neighbors:
            must_mask = self.mask_3d_neighbors_in_msa_seq(must_mask, target_res_must_mask, label, seq, start_offset)
        
        must_mask = self.fix_mask(toks, must_mask)
        return self.mask_token_id(toks, must_mask)