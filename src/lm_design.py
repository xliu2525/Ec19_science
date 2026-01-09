import glob
import os
import random
import tqdm
from typing import Dict, List, Union

import numpy as np
import torch
import esm

from masking import Masking
from utils import load_sequences, read_msa, compute_msa_seq_pseudo_log_likelihood, \
                        set_msa_nseq, find_seqs_differences, read_fasta


class EsmRecoder:
    def __init__(
        self,
        seq_file: os.PathLike,
        msa_dir: Union[os.PathLike, None],
        structure_dir: Union[os.PathLike, None],
        residue_to_recode: str,
        model: esm.model,
        alphabet: esm.data.Alphabet,
        device: str,
        incl_seq_neighbors: bool,
        incl_3d_neighbors: bool,
        incl_evo_neighbors: bool,
        min_3d_dist: int,
        max_3d_dist: int,
        save_muts: bool
    ):
        self.seq_file = seq_file
        self.msa_dir = msa_dir
        self.structure_dir = structure_dir
        self.device = device
        self.alphabet = alphabet
        self.model = model.eval().to(device)
        self.residue_to_recode = residue_to_recode
        self.id_to_recode = self.alphabet.tok_to_idx[residue_to_recode]
        self.mask_id = self.alphabet.mask_idx
        self.eos_id = self.alphabet.tok_to_idx['<eos>']
        self.non_canonical = ['X', 'B', 'U', 'Z', 'O']
        self.special_toks = ['<cls>', '<pad>', '<eos>', '<unk>', '.', '-', '<null_1>', '<mask>']
        self.undesired_toks = self.non_canonical + self.special_toks + [self.residue_to_recode]
        self.undesired_tok_ids = [self.alphabet.get_idx(i) for i in self.undesired_toks]
        self.batch_converter = self.alphabet.get_batch_converter()
        self.save_muts = save_muts
        self.start_offset = 1

        self.masking = Masking(
            self.structure_dir,
            self.id_to_recode,
            self.residue_to_recode, 
            self.mask_id,
            self.eos_id,
            incl_seq_neighbors, 
            incl_3d_neighbors, 
            incl_evo_neighbors, 
            min_3d_dist,
            max_3d_dist,
        )

    def zero_out_undesired_logits(self, logits):
        logits[self.undesired_tok_ids] = -np.inf
        return logits

    def convert_toks_to_seq(self, tokens: torch.Tensor):
        new_seq = ''
        for _, tok_id in enumerate(tokens[self.start_offset:]):  # exclude start token
            new_letter = self.alphabet.get_tok(tok_id)
            new_seq += new_letter
        new_seq = new_seq.split('<eos>')[0]  # remove eos and padding tokens
        return new_seq

    def recode(self):
        if type(self.model) is esm.model.esm2.ESM2:
            dataloader = load_sequences(self.seq_file, self.alphabet)
            designs = self.recode_sequences(dataloader=dataloader)
            return designs

        elif type(self.model) is esm.model.msa_transformer.MSATransformer:

            designs = self.recode_sequences(msa_dir=self.msa_dir, seq_file=self.seq_file)
            return designs
        else:
            raise NotImplementedError('No recoding strategy implemented for this model.')

    def recode_sequences(self, **kwargs) -> Dict[str, str]:
        raise NotImplementedError('Childrens must implement _recode.')


class SimultaneousEsmRecoder(EsmRecoder):

    def recode_sequences(self, **kwargs) -> Dict[str, str]:
        designs = {}
        torch.set_grad_enabled(False)
        dataloader = kwargs['dataloader']

        for batch_idx, (batch_labels, batch_seqs, batch_toks) in tqdm.tqdm(
                enumerate(dataloader), total=len(dataloader), desc='Processing batches.'
        ):
            if ("muts" in batch_labels[0]) and ("worst" in batch_labels[0]):
                recode_from_prev_design = True
            else:
                recode_from_prev_design = False

            # [*, N_res]
            batch_masked_toks = self.masking.do_masking_in_batch_seqs(
                batch_toks, batch_labels, batch_seqs, recode_from_prev_design, self.alphabet, self.start_offset
            )

            # [*, N_res, N_vocab]
            batch_logits = self.model(batch_masked_toks.to(self.device), repr_layers=[33], return_contacts=False)['logits'].cpu()

            # [*, N_res]
            recoded_batch_toks = batch_toks.clone().cpu()

            # Recode with most likely alternative residue
            for i, masked_toks in enumerate(batch_masked_toks): 
                for j, tok in enumerate(masked_toks):
                    if tok != self.mask_id: 
                        continue
                    tok_logits = self.zero_out_undesired_logits(batch_logits[i, j])
                    new_tok = np.argmax(tok_logits)

                    # [*, N_res]
                    recoded_batch_toks[i, j] = new_tok

            # Decode
            for label, seq, seq_toks in zip(batch_labels, batch_seqs, recoded_batch_toks):
                if ("muts" in label) and ("worst" in label):
                    prefix = label.split('|')[0]
                else:
                    prefix = label
                new_seq = self.convert_toks_to_seq(seq_toks)
                diffs = find_seqs_differences(seq, new_seq)
                design_name = f"{prefix}_ESM2_simultaneous"
                if self.save_muts:
                    design_name += f"_{diffs}"
                designs[design_name] = new_seq
        return designs


class AutoregressiveEsmRecoder(EsmRecoder):

    def _recode_one_sequence(self, masked_toks: torch.LongTensor):
        mask_count = (masked_toks == self.mask_id).sum().item()

        # Infer on one masked token a a time
        for _ in range(mask_count):

            # [1, N_res, N_vocab]
            logits = self.model(masked_toks.unsqueeze(0).to(self.device), repr_layers=[33], return_contacts=False)['logits'].cpu()

            # [N_res, N_vocab]
            logits = torch.squeeze(logits, 0)

            idx = masked_toks.tolist().index(self.mask_id)

            # [N_vocab]
            tok_logits = self.zero_out_undesired_logits(logits[idx])
            new_tok = np.argmax(tok_logits)

            masked_toks[idx] = new_tok

        # [N_res]
        return masked_toks

    def recode_sequences(self, **kwargs) -> Dict[str, str]:
        designs = {}
        torch.set_grad_enabled(False)
        dataloader = kwargs['dataloader']
        #is_left_to_right = kwargs['is_left_to_right']

        for batch_idx, (batch_labels, batch_seqs, batch_toks) in tqdm.tqdm(
                enumerate(dataloader), total=len(dataloader), desc='Processing batches.'
        ):
            if ("muts" in batch_labels[0]) and ("worst" in batch_labels[0]):
                recode_from_prev_design = True
            else:
                recode_from_prev_design = False

            # [*, N_res]
            batch_masked_toks = self.masking.do_masking_in_batch_seqs(
                batch_toks, batch_labels, batch_seqs, recode_from_prev_design, self.alphabet, self.start_offset
            )

            #if not is_left_to_right:
                # TODO: only flip tokens between <cls> (idx:0) and <eos> (idx: 2)
                # Difficulty: padding (idx: 1) with batches - can only flip one line at a time...
                #batch_masked_toks = batch_masked_toks.flip(dims=[-1])

            # Infer one sequence at a time
            for label, seq, masked_toks in zip(batch_labels, batch_seqs, batch_masked_toks):
                if self.mask_id not in masked_toks:
                    continue

                if ("muts" in label) and ("worst" in label):
                    prefix = label.split('|')[0]
                else:
                    prefix = label

                predicted_toks = self._recode_one_sequence(masked_toks)
                #if not is_left_to_right:
                    #predicted_toks = predicted_toks.flip(dims=[-1])
                # Decode
                new_seq = self.convert_toks_to_seq(predicted_toks)

                diffs = find_seqs_differences(seq, new_seq)
                design_name = f"{prefix}_ESM2_autoregressive"
                if self.save_muts:
                    design_name += f"_{diffs}"
                designs[design_name] = new_seq

        return designs


class MaxLogLikelihoodEsmRecoder(EsmRecoder):
    def __init__(self,
                 residue_to_recode: str,
                 model: esm.model,
                 alphabet: esm.data.Alphabet,
                 device: str,
                 incl_seq_neighbors: bool, #TODO 
                 incl_3d_neighbors: bool,
                 incl_evo_neighbors: bool):

        self.device = device
        self.alphabet = alphabet
        self.model = model
        self.model.eval().to(device)
        self.ARRecoder = AutoregressiveEsmRecoder(residue_to_recode, model, alphabet, device)

    def recode_sequences(self, **kwargs) -> Dict[str, str]:
        designs = {}
        torch.set_grad_enabled(False)
        dataloader = kwargs['dataloader']
        ltr_count, rtl_count = 0, 0
        left_to_right_designs = self.ARRecoder.recode_sequences(dataloader=dataloader, is_left_to_right=True)

        # TODO: uncomment below lines after fixing bug in autoregressive right to left setup
        #right_to_left_designs = self.ARRecoder.recode_sequences(dataloader=dataloader, is_left_to_right=False)
        
        for name, ltr_seq in tqdm.tqdm(
                left_to_right_designs.items(), total=len(left_to_right_designs), desc='Processing sequences.'
        ):
            """
            ltr_pll = compute_esm_seq_pseudo_log_likelihood(self.model, self.alphabet, ltr_seq, self.device)
            rtl_seq = right_to_left_designs[name]
            rtl_pll = compute_esm_seq_pseudo_log_likelihood(self.model, self.alphabet, rtl_seq, self.device)

            if ltr_pll > rtl_pll:
                designs[name] = ltr_seq
                ltr_count += 1
            else:
                designs[name] = rtl_seq
                rtl_count += 1
            """
            designs[name] = ltr_seq
            ltr_count += 1
        print('left to right', ltr_count)
        print('right to left', rtl_count)
        
        # return designs
        return left_to_right_designs


class GibbsEsmRecoder(EsmRecoder):  # TODO: check code for sampling
    def _compute_seq_pseudo_log_likelihood(self, toks: torch.Tensor, mask_pos: List):
        mask_logprobs = []
        for idx in mask_pos:
            tok_to_predict = toks[idx]

            # [N_res]
            masked_toks = toks.clone()
            masked_toks[idx] = self.alphabet.mask_idx

            # [1, N_res]
            masked_toks = torch.unsqueeze(masked_toks, 0).to(self.device)

            # [1, N_res, N_vocab]
            logits = self.model(masked_toks, repr_layers=[33], return_contacts=False)['logits'].cpu()

            # [N_res, N_vocab]
            logits = torch.squeeze(logits, 0)

            # [N_vocab]
            logprobs = torch.log_softmax(logits[idx], dim=-1)

            mask_logprob = logprobs[tok_to_predict]
            mask_logprobs.append(mask_logprob)
            del masked_toks

        return np.mean(mask_logprobs)

    def _select_seq_with_closest_score_to_wild_type(self, ref_score: float, new_scores: List[float], seqs: List[str]):
        diffs = [abs(ref_score - new_pll) for new_pll in new_scores]
        return seqs[np.argmin(diffs)]

    def _select_seq_with_highest_score(self, new_scores: List[float], seqs: List[str]):
        return seqs[np.argmax(new_scores)]

    def recode_sequences(self, **kwargs) -> Dict[str, str]:
        designs = {}
        torch.set_grad_enabled(False)
        dataloader = kwargs['dataloader']
        iteration_multiplier = 50
        burn_in_multiplier = 5

        for batch_idx, (batch_labels, batch_seqs, batch_toks) in tqdm.tqdm(
                enumerate(dataloader), total=len(dataloader), desc='Processing batches.'
        ):
            for label, seq, toks in zip(batch_labels, batch_seqs, batch_toks):
                if self.id_to_recode not in toks:
                    continue

                if ("muts" in label) and ("worst" in label):
                    recode_from_prev_design = True
                else:
                    recode_from_prev_design = False

                sampled_sequences, sampled_log_likelihoods = [], []

                # [N_res]
                mask_pos = np.where(toks == self.id_to_recode)[0].tolist() # TODO: implement neighbor recoding
                mask_count = len(mask_pos)
                total_iterations = mask_count * iteration_multiplier
                burn_in_iterations = mask_count * burn_in_multiplier

                # [N_res]
                toks = toks.to(self.device)

                # compute wild type pseudo log likelihood (PLL)
                wt_pll = self._compute_seq_pseudo_log_likelihood(toks=toks, mask_pos=mask_pos)

                for i in range(total_iterations):
                    new_tok = np.nan
                    idx = random.choice(mask_pos)
                    toks[idx] = self.mask_id

                    # [1, N_res]
                    toks = torch.unsqueeze(toks, 0)

                    # [1, N_res, N_vocab]
                    logits = self.model(toks, repr_layers=[33], return_contacts=False)['logits'].cpu()

                    # [N_res, N_vocab]
                    logits = torch.squeeze(logits, 0)

                    # [N_res]
                    toks = torch.squeeze(toks, 0)

                    logits[idx] = self.zero_out_undesired_logits(logits[idx])
                    # TODO: what? argmax? shouldn't it be random.choice(, p=logits)
                    new_tok = np.argmax(logits[idx])
                    toks[idx] = new_tok

                    if (i + 1 > burn_in_iterations) and (total_iterations % mask_count == 0):
                        # Decode
                        new_seq = self.convert_toks_to_seq(toks)
                        sampled_sequences.append(new_seq)

                        # compute new seq pseudo log likelihood
                        # TODO: only do at the end, to parallelize operations
                        new_seq_pll = self._compute_seq_pseudo_log_likelihood(toks=toks, mask_pos=mask_pos)
                        sampled_log_likelihoods.append(new_seq_pll)

                # Save sequences with highest PLL, and closest PLL to wild type PLL
                seq_with_closest_pll = self._select_seq_with_closest_score_to_wild_type(
                    ref_score=wt_pll, new_scores=sampled_log_likelihoods, seqs=sampled_sequences)

                seq_with_highest_pll = self._select_seq_with_highest_score(
                    new_scores=sampled_log_likelihoods, seqs=sampled_sequences)

                if ("muts" in label) and ("worst" in label):
                    prefix = label.split('|')[0]
                else:
                    prefix = label
                designs[prefix + '_ESM2_gibbs_closest_PLL'] = seq_with_closest_pll
                designs[prefix + '_ESM2_gibbs_highest_PLL'] = seq_with_highest_pll

        return designs


class SimultaneousMsaTransRecoder(EsmRecoder):

    def recode_sequences(self, **kwargs):
        designs = {}
        torch.set_grad_enabled(False)

        labels, wt_seqs = read_fasta(kwargs['seq_file'])

        if ("muts" in labels[0]) and ("worst" in labels[0]):
            recode_from_prev_design = True
        else:
            recode_from_prev_design = False

        for label, wt_seq in tqdm.tqdm(zip(labels, wt_seqs), total=len(labels), desc='Processing MSAs.'):
            prefix = label.split('|')[0]
            uniprot = prefix.split('_')[1]
            try:
                msa_file = glob.glob(os.path.join(kwargs['msa_dir'], f"{uniprot}*.a3m"))[0]
                print(prefix)
            except:
                print(f"{prefix}: missing MSA")
                continue

            if len(wt_seq) > 1024:
                continue

            # read MSA 
            nseq = set_msa_nseq(wt_seq)
            msa = read_msa(msa_file, nseq=nseq)
            _, msa_seqs, msa_toks = self.batch_converter(msa)
            
            # Sanity checks
            msa_first_seq = msa_seqs[0][0]

            if self.residue_to_recode not in msa_first_seq:
                continue
            try:
                assert msa_first_seq == wt_seq
            except AssertionError:
                print(f"{msa_file}: The MSA first sequence does not match the WT sequence provided in the fasta file.")
                if wt_seq[1:] == msa_first_seq[1:]:
                    print('Start residues differ.')
                    continue

            # Mask tokens to recode and run inference
            # [1, N_res]
            seq_toks = msa_toks[:, 0, :]
            masked_toks = self.masking.do_masking_in_msa_seq(seq_toks, label, msa_first_seq, recode_from_prev_design, self.alphabet, self.start_offset)

            # [1, N_seq - 1, N_res]
            unmasked_toks = msa_toks[:, 1:, :]

            # [1, N_seq, N_res]
            toks = torch.cat((masked_toks.unsqueeze(0), unmasked_toks), dim=1).to(self.device)

            del unmasked_toks
            msa_logits = self.model(toks.to(self.device), repr_layers=[12], return_contacts=False)['logits'].cpu()

            # [N_res, N_vocab]
            seq_logits = msa_logits[0, 0, :, :]

            # [N_res]
            recoded_seq_toks = msa_toks[0][0].clone().cpu()

            # Recode with most likely alternative residue
            for i, tok in enumerate(masked_toks.squeeze(0)):
                if tok != self.mask_id:
                    continue
                tok_logits = self.zero_out_undesired_logits(seq_logits[i])
                new_tok = np.argmax(tok_logits)
                recoded_seq_toks[i] = new_tok
            del toks, masked_toks

            # Decode
            new_seq = self.convert_toks_to_seq(recoded_seq_toks)
            diffs = find_seqs_differences(msa_first_seq, new_seq)
            design_name = f'{prefix}_MsaTransformer_simultaneous'
            if self.save_muts:
                design_name += f"_{diffs}"
            designs[design_name] = new_seq

        return designs


class AutoregressiveMsaTransRecoder(EsmRecoder):

    def _recode_one_sequence(self, masked_toks: torch.LongTensor, unmasked_toks: torch.LongTensor):
        mask_count = (masked_toks[0, :] == self.mask_id).sum().item()

        for _ in range(mask_count):
            # [1, N_seq, N_res]
            toks = torch.cat((masked_toks.unsqueeze(0), unmasked_toks), dim=1).to(self.device)
            msa_logits = self.model(toks, repr_layers=[12], return_contacts=False)['logits'].cpu()
            del toks

            # [N_res, N_vocab]
            seq_logits = msa_logits[0, 0, :, :]

            idx = masked_toks[0, :].tolist().index(self.mask_id)

            # [N_vocab]
            tok_logits = self.zero_out_undesired_logits(seq_logits[idx])
            new_tok = np.argmax(tok_logits)

            masked_toks[0, idx] = new_tok

        # [1, N_res]
        return masked_toks

    def recode_sequences(self, **kwargs) -> Dict[str, str]:
        designs = {}
        torch.set_grad_enabled(False)

        labels, wt_seqs = read_fasta(kwargs['seq_file'])

        if ("muts" in labels[0]) and ("worst" in labels[0]):
            recode_from_prev_design = True
        else:
            recode_from_prev_design = False
       
        for label, wt_seq in tqdm.tqdm(zip(labels, wt_seqs), total=len(labels), desc='Processing MSAs.'):
            prefix = label.split('|')[0]
            uniprot = prefix.split('_')[1]
            try: 
                msa_file = glob.glob(os.path.join(kwargs['msa_dir'], f"{uniprot}*.a3m"))[0]
                print(prefix)
            except:
                print(f"{prefix}: missing MSA")
                continue

            if len(wt_seq) > 1024:
                continue

            # read MSA 
            nseq = set_msa_nseq(wt_seq)
            msa = read_msa(msa_file, nseq=nseq)
            _, msa_seqs, msa_toks = self.batch_converter(msa)

            # Sanity checks
            msa_first_seq = msa_seqs[0][0]

            if self.residue_to_recode not in msa_first_seq:
                continue

            try:
                # MSA first sequence must match the WT sequence from the fasta file
                assert msa_first_seq == wt_seq
            except AssertionError:
                print(f"{msa_file}: The MSA first sequence does not match the WT sequence provided in the fasta file.")
                if wt_seq[1:] == msa_first_seq[1:]:
                    print('Start residues differ.')
                    continue

            # TODO: move legacy code to function in utils, in case want to flip tokens in the future
            """
            if not kwargs['is_left_to_right']:
                # flip tokens after <cls> token
                start_toks = msa_toks[:, :, 0]
                flipped_toks = msa_toks[:, :, 1:].flip(dims=[-1])
                msa_toks = torch.cat((start_toks.unsqueeze(-1), flipped_toks), dim=-1)
                del start_toks, flipped_toks
            """

            # Iterative recoding
            # [1, N_res]
            # masked_toks = self.mask_token_id(msa_toks[:, 0, :], is_msa=True)
            seq_toks = msa_toks[:, 0, :]
            masked_toks = self.masking.do_masking_in_msa_seq(seq_toks, label, msa_first_seq, recode_from_prev_design, self.alphabet, self.start_offset)

            # [1, N_seq - 1, N_res]
            unmasked_toks = msa_toks[:, 1:, :]

            # [1, N_res]
            predicted_toks = self._recode_one_sequence(masked_toks, unmasked_toks)

            # [N_res]
            predicted_toks = predicted_toks.squeeze(0)
            del masked_toks, unmasked_toks

            """
            if not kwargs['is_left_to_right']:
                predicted_flipped_toks = predicted_toks[1:].flip(dims=[-1])

                # [N_res]
                predicted_toks = torch.cat((predicted_toks[0].unsqueeze(0), predicted_flipped_toks), dim=-1)
                del predicted_flipped_toks
            """

            # Decode
            new_seq = self.convert_toks_to_seq(predicted_toks)
            diffs = find_seqs_differences(msa_first_seq, new_seq)
            design_name = f'{prefix}_MsaTransformer_autoregressive'
            if self.save_muts:
                design_name += f"_{diffs}"
            designs[design_name] = new_seq

        return designs


class MaxLogLikelihoodMsaTransRecoder(EsmRecoder):
    def __init__(self,
                 residue_to_recode: str,
                 model: esm.model,
                 alphabet: esm.data.Alphabet,
                 device: str):

        self.device = device
        self.alphabet = alphabet
        self.model = model
        self.model.eval().to(device)
        self.ARRecoder = AutoregressiveMsaTransRecoder(residue_to_recode, model, alphabet, device)

    def recode_sequences(self, **kwargs) -> Dict[str, str]:
        designs = {}
        torch.set_grad_enabled(False)

        ltr_count, rtl_count = 0, 0
        left_to_right_designs = self.ARRecoder.recode_sequences(
            msa_dir=kwargs['msa_dir'],
            seq_file=kwargs['seq_file'],
            is_left_to_right=True
        )
        right_to_left_designs = self.ARRecoder.recode_sequences(
            msa_dir=kwargs['msa_dir'],
            seq_file=kwargs['seq_file'],
            is_left_to_right=False
        )

        print('Select designs with highest pseudo log likelihood.')

        for name, ltr_seq in tqdm.tqdm(
                left_to_right_designs.items(), total=len(left_to_right_designs), desc='Processing sequences.'
        ):
            ltr_pll = compute_msa_seq_pseudo_log_likelihood(
                self.model, self.alphabet, ltr_seq, self.device, name=name, msa_dir=kwargs['msa_dir']
            )
            rtl_seq = right_to_left_designs[name]
            rtl_pll = compute_msa_seq_pseudo_log_likelihood(
                self.model, self.alphabet, rtl_seq, self.device, name=name, msa_dir=kwargs['msa_dir'])

            if ltr_pll > rtl_pll:
                designs[name] = ltr_seq
                ltr_count += 1
            else:
                designs[name] = rtl_seq
                rtl_count += 1

        print('left to right', ltr_count)
        print('right to left', rtl_count)
        return designs
