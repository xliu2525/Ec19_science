import torch
import argparse
import os, sys
# os.chdir("/n/holystore01/LABS/xlin/Lab/xliu2525/Code/Ec19_science")
# sys.path.append("/n/holystore01/LABS/xlin/Lab/xliu2525/Code/Ec19_science/src")
from common import init; init(os.path.expanduser("~/.cache/recoding"))
import esm
from mpnn import init_mpnn_model, load_data_from_pdb_path
from recode_structure import extract_chains_ids
from joint_design import (joint_recode, pad_protein_to_full_seq)
from utils import read_fasta,check_recoded_sequences,write_fasta,get_interacting_indices_from_confind,map_pdb_to_full_seq_idx
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Apply design strategies to recode wild-type genes using ProteinMPNN + ESM2."
    )

    parser.add_argument(
        "--infile",
        type=str,
        required=True,
        default="data/designs/esm2_3d_neighbors/from_best_designs.fasta",
        help="Path to the input FASTA file containing wild-type sequences."
    )
    
    parser.add_argument(
        "--confind_neighbors",
        action="store_true",
        default=False,
        help="If set, recodes residues that are close to the target residue defined using Confind method."
    )
    
    parser.add_argument(
        "--structure_dir", 
        type=str,
        required=True,
        default=None,
        help="Directory containing PDB structures of wild-type genes. "
    )
    
    parser.add_argument(
        "--confind_dir", 
        type=str,
        default=None,
        help="Directory containing confind output "
            "Required when using confind neighbors for recoding."
    )
    
    parser.add_argument(
        "--confind_cutoff",
        type=float,
        default=0.1,
        help="Confind threshold to define neighbors."
    )
    
    # Recoding parameters
    parser.add_argument(
        "--residue",
        type=str,
        default='I',
        choices=['I', 'V'],
        help="Residue to recode (e.g., 'I' for isoleucine, 'V' for valine)."
    )
    
    parser.add_argument(
        "--scheme",
        type=str,
        required=True,
        choices=['autoregressive'],
        help="Design strategy used to recode the wild-type genes. "
            "'autoregressive' modifies sequentially"
    )

    parser.add_argument(
        "--save_muts",
        action="store_true",
        default=False,
        help="If set, appends the list of mutations to the design name in the output FASTA header."
    )

    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Path to the output FASTA file where recoded sequences will be saved."
    )
    
    args = parser.parse_args()
    
    # Load the model 
    # Use the GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load proteinmpnn and ESM2
    mpnn_model = init_mpnn_model("/n/holystore01/LABS/xlin/Lab/xliu2525/Code")
    esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model.eval().to(device)
    
    # Read in fasta file
    headers,seqs=read_fasta(args.infile)
      
    # Collect designs from all sequences: {design_name: design_seq}
    designs={}

    # Recode one seq at a time
    for i in range(len(seqs)):
        uniprot=headers[i].split("_")[0]
        sequence=seqs[i]
        # Load structure
        PDB = f"{args.structure_dir}/{uniprot}_nearby_protein_I.pdb"
        fixed_chains, _ = extract_chains_ids(PDB, "X")
        fixed_chains = [c for c in fixed_chains if c != "X"]
        dataset, chain_id_dict, _ = load_data_from_pdb_path(PDB, "X", ",".join(fixed_chains))
        protein = dataset[0] # dictionary contains info from the PDB file
        
        # Pad PDB sequence to be same as uniprot seq
        padded = pad_protein_to_full_seq(protein, sequence, args.structure_dir, uniprot)
        
        recode_positions=[i for i, c in enumerate(sequence) if c == args.residue]

        if args.confind_neighbors:

            # Find confind neighbors for each chain (0-based)
            neighboring_PDB_idx=get_interacting_indices_from_confind(args.confind_dir, uniprot,
                                                                    args.confind_cutoff)
            interacting_pos=map_pdb_to_full_seq_idx(neighboring_PDB_idx,args.structure_dir,
                                                    uniprot,sequence)

            recode_positions=sorted(set(recode_positions+interacting_pos))

        if args.scheme == "autoregressive":
            # design is a dictionary
            design, df = joint_recode(
                mpnn_model, padded, chain_id_dict, esm_model, esm_alphabet,
                full_seq=sequence,
                recode_full_positions=recode_positions,
                w_mpnn=1.0, w_esm=1.0,
                omit=(args.residue),target_chain='X', device=device,
                seed=0, verbose=True
            )

            # Add uniprot to the design name and accumulate across sequences
            designs.update({f"{uniprot}_{k}": v for k, v in design.items()})

    check_recoded_sequences(designs=designs, residue=args.residue)
    write_fasta(list(designs.values()), list(designs.keys()), out_fp=args.outfile)