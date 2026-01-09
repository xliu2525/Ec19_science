import os
from os import path
import hashlib

from Bio.PDB import PDBParser, Polypeptide
import pandas as pd
from gemmi import cif
import atomium
from atomium import Model

from log import dlog


def positions_hash(positions_list):
    return hashlib.sha1(str(positions_list).encode()).hexdigest()[:6]

def extract_chains_ids(pdb_path, target_chain):
    # Extract the chain names from the PDB file using the BioPython library, return sequence of the target_chain
    parser = PDBParser()
    structure = parser.get_structure("X", pdb_path)
    chains = structure[0]
    chain_ids = []
    initial_seq = None
    for chain in chains:
        chain_ids.append(chain.id)
        if chain.id == target_chain:
            dlog(chain, chains)
            initial_seq = "".join([Polypeptide.protein_letters_3to1.get(res.get_resname(), 'X') for res in chain.get_residues()])
    return chain_ids, initial_seq

def extract_all_chains(pdb_path):
    # Return dict of chain_id: sequence
    parser = PDBParser()
    structure = parser.get_structure("X", pdb_path)
    pdb_chains = structure[0]
    chains = {}
    for chain in pdb_chains:
        chains[chain.id] = "".join([Polypeptide.protein_letters_3to1.get(res.get_resname(), 'X') for res in chain.get_residues()])
    return chains

def extract_fixed_chains(pdb_path, target_chain):
    from protein_mpnn_utils import parse_PDB
    res = parse_PDB(pdb_path)
    fixed_chains = []
    data = res[0]
    initial_seq = None
    for key in data:
        if 'seq_chain_' in key:
            letter = key[-1]
            if letter < 'R':
                fixed_chains.append(letter)
        if key == f'seq_chain_{target_chain}':
            initial_seq = data[key]
        
    return fixed_chains, initial_seq

def get_pdb(pdb_code=""):
  if pdb_code is None or pdb_code == "":
    upload_dict = files.upload()
    pdb_string = upload_dict[list(upload_dict.keys())[0]]
    with open("tmp.pdb","wb") as out: out.write(pdb_string)
    return "tmp.pdb"
  else:
    os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
    return f"{pdb_code}.pdb"

def seq_diff(init_seq, designed_seq):
    for i, (init_aa, gen_aa) in enumerate(zip(init_seq, designed_seq)):
        if init_aa != gen_aa:
            print(f"Diff at {i}, {init_aa}->{gen_aa}")

def get_uniprot_chain_mapping(cif_path):
    doc = cif.read_file(cif_path)  # copy all the data from mmCIF file
    block = doc.sole_block()  # mmCIF has exactly one block
    chain_ids = block.find_values("_struct_ref_seq.pdbx_strand_id")
    uniprot_ids = block.find_values("_struct_ref_seq.pdbx_db_accession")
    
    output = {}
    for uniprot_id, chain_id in zip(uniprot_ids, chain_ids):
        if uniprot_id not in output:
            output[uniprot_id] = chain_id
            
    return output           

def parse_modified_residues(cif_path):
    doc = cif.read_file(cif_path)  # copy all the data from mmCIF file
    block = doc.sole_block()  # mmCIF has exactly one block
    chain_ids = block.find_values("_pdbx_struct_mod_residue.auth_asym_id")
    res_names = block.find_values("_pdbx_struct_mod_residue.auth_comp_id")
    res_indices = block.find_values("_pdbx_struct_mod_residue.auth_seq_id")
    replace_names = block.find_values("_pdbx_struct_mod_residue.parent_comp_id")

    chain_ids_diff = block.find_values("_struct_ref_seq_dif.pdbx_pdb_strand_id")
    res_names_diff = block.find_values("_struct_ref_seq_dif.mon_id")
    res_indices_diff = block.find_values("_struct_ref_seq_dif.seq_num")
    replace_names_diff = block.find_values("_struct_ref_seq_dif.db_mon_id")

    # repackage into a dataframe
    result = pd.DataFrame({
        "chain_id": list(chain_ids) + list(chain_ids_diff),
        "res_name": list(res_names) + list(res_names_diff),
        "res_index": list(res_indices) + list(res_indices_diff),
        "replace_name": list(replace_names) + list(replace_names_diff)
    })
    return result[result.replace_name != "?"]

def is_mostly_modified(c):
    non_modified_AAs = [res for res in c.residues() if res.code != 'X']
    return len(non_modified_AAs) < len(c.residues())/2

def is_rna_chain(c):
    return len(c.residues(name__regex="^.?A$|^.?G$|^.?C$|^.?U$")) != 0

def extract_res_index(res):
    # Handle cases like "C.555A" where the index has a letter suffix
    index_part = res.id.split(".")[-1]
    # Extract numeric part from the index
    return ''.join(c for c in index_part if c.isdigit())

def remap_atom_ids(chains, modified_residues_df):
    counter = 1
    remap = {}
    for i_chain, c in enumerate(chains):
        sorted_atoms = sorted(c.atoms(), key=lambda a: a.id)
        for atom in sorted_atoms:
            remap[atom.id] = counter        
            counter += 1
        for atom in c.atoms():
            atom._id = remap[atom.id]

        new_chain_id = 'X' if i_chain == 0 else chr(ord('A')+i_chain-1)
        for i, res in enumerate(c.residues()):
            if res.code == 'X':
                res_index = extract_res_index(res)
                modified_res = modified_residues_df[(modified_residues_df.chain_id == c.id) & (modified_residues_df.res_index == res_index)]
                dlog(modified_res, c.id, res_index)
                assert len(modified_res) > 0, f"Can't find modified residue {res_index} in chain {c.id}"
                assert modified_res.res_name.iloc[0] == res.name, f"Modified residue {res_index} in chain {c.id} has wrong name {res.name} (should be {modified_res.res_name.item()})"
                res._name = modified_res.replace_name.iloc[0]

            res._id = f"{new_chain_id}.{i+1}"
            # if res.name == 'D2T' or res.name == '0TD':
            #     res._name = 'ASP'
            # if res.name == '4D4':
            #     res._name = 'ARG'
            # if res.name == 'KEO':
            #     res._name = 'LYS'
            # if res.name == 'MEQ':
            #     res._name = 'GLN'
            # if res.name == 'MSE':
            #     res._name = 'MET'
            # if res.name == 'OCS':
            #     res._name = 'LYS'
            if res.code == 'X':
                assert (res.code != 'X'), f"Bad residue: {res}"
        c._id = new_chain_id
                
def generate_neighbor_structures(filename, output_dir, chain_uniprot_mapping, reference_genes, 
        letter_to_find_nearby_chains, include_neighbors = True,
        skip_completed=True):
    # Generates PDB file for each chain in chain_uniprot_mapping, containing the chain and its neighbors
    # if letter_to_find_nearby_chains is set, neighbors are determined by chains close to residues matching code
    # (and optionally their neighbors in the chain)
    mismatch_proteins = []
    pdb = atomium.open(filename)

    modified_residues_df = parse_modified_residues(filename)
    dlog(modified_residues_df)

    for assembly_index in range(1, len(pdb.assemblies)+1):
        assembly = pdb.generate_assembly(assembly_index)
        chain_names = [c.id for c in assembly.chains() if not is_rna_chain(c)]

        for chain_name in chain_names:
            uniprot_name = chain_uniprot_mapping.get(chain_name)
            if uniprot_name is None:
                continue
                
            output_file = path.join(output_dir, f"{uniprot_name}_nearby_protein_{letter_to_find_nearby_chains}.pdb")
            output_rna_file = path.join(output_dir, f"{uniprot_name}_nearby_all_{letter_to_find_nearby_chains}.pdb")
            if path.isfile(output_file) and skip_completed:
                continue
            
            print(f"Processing {chain_name} - {uniprot_name}")
            assembly_copy = pdb.generate_assembly(assembly_index)
            assembly_copy.optimise_distances()
            chain = assembly_copy.chain(chain_name)

            full_seq = reference_genes[reference_genes.uniprot_id == uniprot_name].full_seq.item()

            protein_chains = []
            rna_chains = []

            if letter_to_find_nearby_chains:
                # find residues with code matching letter_to_find_nearby_chains
                target_residues = [res for res in chain.residues() if res.code == letter_to_find_nearby_chains]
                if include_neighbors:
                    target_res_indices = [extract_res_index(res) for res in target_residues]
                    # find neighbor residues which have indices within 1 of target residues
                    # so check if index+1 or index-1 is in target_res_indices
                    nearby_residues = []
                    for res in chain.residues():
                        print(res)
                        res_index = extract_res_index(res)
                        print(res_index)
                        if res_index in target_res_indices:
                            continue
                        if str(int(res_index)-1) in target_res_indices or str(int(res_index)+1) in target_res_indices:
                            nearby_residues.append(res)
                    target_residues += nearby_residues

                # find nearby chains for all the atoms in the target residues
                nearby_chains = set()
                for res in target_residues:
                    for atom in res.atoms():
                        nearby_chains.update(atom.nearby_chains(5))

                nearby_chains.discard(chain)
            else:
                nearby_chains = c.nearby_chains(5)

            for c in nearby_chains:
                if is_rna_chain(c):
                    rna_chains.append(c)
                elif is_mostly_modified(c):
                    print(f"Skipping all modified chain {c.id}")
                    continue
                else:
                    protein_chains.append(c)

            all_protein_chains = [chain] + protein_chains
            remap_atom_ids(all_protein_chains, modified_residues_df)
            
            residue_seq = "".join([res.code for res in chain.residues()])
            if full_seq.find(residue_seq) == -1:
                print(f"Can't find sequence in reference for {uniprot_name}")
                mismatch_proteins.append(uniprot_name)
                continue
            
            m = Model(*all_protein_chains)
            m.save(output_file)
            if rna_chains:
                all_chains = [chain] + protein_chains + rna_chains
                m = Model(*all_chains)
                m.save(output_rna_file)
            
    return mismatch_proteins
