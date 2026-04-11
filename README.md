This repository contains the code accompanying the paper "Toward life with a 19–amino acid alphabet through generative artificial intelligence design." With this pipeline, you can generate and evaluate protein sequences that use only 19 amino acids, leveraging state-of-the-art generative models. The workflow integrates language models (ESM-2 and MSA Transformer) and structure-based methods (AlphaFold2 and ProteinMPNN) for protein design, assessment, and ranking.


## Installation

To set up the environment and dependencies for this pipeline, follow these steps:

### 1. Create and Activate the Conda Environment

It is recommended to use **mamba** (a faster drop-in replacement for conda), but you may also use conda itself.

```bash
cd ec19
mamba env create -f environment.yml
mamba activate ec19_env
```

### 2. Download ProteinMPNN

Clone the [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) repository **next to** this project's root directory (not inside it):

```bash
cd ../ec19
git clone https://github.com/dauparas/ProteinMPNN.git
```
### 3. Download AlphaFold weights

Download AlphaFold weights according to the setup from [ColabDesign](https://github.com/sokrypton/ColabDesign/tree/main/af).

```bash
mkdir params
curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar | tar x -C params
```

### 4. Download TMscore

Download [TMscore](https://aideepmed.com/TM-score/) which is used in scoring designed sequences.

```bash
wget https://aideepmed.com/TM-score/TMscore.cpp
g++ -O3 -o TMscore TMscore.cpp
```

### 5. Download and Setup Required Data

Download structure files and multiple sequence alignments (MSAs) from [Zenodo](https://zenodo.org/records/18155920):

```bash
curl -L https://zenodo.org/api/records/18155920/files-archive -o data.zip
unzip data.zip -d data_dir/
```

**Note:** Replace `data_dir/` with your preferred data directory name or location. This will create the directory and extract the data files required to run the pipeline.

Your data directory should be organized as follows:
```
data_dir/
  genes.fasta                                     # Input FASTA file with wild-type protein sequences
  {UNIPROT_ID}_nearby_protein_I.pdb              # Protein structure file
  {UNIPROT_ID}.i90c90qid30diff384.a3m            # MSA file for the protein
  output/                                        # Output directory (auto-created)
  logs/                                          # Logs directory (auto-created)
```
**Notes:**
- The FASTA file is needed only when designing with protein language models (ESM-2 or MSA Transformer).
- Protein structure (`.pdb`) and multiple sequence alignment (`.a3m`) files are **only needed if you use structure-based methods, design with spatial neighbors, or the MSA Transformer.**
- The `output` and `logs` subdirectories will be generated automatically during runtime if they do not exist.


## Usage

### Input FASTA Format

Your input FASTA file should contain records with headers in the format:  
```
>{Uniprot_ID}_{gene_name}
SEQUENCE
```
**Example:**  
```
>P0A7R9_rpsA
MAVQK... (sequence)
```
- You can include multiple records (genes) in a single FASTA file.
- Each sequence should be the wild-type amino acid sequence for the gene of interest.

We provide a sample input file:  
`data/ecoli_ribosomal_genes.fasta`  
This contains the 50 *E. coli* ribosomal genes designed in the study.

---

### 1. Recoding with Protein Language Models (PLMs)

You can use one of the supported protein language models:

- **esm2**: runs recoding using the ESM-2 sequence-only language model.
- **msa_trans**: runs recoding using the MSA Transformer, which requires a multiple sequence alignment (MSA) in `.a3m` format for each gene.

**Example usage:**
```bash
python recode_lm.py --infile <genes.fasta> --model esm2 --outfile <output.fasta> [additional parameters]
```

#### Key Parameters for `recode_lm.py`

- `--infile`: Path to your input FASTA file. (**required**)
- `--outfile`: Path for saving recoded sequences. (**required**)
- `--model`: Choose which PLM to use. (**required**) Options:
    - `esm2` (ESM-2 language model)
    - `msa_trans` (MSA Transformer; requires `--msa_dir`)
- `--gpu`: GPU index to use (default: 0)
- `--residue`: Residue to recode (default: I; options: I or V)
- `--scheme`: Recoding strategy. (**required**) Options:
    - `simultaneous`: Modify all relevant residues at once (default for most use-cases)
    - `autoregressive`: Sequentially modify residues
    - `gibbs`: Iteratively re-sample residue identities (Gibbs sampling)
- `--msa_dir`: Directory with MSA `.a3m` files (required if using `msa_trans`)
- `--structure_dir`: Directory with PDB files (optional, required if using spatial neighbors)
- `--seq_neighbors`: If set, also recode residues adjacent to target residue in the sequence
- `--spatial_neighbors`: If set, recode residues spatially close to the target (requires PDBs)
- `--evo_neighbors`: If set, recode residues showing evolutionary coupling to target
- `--min_3d_dist` / `--max_3d_dist`: Set thresholds for what counts as a spatial neighbor (in Angstroms)

**Example with spatial neighbors:**
```bash
python recode_lm.py --infile data/ecoli_ribosomal_genes.fasta --model esm2 --outfile my_recoded_seqs.fasta --structure_dir data_dir/ --spatial_neighbors --min_3d_dist 3.0 --max_3d_dist 5.0
```

**Example for MSA Transformer:**
```bash
python recode_lm.py --infile data/ecoli_ribosomal_genes.fasta --model msa_trans --outfile my_recoded_msa.fasta --msa_dir data_dir/
```

For more information about the arguments and their defaults, see [src/recode_lm.py](src/recode_lm.py) or run:
```bash
python recode_lm.py --help
```

### 2. Recoding with Structure-based Models

For structure-based sequence design, you have two main options:

- **mpnn**: Uses [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) to design sequences compatible with a given 3D structure.
- **afdesign_mpnn_bias**: Runs [AFDesign](https://github.com/sokrypton/ColabDesign) with MPNN-based bias for enhanced sequence diversity and structure compatibility.

#### Example Usage

**Quick start/example ("dry run"):**
```bash
python design_and_rank.py data/test.fasta --method mpnn --dry_run --data_dir /path/to/data
```
This runs a configuration check and validates input/output paths without performing full computations, allowing for easier debugging.

**Generate and rank designs using ProteinMPNN across multiple GPUs:**
```bash
python design_and_rank.py data/ecoli_ribosomal_genes.fasta --method mpnn --gpus 4 --data_dir /path/to/data
```

**To use AFDesign + MPNN:**
```bash
python design_and_rank.py data/ecoli_ribosomal_genes.fasta --method afdesign_mpnn_bias --gpus 4 --data_dir /path/to/data
```

#### Key Parameters for `design_and_rank.py`

- `--method`: Structure-based recoding method. (**required**)
    - `mpnn`: Use ProteinMPNN
    - `afdesign_mpnn_bias`: AFDesign with MPNN-biased sampling
- `--infile`: Input FASTA file with gene sequences to design for (positional argument)
- `--data_dir`: Path to data directory (see structure below)
- `--gpus`: Number of GPUs to use for parallel design/ranking (default: 1). Use a single GPU (e.g., `--gpus 1`) if running locally or increase for cluster environments.
- `--dry_run`: If set, validate setup without performing computation
- Other options: See `python design_and_rank.py --help` for advanced controls (e.g., specifying custom structure/model files, output controls, chain selection, etc.)

#### Outputs

- Ranked and designed FASTA files are written to:  
  `data_dir/output/ranked_designs_{method}_I_{date}.fasta`
- Log files and intermediate data are written to `data_dir/logs/` and subfolders.
- **Note:** Due to randomness in neural models and differences in environment or hardware, output sequences may differ across runs, and may not exactly match those from the associated publication.

For full details about arguments and defaults, see [src/design_and_rank.py](src/design_and_rank.py) or run:
```bash
python design_and_rank.py --help
```
