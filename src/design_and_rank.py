import argparse
import logging
import multiprocessing
import os
import sys
from datetime import datetime
from os import path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

script_dir = path.dirname(__file__)
code_root = os.path.abspath(os.path.join(script_dir, '../../'))
print(f"Code root: {code_root}")
mpnn_path = os.path.abspath(os.path.join(code_root, 'ProteinMPNN'))
print(f"Mpnn path: {mpnn_path}")

sys.path.append(code_root)
sys.path.append(mpnn_path)

from common import init
cache_dir = os.path.expanduser('~/.cache/recoding')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(os.path.join(cache_dir, 'logs'), exist_ok=True)
init(cache_dir)

from gpu_parallel import create_context, design_af_with_mpnn_bias_parallel, generate_mpnn_designs_parallel, score_designs_parallel
from recode_structure import save_designs_fasta

def load_genes_from_fasta(fasta_file):
    genes = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        parts = record.id.split('_')
        if len(parts) >= 2:
            uniprot_id, gene = parts[0], parts[1]
        else:
            gene, uniprot_id = record.id, record.id
        genes.append({'gene': gene, 'uniprot_id': uniprot_id, 'full_seq': str(record.seq)})
    return pd.DataFrame(genes)

def generate_designs(target_genes, method, gpus, output_dir, data_root, code_root, num_designs=24, quick=False):
    context = create_context(code_root, data_root, gpus)
    arguments = []

    if method == 'afdesign_mpnn_bias':
        target_genes = target_genes[target_genes.full_seq.str.len() < 500]
        temps = [1.0, 0.5, 0.1, 0.05, 0.02, 0.01][:max(1, num_designs // 4)]
        seeds = [0, 1][:max(1, (num_designs // 2) // len(temps))]

        for _, row in target_genes.iterrows():
            for mpnn_bias_temp in temps:
                for seed in seeds:
                    for include_neighbors in [True, False]:
                        arguments.append((context, row.uniprot_id, 'I', row.full_seq, mpnn_bias_temp, seed,
                                        include_neighbors, 5, 0, None, quick))

        with multiprocessing.get_context('spawn').Pool(len(gpus)) as pool:
            results = pool.starmap(design_af_with_mpnn_bias_parallel, arguments, 1)

        designs_df = pd.DataFrame(results)
        designs_df['recode_positions'] = designs_df.recode_positions.apply(lambda x: ','.join(map(str, x)) if not np.any(pd.isna(x)) else x)
        designs_df = designs_df.drop_duplicates()
        designs_df = designs_df.merge(target_genes[['uniprot_id', 'gene']], on='uniprot_id', how='left')
        af_designs_dir = os.path.join(output_dir, 'Recode_AF_MPNN_designs')
        os.makedirs(af_designs_dir, exist_ok=True)
        csv_path = path.join(af_designs_dir, 'designs_af_mpnn_bias.csv')
        try:
            existing = pd.read_csv(csv_path)
            designs_df = pd.concat([existing, designs_df]).drop_duplicates()
        except FileNotFoundError:
            pass
        designs_df.to_csv(csv_path, index=False)

    elif method == 'mpnn':
        temps = [0.1, 0.5][:max(1, num_designs // 4)]
        num_per_config = max(1, num_designs // (len(temps) * 2))

        for _, row in target_genes.iterrows():
            for include_neighbors in [True, False]:
                for temp in temps:
                    arguments.append((context, row.gene, row.uniprot_id, row.full_seq, 'I',
                                    include_neighbors, temp, num_per_config, 5, 0, None))

        with multiprocessing.get_context('spawn').Pool(len(gpus)) as pool:
            results = pool.starmap(generate_mpnn_designs_parallel, arguments, 1)

        designs_list = [design for result in results for design in result]
        designs_df = pd.DataFrame(designs_list)
        recode_designs_dir = os.path.join(output_dir, 'recode_designs')
        os.makedirs(recode_designs_dir, exist_ok=True)
        csv_path = path.join(recode_designs_dir, 'designs_mpnn.csv')
        try:
            existing = pd.read_csv(csv_path)
            designs_df = pd.concat([existing, designs_df]).drop_duplicates()
        except FileNotFoundError:
            pass
        designs_df.to_csv(csv_path, index=False)

    date = datetime.now().strftime("%Y-%m-%d")
    fasta_records = [SeqRecord(Seq(row.seq), id=f"{row.gene}_{row.uniprot_id}", description='')
                     for _, row in designs_df.iterrows()]
    recode_designs_dir = os.path.join(output_dir, 'recode_designs')
    os.makedirs(recode_designs_dir, exist_ok=True)
    SeqIO.write(fasta_records, path.join(recode_designs_dir, f'designs_raw_{method}_{date}.fasta'), 'fasta')

    return designs_df

def score_all_designs(target_genes, gpus, output_dir, data_root, code_root, method):
    context = create_context(code_root, data_root, gpus)
    arguments = [(context, row.gene, row.uniprot_id, row.full_seq, 'I', 5, 0, 50, False, False, None, False, method)
                 for _, row in target_genes.iterrows()]

    with multiprocessing.get_context('spawn').Pool(len(gpus)) as pool:
        results = pool.starmap(score_designs_parallel, arguments, 1)

    results_df = pd.concat(results)
    date = datetime.now().strftime("%Y-%m-%d")
    recode_designs_dir = os.path.join(output_dir, 'recode_designs')
    os.makedirs(recode_designs_dir, exist_ok=True)
    results_df.to_csv(path.join(recode_designs_dir, f'all_results_{date}.csv'))
    return results_df

def pareto_front(X):
    is_pareto = np.ones(X.shape[0], dtype=bool)
    for i, c in enumerate(X):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(X[is_pareto] < c, axis=1)
            is_pareto[i] = True
    return is_pareto

def rank_mpnn_designs(results_df, target_genes, designs_df):
    final_designs = pd.DataFrame()

    for gene in results_df.gene_name.unique():
        uniprot_id = target_genes[target_genes.gene == gene].uniprot_id.values[0]
        x_metric, y_metric = 'dgram_cce', 'pseudo_log_likelihood'

        for include_neighbors in [True, False]:
            lib = 'MPNN' if include_neighbors else 'MPNN_no_n'
            mpnn_designs = results_df[(results_df.uniprot_id == uniprot_id) & (results_df.lib == lib)]
            x_rows = mpnn_designs[(mpnn_designs.metric == x_metric) & (mpnn_designs.model == 'avg')].sort_values(['lib', 'seq'], ascending=False)
            y_rows = mpnn_designs[(mpnn_designs.metric == y_metric)].sort_values(['lib', 'seq'], ascending=False)
            y_rows.score = -y_rows.score

            best_designs = pd.DataFrame()
            while len(best_designs) < 4 and len(x_rows) > 0:
                X = np.array([x_rows.score.values, y_rows.score.values]).T
                is_p = pareto_front(X)
                pareto_rows = x_rows[is_p].reset_index(drop=True)
                percentiles = np.linspace(0, 100, 4 - len(best_designs))
                pareto_percentiles = np.percentile(pareto_rows.score.values, percentiles, method='nearest')
                indices = [pareto_rows[pareto_rows.score == p].index[0] for p in pareto_percentiles]
                best_designs = pd.concat([best_designs, pareto_rows.loc[np.unique(indices)]])
                x_rows = x_rows[~x_rows.seq.isin(best_designs.seq)]
                y_rows = y_rows[~y_rows.seq.isin(best_designs.seq)]

            best_designs['gene'] = gene
            best_designs['uniprot_id'] = uniprot_id
            final_designs = pd.concat([final_designs, best_designs[['gene', 'uniprot_id', 'seq', 'initial_seq']]])

    return final_designs

def rank_afdesign_designs(results_df, target_genes, designs_df):
    final_designs = pd.DataFrame(columns=['seq', 'uniprot_id', 'initial_seq', 'gene'])

    for uniprot_id in target_genes.uniprot_id.unique():
        for include_neighbors in [True, False]:
            lib = 'afdesign_mpnn_bias' if include_neighbors else 'afdesign_mpnn_bias_no_n'
            protein_scores = results_df[(results_df.uniprot_id == uniprot_id) & (results_df.model == 'avg') & (results_df.lib == lib)].copy()
            protein_scores = protein_scores[['seq', 'metric', 'score', 'initial_seq']]
            afdesigns = designs_df[(designs_df.uniprot_id == uniprot_id) & (designs_df.include_neighbors == include_neighbors)]

            initial_seq = protein_scores[protein_scores.metric == 'dgram_cce'].iloc[0].initial_seq if len(protein_scores) > 0 else ''
            afdesigns = afdesigns[afdesigns.seq.str.len() == len(initial_seq)]
            if initial_seq.startswith('M'):
                afdesigns = afdesigns[afdesigns.seq.str.startswith('M')]

            gene = target_genes[target_genes.uniprot_id == uniprot_id].gene.values[0]
            for mpnn_bias_temp in [0.5, 0.1, 0.05, 0.02]:
                composite_scores = protein_scores[protein_scores.metric == 'composite']
                afdesigns_merge = afdesigns.merge(composite_scores, on='seq', how='inner')
                ordered = afdesigns_merge[(afdesigns_merge.mpnn_bias_temp == mpnn_bias_temp) &
                                         ~afdesigns_merge.seq.isin(final_designs['seq'])].sort_values('score')
                if len(ordered) > 0:
                    best = ordered.iloc[[-1]].copy()
                    best['gene'] = gene
                    final_designs = pd.concat([final_designs, best[['seq', 'uniprot_id', 'initial_seq', 'gene']]])

    return final_designs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta_file', help='FASTA file with genes to design (absolute path or relative to data_dir)')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--method', choices=['mpnn', 'afdesign_mpnn_bias'], required=True, help='Design method to use')
    parser.add_argument('--dry_run', action='store_true', help='Generate minimal designs for testing')
    parser.add_argument('--data_dir', default='.', help='Data directory for input and output (default: current directory)')
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_dir)
    fasta_file = args.fasta_file if os.path.isabs(args.fasta_file) else os.path.join(data_root, args.fasta_file)
    output_dir = os.path.join(data_root, 'output')
    logs_dir = os.path.join(data_root, 'logs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(os.path.join(data_root, 'Recode_AF_MPNN_designs'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'recode_designs'), exist_ok=True)

    # Set up logging
    log_file = os.path.join(logs_dir, f'design_and_rank_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting design_and_rank with method={args.method}, dry_run={args.dry_run}")
    logger.info(f"Data directory: {data_root}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Logs directory: {logs_dir}")

    target_genes = load_genes_from_fasta(fasta_file)
    gpus = list(range(args.gpus))

    num_designs = 2 if args.dry_run else 24
    designs_df = generate_designs(target_genes, args.method, gpus, output_dir, data_root, code_root, num_designs, quick=args.dry_run)
    results_df = score_all_designs(target_genes, gpus, output_dir, data_root, code_root, args.method)

    if args.method == 'mpnn':
        final_designs = rank_mpnn_designs(results_df, target_genes, designs_df)
        lib_name = 'mpnn'
    else:
        final_designs = rank_afdesign_designs(results_df, target_genes, designs_df)
        lib_name = 'afdesign_mpnn_bias'

    date = datetime.now().strftime("%Y-%m-%d")
    output_path = path.join(output_dir, f'ranked_designs_{args.method}')
    save_designs_fasta(final_designs, output_path, lib_name, 'I', target_genes)
    logger = logging.getLogger(__name__)
    logger.info(f"Saved {len(final_designs)} designs to {output_path}_I_{date}.fasta")

if __name__ == '__main__':
    main()
