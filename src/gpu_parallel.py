from collections import namedtuple
from multiprocessing import Manager, get_logger
import logging
import os
import sys

logger = get_logger()

Context = namedtuple('Context', ['code_root', 'data_root', 'gpu_holder', 'gpu_holder_lock'])

def setup_logging(root):
    # logging is alredy setup, just return
    if 'log_to_both' in str(sys.stdout.write):
        return

    pid = os.getpid()
    # log filename should be unique for each process
    log_file = os.path.join(root, f"logs/{pid}.log")

    # setup log format to include process id and time
    log_format = logging.Formatter('%(asctime)s %(process)d %(message)s')

    # setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_format)

    # setup logger
    local_logger = logging.getLogger()
    local_logger.setLevel(logging.CRITICAL)
    local_logger.addHandler(file_handler)

    # make logger not print to stdout
    local_logger.propagate = False

    # make a helper function to log to both log_file and logger, but skip empty lines
    def log_to_both(msg):
        # strip msg including new line
        msg = msg.strip()
        if msg:
            logger.critical(f"{pid}: {msg}")
            local_logger.critical(msg)

    sys.stdout.write = log_to_both
    
def create_context(code_root, data_root, gpus_available):
    manager = Manager()
    gpu_holder_dict = manager.dict()
    for gpu_id in gpus_available:
        gpu_holder_dict[gpu_id] = -1
    return Context(code_root, data_root, gpu_holder_dict, manager.Lock())

def get_holder_lock():
    # create a shared GPU holder dict for all processes
    manager = Manager()
    gpu_holder = manager.dict() # {gpu_id: process_id}
    gpu_holder_lock = manager.Lock()
    return gpu_holder, gpu_holder_lock

def allocate_gpu(gpu_holder, gpu_holder_lock):
    # get the process id
    process_id = os.getpid()  
    with gpu_holder_lock:
        print(f"Process {process_id} is trying to allocate a GPU\n")
        print(gpu_holder)
        target_gpu_id = None
        # first, check if the process already has a gpu
        for gpu_id, pid in gpu_holder.items():
            if pid == process_id:
                target_gpu_id = gpu_id
        # if not, allocate a gpu
        if target_gpu_id is None:
            for gpu_id, pid in gpu_holder.items():
                if pid == -1:
                    gpu_holder[gpu_id] = process_id
                    target_gpu_id = gpu_id
                    break
    if target_gpu_id is not None:
        # set the gpu with env variable
        print(f"Allocated GPU {target_gpu_id} for process {process_id}\n")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(target_gpu_id)
        return target_gpu_id
    else:
        raise Exception(f'No GPU available for process {process_id}')    

def design_af_with_mpnn_bias_parallel(context, 
        uniprot_id, letter_to_redesign, reference_seq, mpnn_bias_temp, seed, include_neighbors, redesign_radius, top_to_take, 
        config_version=None, quick=False, skip_completed=True):
    setup_logging(context.data_root)
    try:
        allocate_gpu(context.gpu_holder, context.gpu_holder_lock)

        from common import init
        init(context.code_root)
        from common import memory
        from afdesign import design_af_with_mpnn_bias

        design_af_with_mpnn_bias_cached = memory.cache(design_af_with_mpnn_bias)
        seq, metric, recode_positions = design_af_with_mpnn_bias(context.code_root, data_root=context.data_root, uniprot_id=uniprot_id, 
            letter_to_redesign=letter_to_redesign, reference_seq=reference_seq, mpnn_bias_temp=mpnn_bias_temp, seed=seed, 
            include_neighbors=include_neighbors, redesign_radius=redesign_radius, top_to_take=top_to_take, 
            quick=quick, skip_completed=skip_completed, config_version=config_version)

        return {'uniprot_id': uniprot_id, 'seq': seq, 'metric': metric, 'seed': seed, 'mpnn_bias_temp': mpnn_bias_temp, 
                'redesign_radius': redesign_radius, 'include_neighbors': include_neighbors, 'recode_positions': recode_positions}
    except Exception as e:
        import traceback
        exception_string = traceback.format_exc()
        print(f"Exception: {exception_string}, Uniprot: {uniprot_id}")
        raise e

def generate_mpnn_designs_parallel(context, gene_name, uniprot_id, reference_seq, letter_to_redesign,
    include_neighbors, temp, mpnn_designs_num, redesign_radius, top_to_take, config_version=None):
    setup_logging(context.data_root)
    try:
        allocate_gpu(context.gpu_holder, context.gpu_holder_lock)
        from common import init
        init(context.code_root)
        from recode_structure import generate_mpnn_designs

        designs = generate_mpnn_designs(context.data_root, context.code_root, gene_name, uniprot_id, reference_seq, letter_to_redesign,
            include_neighbors, temp, mpnn_designs_num, redesign_radius, top_to_take, config_version)
        return designs
    except Exception as e:
        import traceback
        print(f"Exception: {traceback.format_exc()}, Uniprot: {uniprot_id}")
        raise e

def score_designs_parallel(context, gene_name, uniprot_id, reference_seq, letter_to_redesign, redesign_radius, top_to_take, mpnn_designs_num,
    multimer=False, single_chain=False, config_version=None, llm_designs=True, method='mpnn'):
    setup_logging(context.data_root)
    try:
        allocate_gpu(context.gpu_holder, context.gpu_holder_lock)
        from common import init
        init(context.code_root)
        from common import memory

        from recode_structure import score_designs
        score_designs_cached = memory.cache(score_designs)
        print(f"Scoring designs for {uniprot_id} with args: gene_name={gene_name}, letter_to_redesign={letter_to_redesign}, redesign_radius={redesign_radius}, top_to_take={top_to_take}, mpnn_designs_num={mpnn_designs_num}, multimer={multimer}, single_chain={single_chain}, config_version={config_version}, llm_designs={llm_designs}, method={method}")
        results = score_designs(context.data_root, context.code_root, gene_name, uniprot_id, reference_seq, letter_to_redesign, redesign_radius, top_to_take,
            mpnn_designs_num, multimer, single_chain, config_version, llm_designs, method)
        results['uniprot_id'] = uniprot_id
        results['gene_name'] = gene_name
    except Exception as e:
        # get exception and stack info to the string
        import traceback
        exception_string = traceback.format_exc()
        print(f"Exception: {exception_string}")
        raise e
    return results


