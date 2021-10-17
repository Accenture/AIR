import os
import random
import numpy as np
import tensorflow as tf

def seed(seed):
    """ fix random seeds to minimize random variance """
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

def optimize_tf_parallel_processing(num_par_exec_units, config=None, gpu_id=None):
    config = config if config else tf.compat.v1.ConfigProto()
    config.intra_op_parallelism_threads = num_par_exec_units
    config.inter_op_parallelism_threads = 2
    config.allow_soft_placement = True
    config.device_count["CPU"] = num_par_exec_units
    
    if isinstance(gpu_id, int):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        config.gpu_options.allow_growth = True

    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    os.environ["OMP_NUM_THREADS"] = str(num_par_exec_units)
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"