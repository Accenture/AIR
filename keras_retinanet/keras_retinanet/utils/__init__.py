"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications copyright (c) 2020-2021 Accenture
"""

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
    if not "AIR_VERBOSE" in os.environ:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    config = config if config else tf.compat.v1.ConfigProto()
    config.intra_op_parallelism_threads = num_par_exec_units
    config.inter_op_parallelism_threads = 2
    config.allow_soft_placement = True
    config.device_count["CPU"] = num_par_exec_units
    
    if isinstance(gpu_id, int) and gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        config.gpu_options.allow_growth = True

    os.environ["OMP_NUM_THREADS"] = str(num_par_exec_units)
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))