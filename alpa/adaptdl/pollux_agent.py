import time
import numpy as np
from numpy.typing import NDArray
from typing import TypeVar
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF, DotProduct
from collections import defaultdict
import pickle
import os
import atexit
import threading
import asyncio
from collections.abc import Iterable
import jax.numpy as jnp
import alpa
from typing import Union, Optional, List, Tuple, Set, Dict
from sklearn.base import RegressorMixin
import wandb

linear_rbf_kernel = DotProduct() + RBF() + WhiteKernel(noise_level_bounds=(1e-10, 1e5)) # lower bound lowered to avoid a warning

class PolluxAgent:
    def __init__(self, state=None):
        self.NUM_SYNC_PER_CONFIG = 100 # number of times to synchronize iterations to measure T_iter for each configuration
        self.IS_COMPIL_THRESHOLD = 2 # number of seconds, above which time spent counts as compilation overhead
        self.BS_RECOMPILATION_FACTOR_ENABLED = True

        self.state = state
        self.train_metric = None
        self.iter = 0
        self.t_iters = []
        self.t_grads = []
        self.throughputs = []
        self._total_batch_size = None # total batch size (all GPUs)
        self.last_state_retrieved_batch_size = None
        self.t_compilation = {}
        self.dataset_size = None
        self._alloc_vector = None # allocation vector in AdaptDL

        self.alloc_config_regressor : Dict[Tuple[int, int, int], RegressorMixin] = defaultdict(init_regressor)

        self.bs_t_iter = defaultdict(list)
        self.bs_t_exec_timecosts = defaultdict(list)
        self.bs_t_diff = defaultdict(list)

        self.config_t_iter = defaultdict(list) # dictionary of config : T_iter (from self.get_current_config())
        
        self.periodic_bs_sync_enabled = True
        self.bs_sync_starttime = None
        self.bs_sync_interval = 30 # seconds

        self.scheduler_update_last_time = None
        self.scheduler_update_interval = 10 # seconds

        self.wandb_logging_enabled = True
        self.wandb_log_last_time = None
        self.wandb_log_interval = 10 # seconds
        
        self.scheduler_enabled = False
        self.scheduler_address = None
        self.namespace = "Alpa-AdaptDL-Ray-NameSpace"
        self.job_id = None
        self.reallocation_approaching = False
        self.update_dataloader_batchsize = False

        self.p_train_step = None

        self.training_started_for_config = defaultdict(bool)

        # PGNS-related parameters.
        # Normally, they are DistributedArray's, need to be materialized (causes sync) before sending/using
        self.grad_norm_sqr_abstract = None
        self.grad_variance_abstract = None
        self.grad_norm_sqr = None
        self.grad_variance = None
        #

        # Dataloader-related parameters
        self.max_batch_size = None
        self.local_bsz_bounds = None
        self.init_batch_size = None
        #

        self.fixed_regressors = False

        self.start_time = time.time()
        self.total_overhead_time = float()
        self.overhead_time_list = list()

        # print("PolluxAgent initialized.")

    def init_sched_utils(self):
        from alpa.adaptdl.sched_requests import release_resources
        atexit.register(release_resources)

        from alpa.adaptdl.websocket_client import start_websocket_client
        thread = threading.Thread(target=start_websocket_client, daemon=True)
        thread.start()
        
    @property
    def total_batch_size(self):
        return self._total_batch_size
    
    @total_batch_size.setter
    def total_batch_size(self, new_total_batch_size):
        new_total_batch_size = int(new_total_batch_size)
        self._total_batch_size = new_total_batch_size


    @property
    def alloc_vector(self):
        return self._alloc_vector
    
    @alloc_vector.setter
    def alloc_vector(self, new_alloc_vector):
        self._alloc_vector = new_alloc_vector


    def fix_regressors(self):
        self.fixed_regressors = True
        self.NUM_SYNC_PER_CONFIG = 0

    
    def get_current_config(self):
        num_gpus = self._alloc_vector[0] # assumes that each allocated node has the same # of GPUs - Alpa
        num_nodes = len(self._alloc_vector)
        return (self._total_batch_size, num_gpus, num_nodes)


    def get_current_alloc_config(self):
        num_gpus = self._alloc_vector[0]
        num_nodes = len(self._alloc_vector)
        return (num_gpus, num_nodes)


    def count_bs_observations(self, alloc_config=None):
        if alloc_config is None:
            alloc_config = self.get_current_alloc_config()
        count = {}
        for k, v in self.config_t_iter.items():
            if (k[1], k[2]) == (alloc_config):
                count[k[0]] = len(v)
        return count
    
    def report_iteration(self, state, train_metric, t_iter=None, executable_time_cost=None):
        # assert self.total_batch_size != None, "Total batch size should be set in the training code using pollux_agent.total_batch_size"
        self.state = state
        self.train_metric = train_metric
        if t_iter is not None:
            self.config_t_iter[self.get_current_config()].append(t_iter)
        self.iter += 1
        
        if self.scheduler_enabled and (self.scheduler_update_last_time is None or time.time() - self.scheduler_update_last_time > self.scheduler_update_interval):
            self.pickle_and_update_scheduler()
            self.scheduler_update_last_time = time.time()
        if self.wandb_logging_enabled and (self.wandb_log_last_time is None or time.time() - self.wandb_log_last_time > self.wandb_log_interval):
            self.wandb_log()
            self.wandb_log_last_time = time.time()
        if self.iter % 500 == 0:
            self._save_objects(f'pickle_objects/objects_iteration{self.iter}.pkl')
        if not self.training_started_for_config[self.get_current_config()]:
            self.training_started_for_config[self.get_current_config()] = True
        # if self.iter % 100 == 0:
        #     print(f"Throughput for each seen allocation:")
        #     self._fit_config_iter()
        #     for alloc in self.alloc_config_regressor.keys():
        #         print(f"Allocation {alloc} - {self.predict_throughput(self.total_batch_size, alloc_config=alloc)}")
        #     print(f"Count of observations: {self.count_bs_observations()}")
        #     print(f"Median T_iter list - {list([np.median(np.sort(np.array(l))) for l in self.config_t_iter.values()])}")
        #     print(f"Parameters of regressors:")
        #     for alloc, regressor in self.alloc_config_regressor.items():
        #         print(f"Allocation {alloc} - coef: {regressor.coef_}, intercept: {regressor.intercept_}")


    def pickle_and_update_scheduler(self):
        dumped = pickle.dumps(self)
        from alpa.adaptdl.sched_requests import update_state
        update_state(dumped)


    def log_async(self, data):
        def log_thread():
            wandb.log(data)
        threading.Thread(target=log_thread).start()
    
    
    def wandb_log(self):
        from alpa.adaptdl.goodput import GoodputFunction

        if not self.scheduler_enabled:
            if isinstance(self.grad_norm_sqr_abstract, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)) \
                and isinstance(self.grad_variance_abstract, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
                self.grad_norm_sqr = self.grad_norm_sqr_abstract._value.item()
                self.grad_variance = self.grad_variance_abstract._value.item()

        stat_eff_table = None
        throughput_table = None
        goodput_table = None
        current_stat_eff = None 
        current_throughput = None
        current_goodput = None

        if self.grad_norm_sqr is not None and self.grad_variance is not None and self.local_bsz_bounds is not None:
            goodput_fn = GoodputFunction((self.grad_norm_sqr, self.grad_variance), self.init_batch_size)
            min_batch_size = jnp.maximum(self.init_batch_size, self.local_bsz_bounds[0] * alpa.get_global_num_devices())
            batch_size = jnp.geomspace(min_batch_size, self.max_batch_size)
            eps = 1e-8
            batch_size = jnp.ceil(batch_size - eps)
            stat_eff = goodput_fn.efficiency(batch_size)
            throughput = jnp.ravel(self.predict_throughput(batch_size))
            goodput = stat_eff * throughput
            table_data_se = [[x, y] for (x, y) in zip(batch_size, stat_eff)]
            table_data_throughput = [[x, y] for (x, y) in zip(batch_size, throughput)]
            table_data_goodput = [[x, y] for (x, y) in zip(batch_size, goodput)]
            stat_eff_table = wandb.Table(data=table_data_se, columns=["batch_size", "SE"])
            throughput_table = wandb.Table(data=table_data_throughput, columns=["batch_size", "Throughput"])
            goodput_table = wandb.Table(data=table_data_goodput, columns=["batch_size", "Goodput"])
            current_stat_eff = goodput_fn.efficiency(self.total_batch_size)
            current_throughput = self.predict_throughput(self.total_batch_size)[0]
            current_goodput = current_stat_eff * current_throughput

        data = {
            "loss": self.train_metric['loss']._value, 
            "lr": self.train_metric['learning_rate']._value,
            "job_age": self.get_job_age(),
            "pure_training_time": self.get_job_age() - self.total_overhead_time,
            "time": time.time(),
            "total_overhead_time": self.total_overhead_time,
            "batch_size": self.total_batch_size,
            "num_gpus": alpa.get_global_num_devices(),
            "grad_norm_sqr": self.grad_norm_sqr,
            "grad_variance": self.grad_variance,
            "SE_vs_BS": wandb.plot.line(stat_eff_table, "batch_size", "SE", title="Statistical Efficiency vs. Batch Size Plot") if stat_eff_table is not None else None,
            "current_stat_eff": current_stat_eff,
            "Throughput_vs_BS": wandb.plot.line(throughput_table, "batch_size", "Throughput", title="Throughput vs. Batch Size Plot") if throughput_table is not None else None,
            "current_throughput": current_throughput,
            "Goodput_vs_BS": wandb.plot.line(goodput_table, "batch_size", "Goodput", title="Goodput vs. Batch Size Plot") if goodput_table is not None else None,
            "current_goodput": current_goodput,
            }

        # Asynchronously logging wandb data so that training is not blocked. Each log otherwise takes ~1s.
        self.log_async(data)

    
    def __getstate__(self):
        """
        This function is to delete non-material variables of the PolluxAgent object (e.g., DistributedArrays) because
        they cannot be pickled. This should only affect pickling.
        """
        state = self.__dict__.copy()
        del state['p_train_step']
        del state['state']
        del state['train_metric']
        # TODO: also make sure that PGNS value is materialized
        if isinstance(self.grad_norm_sqr_abstract, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)) \
             and isinstance(self.grad_variance_abstract, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)):
            self.grad_norm_sqr = self.grad_norm_sqr_abstract._value.item()
            self.grad_variance = self.grad_variance_abstract._value.item()
        del state['grad_norm_sqr_abstract']
        del state['grad_variance_abstract']
        # Another approach to avoid data race - create separate variables for materialized versions of PGNS values, and delete DistributedArray variables from the dict here
        return state


    def __setstate__(self, state):
        self.__dict__.update(state)
        # self.p_train_step = None

    
    def _fit_config_iter(self):
        #assert len(self.bs_t_exec_timecosts) >= 2, "At least 2 batch size - execution time costs are required to fit the regressor."

        if self.fixed_regressors:
            return
        
        x_bs = np.array(list(self.config_t_iter.keys()))
        y_dp = np.array(list([np.median(np.sort(np.array(l))) for l in self.config_t_iter.values()]))

        alloc_config_columns = x_bs[:, 1:3]
        alloc_configs = np.unique(alloc_config_columns, axis=0)

        for alloc_config in alloc_configs:
            indices = np.where((x_bs[:, 1] == alloc_config[0]) & (x_bs[:, 2] == alloc_config[1]))[0]

            x_bs_subset = x_bs[indices, 0].reshape(-1, 1)
            y_dp_subset = y_dp[indices]

            regressor = self.alloc_config_regressor[tuple(alloc_config)]

            regressor.fit(x_bs_subset, y_dp_subset)
        
    def predict_t_iter(self, batch_sizes, alloc_config): # TODO: handle typing
        #assert len(self.bs_t_exec_timecosts) >= 2, "At least 2 batch size - execution time costs are required to make predictions."
        # assert batch_sizes.ndim == 2 and batch_sizes.shape[1] == 1, "Input batch sizes np.ndarray should be of shape (N, 1)."
        self._fit_config_iter()
        return self.alloc_config_regressor[alloc_config].predict(np.array(batch_sizes).reshape(-1, 1))
    
    def predict_throughput(self, batch_sizes, alloc_config=None):
        # TODO: clean up unnecessary reshapes
        if alloc_config is None:
            alloc_config = self.get_current_alloc_config()
        if not isinstance(batch_sizes, Iterable):
            batch_sizes = [batch_sizes]
        elif isinstance(batch_sizes, jnp.DeviceArray) and batch_sizes.ndim == 0:
            batch_sizes = [float(batch_sizes)]
        return np.array(batch_sizes).reshape(-1, 1) / self.predict_t_iter(batch_sizes, alloc_config).reshape(-1, 1)

    def predict_t_iter_from_configs(self, configs): # TODO: handle typing
        raise NotImplementedError
        self._fit_config_iter()
        return self.bs_t_iter_regressor.predict(np.array(configs))

    def predict_throughput_from_configs(self, configs):
        """
        Example usage:      pollux_agent.predict_throughput_from_configs([(128, 4, 1), (128, 2, 1), (256, 1, 2)])
        """
        raise NotImplementedError
        return np.array(configs)[:, 0].reshape(-1, 1) / self.predict_t_iter_from_configs(configs).reshape(-1, 1)

    def get_job_age(self):
        return time.time() - self.start_time

    def expected_recompilation_overhead(self):
        if len(self.overhead_time_list) > 1:
            return self.overhead_time_list[1]
        elif len(self.overhead_time_list) == 1:
            return self.overhead_time_list[0]
        else:
            return None
        
    def _save_objects(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename,'wb') as f:
            pickle.dump(self, f)

def init_regressor():
    return LinearRegression()
    
pollux_agent = PolluxAgent()