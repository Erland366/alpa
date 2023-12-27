import time
import numpy as np
from numpy.typing import NDArray
from typing import TypeVar
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import pickle

class PolluxAgent:
    def __init__(self, state=None):
        self.state = state
        self.iter = 0
        self.t_iters = []
        self.t_grads = []
        self.throughputs = []
        self._total_batch_size = None # total batch size (all GPUs)
        self.last_state_retrieved_batch_size = None
        self.t_compilation = None
        self.dataset_size = None
        self.alloc_vector = None # allocation vector in AdaptDL
        self.training_dp_cost = None
        self.bs_dp = {}
        self.bs_dp_regressor = LinearRegression()
        self.bs_exectime_regressor = LinearRegression()
        self.bs_t_iter = defaultdict(list)
        self.bs_t_exec_timecosts = defaultdict(list)
        self.bs_t_diff = defaultdict(list)
        
        self.bs_sync_starttime = None
        self.bs_sync_interval = 30 # seconds
        print("PolluxAgent initialized.")
        
    @property
    def total_batch_size(self):
        return self._total_batch_size
    
    @total_batch_size.setter
    def total_batch_size(self, new_total_batch_size):
        self._total_batch_size = new_total_batch_size
        
    
    def report_iteration(self, state, t_iter, executable_time_cost=None):
        # assert self.total_batch_size != None, "Batch size should be set in the training code using pollux_agent.total_batch_size"
        self.state = state
        self.t_iters.append(t_iter)
        self.bs_t_iter[self.total_batch_size].append(t_iter)
        if executable_time_cost is not None:
            self.bs_t_exec_timecosts[self.total_batch_size].append(executable_time_cost)
            # self.bs_t_exec_timecosts[self.total_batch_size].append(t_iter)
            self.bs_t_diff[self.total_batch_size].append(t_iter - executable_time_cost)
        self.iter += 1
        self.throughputs.append(self.total_batch_size / t_iter if self.total_batch_size is not None else 1 / t_iter)
        if self.iter % 20 == 0 and self.total_batch_size is not None: # printing for debugging purposes # TODO remove this
            print(f"Iteration #{self.iter}. Time taken - {t_iter} seconds. Throughput - {self.throughputs[-1]} samples/sec. \
                Median/mean throughput - {np.median(np.array(self.throughputs))} / {np.mean(np.array(self.throughputs))}.\n \
                    Training DP cost - {self.training_dp_cost} \
                        Median/mean iteration time - {np.median(np.array(self.t_iters))} / {np.mean(np.array(self.t_iters))} \
                            Compilation time - {self.t_compilation}")
            if executable_time_cost is not None:
                print(f"Median current BS {self.total_batch_size} iteration time - {np.median(np.array(self.bs_t_iter[self.total_batch_size]))} \
                    Median current BS {self.total_batch_size} 'pure' execution time - {np.median(np.array(self.bs_t_exec_timecosts[self.total_batch_size]))} \
                    Median current BS {self.total_batch_size} 'sync' time - {np.median(np.array(self.bs_t_diff[self.total_batch_size]))}")
        # if self.iter % 500 == 0:
            # self._save_objects(f'/home/haifatl/Documents/alpa/alpa-adaptdl_8/alpa-adaptdl/alpa/adaptdl/pickle_objects/4gp/objects_iteration{self.iter}.pkl')
            
            
    def _fit_batchsize_dynp(self):
        assert len(self.bs_dp) >= 2, "At least 2 batch size - DynP costs are required to fit the regressor."
        
        x_bs = np.array(list(self.bs_dp.keys())).reshape(-1, 1)
        y_dp = np.array(list(self.bs_dp.values())).reshape(-1, 1)
        
        self.bs_dp_regressor.fit(x_bs, y_dp)
        
    def predict_dynp_cost(self, batch_sizes): # TODO: handle typing
        assert len(self.bs_dp) >= 2, "At least 2 batch size - DynP costs are required to make predictions."
        assert batch_sizes.ndim == 2 and batch_sizes.shape[1] == 1, "Input batch sizes np.ndarray should be of shape (N, 1)."
        self._fit_batchsize_dynp()
        return self.bs_dp_regressor.predict(batch_sizes)
    
    def _fit_batchsize_exectime(self):
        #assert len(self.bs_t_exec_timecosts) >= 2, "At least 2 batch size - execution time costs are required to fit the regressor."
        
        x_bs = np.array(list(self.bs_t_exec_timecosts.keys())).reshape(-1, 1)
        y_dp = np.array(list([np.median(np.array(l)) for l in self.bs_t_exec_timecosts.values()])).reshape(-1, 1)
        
        self.bs_exectime_regressor.fit(x_bs, y_dp)
        
    def predict_exectime(self, batch_sizes): # TODO: handle typing
        #assert len(self.bs_t_exec_timecosts) >= 2, "At least 2 batch size - execution time costs are required to make predictions."
        assert batch_sizes.ndim == 2 and batch_sizes.shape[1] == 1, "Input batch sizes np.ndarray should be of shape (N, 1)."
        self._fit_batchsize_exectime()
        return self.bs_exectime_regressor.predict(batch_sizes)
    
    def predict_throughput(self, batch_sizes):
        if self.training_dp_cost is not None:
            return np.array(batch_sizes).reshape(-1, 1) / self.predict_dynp_cost(np.array(batch_sizes).reshape(-1, 1))
        else:
            print(f"Collected batch sizes before predict - {self.bs_t_exec_timecosts.keys()}")
            return np.array(batch_sizes).reshape(-1, 1) / self.predict_exectime(np.array(batch_sizes).reshape(-1, 1))
        
    def _save_objects(self, filename):
        with open(filename,'wb') as f:
            pickle.dump({'iter': self.iter, 't_iters': self.t_iters, 't_grads': self.t_grads, 'throughputs': self.throughputs, 
                        '_total_batch_size': self._total_batch_size, 'last_state_retrieved_batch_size': self.last_state_retrieved_batch_size,
                        't_compilation': self.t_compilation, 'dataset_size': self.dataset_size, 'alloc_vector': self.alloc_vector,
                        'training_dp_cost': self.training_dp_cost, 'bs_dp': self.bs_dp, 'bs_dp_regressor': self.bs_dp_regressor,
                        'bs_exectime_regressor': self.bs_exectime_regressor, 'bs_t_iter': self.bs_t_iter,
                        'bs_t_exec_timecosts': self.bs_t_exec_timecosts, 'bs_t_diff': self.bs_t_diff},
                        f)
    
pollux_agent = PolluxAgent()