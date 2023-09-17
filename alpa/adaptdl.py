import time

class PolluxAgent:
    def __init__(self, state=None):
        self.state = state
        self.iter = 0
        self.t_iters = []
        self.throughputs = []
        self.total_batch_size = None # total batch size (all GPUs)
        self.t_compilation = None
        self.dataset_size = None
        self.alloc_vector = None # allocation vector in AdaptDL
        print("PolluxAgent initialized.")
    
    def report_iteration(self, state, t_iter):
        self.state = state
        self.t_iters.append(t_iter)
        pollux_agent.iter += 1
        self.throughputs.append(self.total_batch_size / t_iter)
        if self.iter % 20 == 0:
            print(f"Iteration #{self.iter}. Time taken - {t_iter} seconds. Throughput - {self.throughputs[-1]} samples/sec.")
    
pollux_agent = PolluxAgent()