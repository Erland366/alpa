import time

class PolluxAgent:
    def __init__(self, state=None):
        self.state = state
        self.iter = 0
        self.t_iters = []
        self.t_grads = []
        self.throughputs = []
        self.total_batch_size = None # total batch size (all GPUs)
        self.t_compilation = None
        self.dataset_size = None
        self.alloc_vector = None # allocation vector in AdaptDL
        self.training_dp_cost = None
        print("PolluxAgent initialized.")
    
    def report_iteration(self, state, t_iter):
        assert self.total_batch_size != None, "Batch size should be set in the training code using pollux_agent.total_batch_size"
        self.state = state
        self.t_iters.append(t_iter)
        pollux_agent.iter += 1
        self.throughputs.append(self.total_batch_size / t_iter)
        if self.iter % 20 == 0: # printing for debugging purposes # TODO remove this
            print(f"Iteration #{self.iter}. Time taken - {t_iter} seconds. Throughput - {self.throughputs[-1]} samples/sec. \
                All-time average throughput - {sum(self.throughputs) / len(self.throughputs)}.\n \
                    Training DP cost - {self.training_dp_cost}")
    
pollux_agent = PolluxAgent()