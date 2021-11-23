"""
Cost model optimizer based on SRTuner
"""

import heapq
import logging
import time

import numpy as np
from ..utils import sample_ints
from .model_based_tuner import ModelOptimizer, knob2point, point2knob
import copy

from SRTuner import SRTunerModule, FlagInfo

logger = logging.getLogger('autotvm')

FLOAT_MAX = float('inf')

class Wrapper:
    def __init__(self, score, tryout):
        self.score = score
        self.tryout = tryout

    def __lt__(self, other):
        assert(self.score is not None and other.score is not None)
        assert(self.tryout is not None and other.tryout is not None)

        return self.score < other.score
        #return self.tryout[2] < other.tryout[2]

    def getIndex(self):
        assert(self.tryout is not None)
        return self.tryout[2]

    def unwrap(self):
        return self.tryout


def random_walk(p, dims):
    """random walk as local transition

    Parameters
    ----------
    p: int
        index of the ConfigEntity
    dims: Array of int
        sizes of each dimension

    Returns
    -------
    new_p: int
        new neighborhood index
    """
    # transform to knob form
    old = point2knob(p, dims)
    new = list(old)

    # mutate
    while new == old:
        from_i = np.random.randint(len(old))
        to_v = np.random.randint(dims[from_i])
        new[from_i] = to_v

    # transform to index form
    return knob2point(new, dims)


"""
def getNumOptions(app_driver, stdOptLv, idx):
    assert(idx>=0)
    flag = app_driver.availFlags[stdOptLv][idx]
    return len(flag[1])


def getIndex(app_driver, mask):
    index = 0
    assert(len(mask) == len(app_driver.availFlags[1])+1)
    for entry in zip(mask[1:], app_driver.availFlags[1]):
        index += entry[0] * app_driver.posMap[entry[1][0]]
    return index


def getMask(app_driver, task, index):
    config = task.config_space.get(index)
    mask = [0]
    for item in app_driver.availFlags[1]:
        key = item[0]
        val = config[key]
        for i, v in enumerate(list(task.config_space.space_map[key])):
            if str(v) == str(val):
                mask.append(i)
                break

    assert(getIndex(app_driver, mask) == index)
    return mask
"""


class SRTunerOptimizer(ModelOptimizer):
    """parallel SRTuner optimization algorithm

    Parameters
    ----------
    task: Task
        The tuning task
    n_iter: int
        The number of iterations of SRTuner
    temp: float or Array of float
        If is a single float, then use a constant temperature.
        If is an Array, then perform linear cooling from temp[0] to temp[1]
    early_stop: int, optional
        Stop iteration if the optimal set do not change in `early_stop` rounds
    log_interval: int, optional
        Print log every `log_interval` iterations
    """
    def __init__(self, optimizer_name, task, n_iter=500,  persistent=True, parallel_size=128,
            early_stop=50, log_interval=50, default_perf=FLOAT_MAX):  #NOTE: Both two were 50
        super(SRTunerOptimizer, self).__init__(optimizer_name)

        self.task = task
        self.dims = [len(x) for x in self.task.config_space.space_map.values()]

        self.n_iter = n_iter
        self.persistent = persistent
        self.parallel_size = min(parallel_size, len(self.task.config_space))
        self.early_stop = early_stop or 1e9
        self.log_interval = log_interval
        
        self.tryouts = None
        self.points = None

              
        self.config_space = self.task.config_space.space_map
        self.config_space_keys = list(self.config_space.keys())
        self.config_space_values = list(self.config_space.values())
        self.numFlags = len(self.config_space_values)

        # define search space
        class TVMFlagInfo(FlagInfo):
            def __init__(self, name, configs, raw_configs):
                super().__init__(name, configs)
                self.raw_configs = raw_configs

        self.posMap = dict()
        search_space = dict()
        pos = 1
        for i in range(self.numFlags):
            key = self.config_space_keys[i]
            configs = self.config_space_values[i]
            numConfigs = len(self.config_space_values[i])
            self.posMap[key] = pos
            pos *= numConfigs
            search_space[key] = TVMFlagInfo(key, [j for j in range(numConfigs)], configs)
        
        self.SRTunerMod = SRTunerModule(search_space, default_perf = default_perf)


    def getIndex(self, opt_setting):
        index = 0
        assert(len(opt_setting) == self.SRTunerMod.num_optimizations)
        for opt_name, config in opt_setting.items():
            index += config * self.posMap[opt_name]
        return index

    def get_opt_setting(self, tvm_opt_setting):
        opt_setting = dict()
        for i in range(self.numFlags):
            key = self.config_space_keys[i]
            found = -1
            for j, config in enumerate(self.SRTunerMod.search_space[key].raw_configs):
                if config == tvm_opt_setting[key]:
                    found = j
                    break
            assert(found>=0)
            opt_setting[key] = found
        return opt_setting

    
    # Expansion during this process should be abandoned.
    # Method1: Deepcopy e.g., dummy_root = copy.deepcopy(app_driver.root)
    #       -> Might be really slow with huge tree
    # Method2: Expand and rollback. Drop all the changes except the selected ones.
    # Method3: Don't expand any node during find_maximums
    def find_maximums(self, model, num, exclusive):
        tic = time.time()
        n_iter, early_stop, log_interval = (
            self.n_iter,
            self.early_stop,
            self.log_interval,
        )

        points = []
        assert not self.persistent, "Persistent mode is not supported for now"
        
         # build heap and insert initial points
        heap_items = [(float("-inf"), -1 - i) for i in range(num)]
        heapq.heapify(heap_items)
        in_heap = set(exclusive)
        in_heap.update([x[1] for x in heap_items])

        k = 0
        k_last_modify = 0
        
        while k < n_iter and k < k_last_modify + early_stop:
            tryouts = self.SRTunerMod.generate_candidates(num, enable_expansion=False)
            new_points = [ self.getIndex(tryout) for tryout in tryouts ]
            new_scores = model.predict(new_points)
    
            for s, p in zip(new_scores, new_points):
                if s > heap_items[0][0] and p not in in_heap:
                    pop = heapq.heapreplace(heap_items, (s, p))
                    in_heap.remove(pop[1])
                    in_heap.add(p)
                    k_last_modify = k
            k += 1

            if log_interval and k % log_interval == 0:
                logger.debug("SRTuner Opt iter: %d\tlast_update: %d\tmax-0: %.2f\tmax-1: %.2f\t"
                             "elapsed: %.2f",
                             k, k_last_modify, heap_items[0][0],
                             np.max([v for v, _ in heap_items]), time.time() - tic)

        
        heap_items.sort(key=lambda item: -item[0])
        heap_items = [x for x in heap_items if x[0] >= 0]
        logger.debug("SRTuner iter: %d\tlast_update: %d\telapsed: %.2f",
                     k, k_last_modify, time.time() - tic)
        logger.debug("SRTuner Maximums: %s", heap_items)
  
        if self.persistent:
            self.points = points

        return [x[1] for x in heap_items]