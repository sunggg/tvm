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

#from SRTuner import SRTunerModule

logger = logging.getLogger('autotvm')

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
    def __init__(self, task, n_iter=500, temp=(1, 0), persistent=True, parallel_size=128,
            early_stop=50, log_interval=50):  #NOTE: Both two were 50
        super(SRTunerOptimizer, self).__init__()

        #print("{} {} {}".format(persistent, early_stop, n_iter))
        self.task = task
        self.dims = [len(x) for x in self.task.config_space.space_map.values()]

        self.n_iter = n_iter
        self.temp = temp
        self.persistent = persistent
        self.parallel_size = min(parallel_size, len(self.task.config_space))
        self.early_stop = early_stop or 1e9
        self.log_interval = log_interval
        
        self.tryouts = None

        # [TODO] temp
        self.points = None

    def find_maximums(self, model, num, exclusive):
        tic = time.time()
        temp, n_iter, early_stop, log_interval = (
            self.temp,
            self.n_iter,
            self.early_stop,
            self.log_interval,
        )

        if self.persistent and self.points is not None:
            points = self.points
        else:
            points = np.array(sample_ints(0, len(self.task.config_space), self.parallel_size))

        scores = model.predict(points)

        # build heap and insert initial points
        heap_items = [(float("-inf"), -1 - i) for i in range(num)]
        heapq.heapify(heap_items)
        in_heap = set(exclusive)
        in_heap.update([x[1] for x in heap_items])

        for s, p in zip(scores, points):
            if s > heap_items[0][0] and p not in in_heap:
                pop = heapq.heapreplace(heap_items, (s, p))
                in_heap.remove(pop[1])
                in_heap.add(p)

        k = 0
        k_last_modify = 0

        if isinstance(temp, (tuple, list, np.ndarray)):
            t = temp[0]
            cool = 1.0 * (temp[0] - temp[1]) / (n_iter + 1)
        else:
            t = temp
            cool = 0

        while k < n_iter and k < k_last_modify + early_stop:
            new_points = np.empty_like(points)
            for i, p in enumerate(points):
                new_points[i] = random_walk(p, self.dims)

            new_scores = model.predict(new_points)

            ac_prob = np.exp(np.minimum((new_scores - scores) / (t + 1e-5), 1))
            ac_index = np.random.random(len(ac_prob)) < ac_prob

            points[ac_index] = new_points[ac_index]
            scores[ac_index] = new_scores[ac_index]

            for s, p in zip(new_scores, new_points):
                if s > heap_items[0][0] and p not in in_heap:
                    pop = heapq.heapreplace(heap_items, (s, p))
                    in_heap.remove(pop[1])
                    in_heap.add(p)
                    k_last_modify = k

            k += 1
            t -= cool

            if log_interval and k % log_interval == 0:
                t_str = "%.2f" % t
                logger.debug(
                    "SA iter: %d\tlast_update: %d\tmax-0: %.2f\tmax-1: %.2f\ttemp: %s\t"
                    "elapsed: %.2f",
                    k,
                    k_last_modify,
                    heap_items[0][0],
                    np.max([v for v, _ in heap_items]),
                    t_str,
                    time.time() - tic,
                )

        heap_items.sort(key=lambda item: -item[0])
        heap_items = [x for x in heap_items if x[0] >= 0]
        logger.debug(
            "SA iter: %d\tlast_update: %d\telapsed: %.2f", k, k_last_modify, time.time() - tic
        )
        logger.debug("SA Maximums: %s", heap_items)

        if self.persistent:
            self.points = points

        return [x[1] for x in heap_items]


    """
    # Expansion during this process should be abandoned.
    # Method1: Deepcopy e.g., dummy_root = copy.deepcopy(app_driver.root)
    #       -> Might be really slow with huge tree
    # Method2: Expand and rollback. Drop all the changes except the selected ones.
    # Method3: Don't expand any node during find_maximums
    def find_maximums(self, app_driver, model, num, exclusive):
        #print("\n  -- find maximums...\n")
        tic = time.time()
        temp, n_iter, early_stop, log_interval = \
                self.temp, self.n_iter, self.early_stop, self.log_interval

        excludes = exclusive.copy()

        if self.persistent and (self.tryouts is not None):
            tryouts = self.tryouts
            points = [ tryout[2] for tryout in tryouts ]
            masks = [ tryout[1] for tryout in tryouts ]
            excludes.update(points)
        else:
            tryouts = batch_mc_tryout(app_driver, self.parallel_size, is_sim = True, exclude = excludes)
            points = [ tryout[2] for tryout in tryouts ]
            masks = [ tryout[1] for tryout in tryouts ]
            excludes.update(points)
        
        try:
            old_points = points.copy()
            points = []
            for point in old_points:
                if point is not None:
                    points.append(point)

            if len(points) > 0:
                scores = model.predict(points)
            else:
                scores = []
        except TypeError:
            print("\nPoints:")
            print(points)
            sys.exit(-1)


        # build heap and insert initial points
        heap_items = [(float('-inf'), Wrapper(float('-inf'), [None, None, -1-i])) for i in range(num)]
        #heap_items = [(float('-inf'), [None, None, - 1 - i]) for i in range(num)]
        heapq.heapify(heap_items)
        in_heap = set(excludes)
        in_heap.update([x[1].getIndex() for x in heap_items])
        #in_heap.update([x[1][2] for x in heap_items])

        for s, p in zip(scores, tryouts):
            if s > heap_items[0][0] and p[2] not in in_heap:
                try:
                    pop = heapq.heapreplace(heap_items, (s, Wrapper(s, p)))
                except TypeError:
                    print("\nExcludes: {}\nPoints: {}\n".format(excludes, points))
                    print("\nType Error\n ==> heap item {} \n ==> s {} / p {}".format(heap_items[0], s, p))
                    assert(0)

                in_heap.remove(pop[1].getIndex())
                in_heap.add(p[2])

        k = 0
        k_last_modify = 0

        if isinstance(temp, (tuple, list, np.ndarray)):
            t = temp[0]
            cool = 1.0 * (temp[0] - temp[1]) / (n_iter + 1)
        else:
            t = temp
            cool = 0

        while k < n_iter and k < k_last_modify + early_stop:
            tryouts = batch_mc_tryout(app_driver, self.parallel_size, is_sim = True, exclude = excludes)
            new_points = [ tryout[2] for tryout in tryouts ]
            new_scores = model.predict(new_points)
            excludes.update(new_points)

            assert(min([ x[0] for x in heap_items ]) == heap_items[0][0])

            for s, p in zip(new_scores, tryouts):
                if s > heap_items[0][0] and p[2] not in in_heap:
                    try:
                        pop = heapq.heapreplace(heap_items, (s, Wrapper(s, p)))
                    except TypeError:
                        print("\nExcludes: {}\nPoints: {}\n".format(excludes, new_points))
                        print("\nType Error\n ==> heap item {} \n ==> s {} / p {}".format(heap_items[0], s, p))
                        assert(0)
                    
                    in_heap.remove(pop[1].getIndex())
                    in_heap.add(p[2])
                    k_last_modify = k

            k += 1
            t -= cool

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
  
        tryouts = [x[1].unwrap() for x in heap_items]
        if self.persistent:
            self.tryouts = tryouts

        return tryouts
        """