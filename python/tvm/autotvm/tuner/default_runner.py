from tvm import autotvm
from .tuner import Tuner
from ..task.space import FallbackConfigEntity

class DefaultRunner(Tuner):
    """Enumerate the search space in a random order

    Parameters
    ----------
    task: autotvm.task.Task
        Tuning Task

    range_idx: Optional[Tuple[int, int]]
        A tuple of index range to random
    """
    def __init__(self, task, opts=None):
        super(DefaultRunner, self).__init__(task)
        self.opts = opts
        self.config_space = self.task.config_space.space_map
        self.config_space_keys = list(self.config_space.keys())
        self.config_space_values = list(self.config_space.values())
        self.numFlags = len(self.config_space_values)

    
   
    def next_batch(self, batch_size):

        models = ['1080ti', 'titanx', 'tx2', 'jetson-nano', 'v100' ]
        ret = []

        for model in models:
            cfg = FallbackConfigEntity()
            func_name = self.task.name
            target = self.task.target

            ref_log = autotvm.tophub.load_reference_log(
                             target.kind, model, func_name)
                            
            cfg.__copy__(self.task.config_space)
            cfg.fallback_with_reference_log(ref_log)
            ret.append(cfg)
        
        return ret

    def has_next(self):
        return True