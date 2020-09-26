# import numba
from numba.core import registry
from numba import types
import dpctl.ocldrv as ocldrv


class TargetDispatcher():
    __numba__ = 'py_func'

    def __init__(self, py_func, wrapper, target):
        
        self.py_func = py_func
        self.target = target
        self.wrapper = wrapper
        self.disp = self.get_current_disp(self.target)
        self.__compiled = {}

    def __call__(self, *args, **kwargs):
        if not self.disp in self.__compiled.keys():
            self.__compiled[self.disp] = self.wrapper(self.py_func, self.disp)
        
        return self.__compiled[self.disp](*args, **kwargs)

    @property
    def _numba_type_(self):
        if not self.disp in self.__compiled.keys():
            self.__compiled[self.disp] = self.wrapper(self.py_func, self.disp)

        return types.Dispatcher(self.__compiled[self.disp])
    
    def get_current_disp(self, target):
        gpu_env = None
        if ocldrv.runtime.has_gpu_device():
            gpu_env = ocldrv.runtime.get_gpu_device().get_env_ptr()
        current_env = None
        if ocldrv.runtime.has_current_device():
            current_env = ocldrv.runtime.get_current_device().get_env_ptr()
        if (gpu_env == current_env) and ocldrv.is_available():
            from numba.dppl import dppl_offload_dispatcher
            return registry.dispatcher_registry['__dppl_offload_gpu__']

        return registry.dispatcher_registry[target]
