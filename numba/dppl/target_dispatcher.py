from numba.core import registry
from numba import types
import dpctl.ocldrv as ocldrv


class TargetDispatcher():
    __numba__ = 'py_func'

    def __init__(self, py_func, wrapper, target):
        self.py_func = py_func
        self.target = target
        self.wrapper = wrapper
        self.__compiled = {}
        self.__doc__ = py_func.__doc__
        self.__name__ = py_func.__name__
        self.__module__ = py_func.__module__

    def __call__(self, *args, **kwargs):
        return self.get_compiled()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.get_compiled(), name)

    @property
    def _numba_type_(self):
        return types.Dispatcher(self.get_compiled())

    def get_compiled(self):
        disp = self.get_current_disp(self.target)
        if not disp in self.__compiled.keys():
            self.__compiled[disp] = self.wrapper(self.py_func, disp)

        return self.__compiled[disp]
    
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
