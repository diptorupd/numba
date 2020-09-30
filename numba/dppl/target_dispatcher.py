from numba.core import registry, serialize, dispatcher
from numba import types
import dpctl.ocldrv as ocldrv


class TargetDispatcher(serialize.ReduceMixin, metaclass=dispatcher.DispatcherMeta):
    __numba__ = 'py_func'

    def __init__(self, py_func, wrapper, target, compiled=None):

        self.__py_func = py_func
        self.__target = target
        self.__wrapper = wrapper
        self.__compiled = compiled if compiled is not None else {}
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

    @classmethod
    def _rebuild(cls, py_func, wrapper, target, compiled):
        self = cls(py_func, wrapper, target, compiled)
        return self

    def get_compiled(self):
        disp = self.get_current_disp(self.__target)
        if not disp in self.__compiled.keys():
            self.__compiled[disp] = self.__wrapper(self.__py_func, disp)

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

    def _reduce_states(self):
        return dict(
            py_func=self.__py_func,
            wrapper=self.__wrapper,
            target=self.__target,
            compiled=self.__compiled
        )
