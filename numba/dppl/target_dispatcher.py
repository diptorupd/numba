from numba.core import registry, serialize, dispatcher
from numba import types
from numba.core.errors import UnsupportedError
import dpctl
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

    def __get__(self, obj, objtype=None):
        return self.get_compiled().__get__(obj, objtype)

    def __repr__(self):
        return self.get_compiled().__repr__()

    @classmethod
    def _rebuild(cls, py_func, wrapper, target, compiled):
        self = cls(py_func, wrapper, target, compiled)
        return self

    def get_compiled(self, target=None):
        if target is None:
            target = self.__target

        disp = self.get_current_disp()
        if not disp in self.__compiled.keys():
            self.__compiled[disp] = self.__wrapper(self.__py_func, disp)

        return self.__compiled[disp]

    def get_current_disp(self):
        if dpctl.is_in_device_context():
            # TODO: Add "with cpu context" behaviour
            from numba.dppl import dppl_offload_dispatcher
            return registry.dispatcher_registry['__dppl_offload_gpu__']
        if self.__target is None:
            self.__target = 'cpu'
        return registry.dispatcher_registry[self.__target]

    def _reduce_states(self):
        return dict(
            py_func=self.__py_func,
            wrapper=self.__wrapper,
            target=self.__target,
            compiled=self.__compiled
        )
