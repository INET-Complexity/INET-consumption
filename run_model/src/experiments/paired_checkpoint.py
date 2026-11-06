"""In-memory checkpoints for serial paired validation; never installs run hooks."""

import ctypes
import os
import random
import threading
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from types import FunctionType, MethodType, ModuleType

import numpy as np
from numba import _helperlib
from numba.core.registry import cpu_target
from numba.cpython import randomimpl

from macromodel.timeseries import TimeSeries


class _NumbaState(ctypes.Structure):
    # Numba's _random.c / randomimpl.rnd_state_t, including the normal cache.
    _fields_ = [
        ("index", ctypes.c_int32),
        ("mt", ctypes.c_uint32 * 624),
        ("has_gauss", ctypes.c_int32),
        ("gauss", ctypes.c_double),
        ("is_initialized", ctypes.c_int32),
    ]


def _numba_pointers():
    layout = randomimpl.rnd_state_t
    target = cpu_target.target_context.target_data
    if (
        str(layout) != "{i32, [624 x i32], i32, double, i32}"
        or any(
            getattr(_NumbaState, name).offset != layout.get_element_offset(target, i)
            for i, (name, _) in enumerate(_NumbaState._fields_)
        )
        or ctypes.sizeof(_NumbaState) != layout.get_abi_size(target)
    ):
        raise RuntimeError("Unsupported Numba RNG layout; cannot checkpoint safely.")
    return (_helperlib.rnd_get_np_state_ptr(), _helperlib.rnd_get_py_state_ptr())


@dataclass(frozen=True)
class _RandomState:
    python: tuple
    numpy: tuple
    numba: tuple[bytes, bytes]

    @classmethod
    def capture(cls):
        return cls(
            random.getstate(),
            deepcopy(np.random.get_state()),
            tuple(ctypes.string_at(ptr, ctypes.sizeof(_NumbaState)) for ptr in _numba_pointers()),
        )

    def restore(self):
        pointers = _numba_pointers()  # Validate before altering any streams.
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        for pointer, state in zip(pointers, self.numba, strict=True):
            ctypes.memmove(pointer, state, len(state))


def _clone(model):
    """Preseed every reachable TimeSeries, including on old-main Country objects."""
    memo, series, seen, pending = {}, [], set(), [model]
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, TimeSeries):
            memo[id(value)] = type(value).__new__(type(value))
            series.append(value)
        if isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)
        elif not isinstance(value, (type, FunctionType, MethodType, ModuleType)):
            try:
                pending.extend(object.__getattribute__(value, "__dict__").values())
            except AttributeError:
                pass
    for value in series:
        memo[id(value)].__dict__.update(deepcopy(value.__dict__, memo))
    return deepcopy(model, memo)


class PairedCheckpoint:
    """Snapshot model and seeded RNGs without advancing the model or reseeding.

    Capture before installing experiment hooks: Python closures cannot safely be
    rebound by deepcopy. Use clone() for each arm and activate_rng() around its
    entire serial continuation. NumPy/Python globals and Numba's two model-used
    legacy thread-local streams are restored on entry; ambient states are restored
    even on failure. Object-owned Generators/RandomStates are copied with the model.
    This is an in-memory, same-process/thread API, not a portable checkpoint file
    or a parallel execution mechanism. Numba's private layout is checked fail-loud.
    """

    def __init__(self, model):
        if model.prehooks or model.posthooks:
            raise ValueError("Capture before installing hooks; deepcopy cannot rebind closures.")
        self._owner = (os.getpid(), threading.get_ident())
        self._rng = _RandomState.capture()
        try:
            self._model = _clone(model)
        finally:
            self._rng.restore()

    def _check_owner(self):
        if self._owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError("PairedCheckpoint requires its original process and thread.")

    def clone(self):
        """Return an independent arm; cloning does not activate its global RNGs."""
        self._check_owner()
        ambient = _RandomState.capture()
        try:
            return _clone(self._model)
        finally:
            ambient.restore()

    @contextmanager
    def activate_rng(self):
        """Replay initial global streams for one arm, restoring ambient on exit."""
        self._check_owner()
        ambient = _RandomState.capture()
        try:
            self._rng.restore()
            yield
        finally:
            ambient.restore()
