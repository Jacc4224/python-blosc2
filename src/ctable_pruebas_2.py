#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""Imports para CTable"""

from __future__ import annotations
from collections.abc import Iterable

from dataclasses import Field
from email.policy import strict
from typing import TYPE_CHECKING, Annotated, Any, Generic, TypeVar, List

import numpy as np
from algoritmia.schemes.dac_scheme import tail_dec_solve
from line_profiler import profile
from numpy.ma.core import append
from pydantic import BaseModel, Field, create_model, ValidationError

import blosc2
from blosc2 import concat, compute_chunks_blocks

""" Imports extra """
import time
import random

import pandas as pd
from typing import Annotated

if TYPE_CHECKING:
    from collections.abc import Iterable

RowT = TypeVar("RowT", bound=BaseModel)


class NumpyDtype:
    def __init__(self, dtype):
        self.dtype = dtype


class MaxLen:
    def __init__(self, length: int):
        self.length = int(length)


class RowModel(BaseModel):
    id: Annotated[int, NumpyDtype(np.int64)] = Field(ge=0)
    c_val: Annotated[complex, NumpyDtype(np.complex128)] = Field(default=0j)
    score: Annotated[float, NumpyDtype(np.float64)] = Field(ge=0, le=100)
    active: Annotated[bool, NumpyDtype(np.bool_)] = True

'''class RowModel(BaseModel):
    id: Annotated[int, NumpyDtype(np.int16)] = Field(ge=0)
    name: Annotated[str, MaxLen(10)] = Field(default="unknown")
    # name: Annotated[bytes, MaxLen(10)] = Field(default=b"unknown")
    score: Annotated[float, NumpyDtype(np.float32)] = Field(ge=0, le=100)
    active: Annotated[bool, NumpyDtype(np.bool_)] = True'''

class RowModel2(BaseModel):
    id: Annotated[int, NumpyDtype(np.int16)] = Field(ge=0)
    #name: Annotated[str, MaxLen(10)] = Field(default="unknown")
    name: Annotated[bytes, MaxLen(10)] = Field(default=b"unknown")





class _RowIndexer:
        def __init__(self, table):
            self._table = table

        def __getitem__(self, item):
            return self._table._run_row_logic(item)


class CTable(Generic[RowT]):

    def __init__(self, row_type: type[RowT], new_data = None, key: str = None, expected_size: int = 1_048_576) -> None:
        self._row_type = row_type
        self._cols: dict[str, blosc2.NDArray] = {}
        self._capacity: int = expected_size
        self._n_rows: int = 0
        self._col_widths: dict[str, int] = {}
        self._col_names = []
        self.row = _RowIndexer(self)
        self._key: str = key
        self._key_set: set[Any] = set()

        c, b = compute_chunks_blocks((expected_size,))
        self._valid_rows = blosc2.zeros(shape=c, dtype = np.bool_ , chunks=c, blocks=b)


        for name, field in row_type.model_fields.items():
            self._col_names.append(name)
            origin = getattr(field.annotation, "__origin__", field.annotation)

            # We need to check for other posibilities...
            if origin == str or field.annotation == str:
                max_len = 32  # Default MaxLen
                if hasattr(field.annotation, "__metadata__"):
                    for meta in field.annotation.__metadata__:
                        if isinstance(meta, MaxLen):
                            max_len = meta.max_length
                            break
                dt = np.dtype(f"U{max_len}")
                display_width = max(10, min(max_len, 50))

            elif origin == bytes or field.annotation == bytes:
                max_len = 32    # Default MaxLen
                if hasattr(field.annotation, "__metadata__"):
                    for meta in field.annotation.__metadata__:
                        if isinstance(meta, MaxLen):
                            max_len = meta.max_length
                            break
                dt = np.dtype(f"S{max_len}")
                display_width = max(10, min(max_len, 50))

            elif origin == int or field.annotation == int:
                dt = np.int64
                display_width = 12

            elif origin == float or field.annotation == float:
                dt = np.float64
                display_width = 15

            elif origin == bool or field.annotation == bool:
                dt = np.bool_
                display_width = 6  # "True" / "False" fit in 5-6 chars

            elif origin == complex or field.annotation == complex:
                dt = np.complex128
                display_width = 25
            else:
                dt = np.object_
                display_width = 20

            final_width = max(len(name), display_width)
            self._col_widths[name] = final_width        # Usefull in __str__

            self._cols[name] = blosc2.zeros(shape=c, dtype=dt, chunks=c, blocks=b)

        if new_data is not None:
            is_append = False

            if isinstance(new_data, (np.void, np.record)):
                is_append = True
            elif isinstance(new_data, np.ndarray):
                if new_data.dtype.names is not None and new_data.ndim == 0:
                    is_append = True
            elif isinstance(new_data, list) and len(new_data) > 0:
                first_elem = new_data[0]
                if isinstance(first_elem, (str, bytes, int, float, bool, complex)):
                    is_append = True

            if is_append:
                self.append(new_data)
            else:
                self.extend(new_data)


    def __str__(self):
        retval = []
        cont = 0

        # We print the header
        for name in self._cols.keys():
            retval.append(f"{name:^{self._col_widths[name]}} |")
            cont += self._col_widths[name]+2
        retval.append("\n")
        for i in range(cont):
            retval.append("-")
        retval.append("\n")


        # We print the rows
        j=0
        for i in range(self._n_rows):
            while not self._valid_rows[j]:
                j += 1
            for name in self._cols.keys():
                retval.append(f"{self._cols[name][j]:^{self._col_widths[name]}}")
                retval.append(f" |")
            retval.append("\n")
            for _ in range(cont):
                retval.append("-")
            retval.append("\n")
            j+=1
        return "".join(retval)

    def __len__(self):
        return self._n_rows

    def head(self, n: int = 5) -> CTable:
        if n <= 0:
            return CTable(self._row_type)

        real_poss = blosc2.where(self._valid_rows, np.array(range(len(self._valid_rows)))).compute()
        n_take = min(n, self._n_rows)

        retval = CTable(self._row_type)
        retval._n_rows = n_take
        retval._valid_rows[:n_take] = True

        for k in self._cols.keys():
            retval._cols[k][:n_take] = self._cols[k][real_poss[:n_take]]

        return retval

    def tail(self, n: int = 5) -> CTable:
        if n <= 0:
            return CTable(self._row_type)

        real_poss = blosc2.where(self._valid_rows, np.array(range(len(self._valid_rows)))).compute()
        start = max(0, self._n_rows - n)
        n_take = min(n, self._n_rows)

        retval = CTable(self._row_type)
        retval._n_rows = n_take
        retval._capacity = n_take
        retval._valid_rows[:n_take] = True

        for k in self._cols.keys():
            retval._cols[k][:n_take] = self._cols[k][real_poss[start:start + n_take]]

        return retval

    def __getitem__(self, s: str):
        return self._cols[s] if s in self._cols else None

    def __getattr__(self, s: str):
        return self._cols[s] if s in self._cols else super().__getattribute__(s)

    def _compact(self):
        real_poss = blosc2.where(self._valid_rows, np.array(range(len(self._valid_rows)))).compute()
        for k, v in self._cols.items():
            v[:self._n_rows] = v[real_poss[:self._n_rows]]
        self._valid_rows[:self._n_rows] = True
        self._valid_rows[self._n_rows:] = False

    # Revisar
    def __setitem__(self, key, value):
        if key not in self._cols.keys():
            raise KeyError(f"Key {key} not in ColumnTable")

        if key == self._key:
            raise KeyError("Cannot modify column set as key")

        if isinstance(value, blosc2.LazyExpr):
            value = value.compute()
        elif isinstance(value, np.ndarray):
            value = blosc2.asarray(value)
        elif not isinstance(value, blosc2.NDArray):
            try:
                value = blosc2.asarray(value)
            except Exception as e:
                raise TypeError(
                    f"Could not turn value for '{key}' to a NDArray. "
                    f"Accepted types: blosc2.NDArray, numpy.ndarray, blosc2.LazyExpr, lists. "
                    f"Error: {e}"
                )


        if len(value) != self._n_rows:
            raise ValueError(
                f"Inconsistent length. Table has {self._nrows} rows, "
                f"but column '{key}' has {value.shape[0]} rows."
            )


        if value.dtype != self._cols[key].dtype:
            raise TypeError(
                    f"Inconsistent dtype. The column '{key}' is of type {self._cols[key].dtype}., "
                    f"but the new value is of type {value.dtype}."
                )
        # 3. Asignar al diccionario interno
        self._cols[key] = value

    @property
    def nrows(self) -> int:
        return self._n_rows

    @property
    def ncols(self) -> int:
        return len(self._cols)

    def info(self):
        """
        nºColumns:
        nºRows:
        Key:

        #   Column  Non-Null Count  Dtype
       ---  ------  --------------  -----
        0   id      50 non-null     int64
        1   name    50 non-null     <U32
        2   score   50 non-null     float32
        3   a

        memory usage:


        """
        ...

    def append(self, data: list | np.void | np.ndarray) -> None:
        is_list = isinstance(data, (list, tuple))
        col_values = list(self._cols.values())
        col_names = self._col_names

        if isinstance(data, dict):
            raise TypeError("Dictionaries are not supported in append.")

        if is_list and len(data) != len(col_values):
            raise ValueError(f"Expected {len(col_values)} values, received {len(data)}")

        if is_list:
            for i, val in enumerate(data):
                target_dtype = col_values[i].dtype
                try:
                    np.array(val, dtype=target_dtype)
                except (ValueError, TypeError):
                    raise TypeError(
                        f"Value '{val}' is not compatible with column '{col_names[i]}' of type {target_dtype}")
        else:
            for name, arr in self._cols.items():
                try:
                    val = data[name]
                except (IndexError, KeyError, ValueError):
                    raise ValueError(f"Input data does not contain required field '{name}'")

                try:
                    np.array(val, dtype=arr.dtype)
                except (ValueError, TypeError):
                    raise TypeError(f"Value '{val}' in field '{name}' is not compatible with type {arr.dtype}")

        pos = 0
        cont = 0
        while cont < self._n_rows:
            if self._valid_rows[pos]:
                cont += 1
            pos += 1


        if is_list:
            for i, col_array in enumerate(col_values):
                col_array[pos] = data[i]
        else:
            for name, col_array in self._cols.items():
                col_array[pos] = data[name]
        self._valid_rows[pos] = True

        self._n_rows += 1

    def delete(self, pos: int | list[int]) -> None:

        if isinstance(pos, list) or isinstance(pos, int):
            if isinstance(pos, list) and not isinstance(pos[0], int):
                raise TypeError("Position must be an integer or a list of integers")
            elif isinstance(pos, int) and pos > self._n_rows:
                raise IndexError("Index out of range")
            elif isinstance(pos, int) and pos < 0:
                pos = self._n_rows + pos
            pos = [pos] if isinstance(pos, int) else pos

            LS = np.array(pos)

            real_poss = blosc2.where(self._valid_rows, np.array(range(len(self._valid_rows)))).compute()

            self._valid_rows[real_poss[LS]] = False
            self._n_rows = self._n_rows - len(pos)

        else:
            raise TypeError("Position must be an integer or a list of integers")

    def extend(self, data: list | CTable | Any) -> None:
        ultimas_validas = blosc2.where(self._valid_rows, np.array(range(len(self._valid_rows)))).compute()
        start_pos = ultimas_validas[-1] + 1 if len(ultimas_validas) > 0 else 0

        current_col_names = self._col_names
        columns_to_insert = []
        new_nrows = 0

        if hasattr(data, "_cols") and hasattr(data, "_n_rows"):
            for name in current_col_names:
                col = data._cols[name][:data._n_rows]
                columns_to_insert.append(col)
            new_nrows = data._n_rows
        else:
            if isinstance(data, np.ndarray) and data.dtype.names is not None:
                for name in current_col_names:
                    columns_to_insert.append(data[name])
                new_nrows = len(data)
            else:
                columns_to_insert = list(zip(*data))
                new_nrows = len(data)

        processed_cols = []
        for i, raw_col in enumerate(columns_to_insert):
            target_dtype = self._cols[current_col_names[i]].dtype
            b2_arr = blosc2.asarray(raw_col, dtype=target_dtype)
            processed_cols.append(b2_arr)

        end_pos = start_pos + new_nrows

        if end_pos >= len(self._valid_rows):
            self._compact()
            ultimas_validas = blosc2.where(self._valid_rows, np.array(range(len(self._valid_rows)))).compute()
            start_pos = ultimas_validas[-1] + 1 if len(ultimas_validas) > 0 else 0
            end_pos = start_pos + new_nrows

        while end_pos > len(self._valid_rows):
            c = len(self._valid_rows)
            for name in current_col_names:
                self._cols[name].resize((c*2,))
            self._valid_rows.resize((c*2,))


        self._n_rows = max(self._n_rows, end_pos) #revisar

        for j, name in enumerate(current_col_names):
            self._cols[name][start_pos:end_pos] = processed_cols[j][:]

        self._valid_rows[start_pos:end_pos] = True

    @profile
    def filter(self, expr_result) -> CTable:
        if not (isinstance(expr_result, (blosc2.NDArray, blosc2.LazyExpr)) and
                (getattr(expr_result, 'dtype', None) == np.bool_)):
            raise TypeError(f"Expected boolean blosc2.NDArray or LazyExpr, got {type(expr_result).__name__}")

        filtro = expr_result.compute() if isinstance(expr_result, blosc2.LazyExpr) else expr_result
        filtro = filtro[:self._n_rows]

        real_poss = blosc2.where(self._valid_rows[:self._n_rows],
                                 np.array(range(self._n_rows))).compute()

        filtro_validas = filtro[real_poss]

        retval = CTable(self._row_type)
        n_true = len(blosc2.where(filtro_validas, np.arange(len(filtro_validas))).compute())
        retval._n_rows = n_true
        retval._capacity = n_true

        for k in self._cols.keys():
            source_filt = self._cols[k][real_poss][filtro_validas]
            retval._cols[k][:n_true] = source_filt

        return retval

    def _run_row_logic(self, ind: int | slice | str) -> list | list[list] | None:
        if isinstance(ind, str):
            try:
                parts = [p.strip() for p in ind.split(':')]
                if len(parts) > 3 or len(parts) < 2:
                    raise ValueError

                slice_args = [int(p) if p else None for p in parts]

                return self._run_row_logic(slice(*slice_args))

            except ValueError:
                raise ValueError(
                    f"Invalid slice string format: '{ind}'. Expected format 'start:stop' or 'start:stop:step' with integers.")

        if isinstance(ind, int):
            if 0 <= ind < self._n_rows:
                index = ind
            elif -self._n_rows <= ind < 0:
                index = self._n_rows + ind
            else:
                raise IndexError("list index out of range")

            return [col[index][()] for col in self._cols.values()]

        if isinstance(ind, slice):
            indices = range(*ind.indices(self._n_rows))
            return [self._run_row_logic(i) for i in indices]

        if isinstance(ind, (list, tuple)) or (isinstance(ind, Iterable) and not isinstance(ind, str)):
            return [self._run_row_logic(i) for i in ind]

        raise TypeError(
            f"Invalid argument type. Expected 'int' or 'slice', "
            f"but got '{type(ind).__name__}'."
        )



    """Save y load por revisar: ha habido cambios como _key"""

    def save(self, urlpath: str, group: str = "table") -> None:
        """
        Persist columns into a single TreeStore container.
        Each column is stored under a group / colname.
        """
        with blosc2.TreeStore(urlpath, mode="w") as ts:
            arrays = self._cols
            for name, arr in arrays.items():
                node_path = f"{group}/{name}"
                # Store as compressed NDArray inside the tree
                print(f"Storing {name} with shape {arr[:self._n_rows].shape} and dtype {arr.dtype} in {node_path}")
                ts[node_path] = arr[:self._n_rows]

    @classmethod
    def load(cls, urlpath: str, group: str = "table", row_type: type[RowT] | None = None) -> CTable:
        with blosc2.TreeStore(urlpath, mode="r") as ts:
            keys = list(ts.keys()) if hasattr(ts, "keys") else []
            prefix = f"/{group}/" if not group.startswith("/") else f"{group}/"
            field_names = []
            for k in keys:
                k_norm = f"/{k.strip('/')}"
                if k_norm.startswith(prefix):
                    field_names.append(k_norm[len(prefix):])

            field_names.sort()

            b2_arrays: dict[str, blosc2.NDArray] = {}
            annotations: dict[str, Any] = {}
            defaults: dict[str, Any] = {}
            loaded_nrows = 0

            for field in field_names:
                node_path = f"{group}/{field}"
                stored_array = ts[node_path]
                b2_arr = blosc2.asarray(stored_array)
                b2_arrays[field] = b2_arr

                if loaded_nrows == 0 and b2_arr.shape[0] > 0:
                    loaded_nrows = b2_arr.shape[0]

                if row_type is None:
                    dt = b2_arr.dtype
                    kind = dt.kind

                    if kind == "U":
                        char_size = np.dtype("U1").itemsize
                        max_len = dt.itemsize // char_size
                        annotations[field] = Annotated[str, MaxLen(int(max_len))]
                        defaults[field] = Field(default="")
                    elif kind == "S":
                        max_len = dt.itemsize
                        annotations[field] = Annotated[bytes, MaxLen(int(max_len))]
                        defaults[field] = Field(default=b"")
                    elif kind in ("i", "u"):
                        annotations[field] = Annotated[int, NumpyDtype(dt)]
                        defaults[field] = Field(default=0)
                    elif kind == "f":
                        annotations[field] = Annotated[float, NumpyDtype(dt)]
                        defaults[field] = Field(default=0.0)
                    elif kind == "c":
                        annotations[field] = Annotated[complex, NumpyDtype(dt)]
                        defaults[field] = Field(default=0j)
                    elif kind == "b":
                        annotations[field] = Annotated[bool, NumpyDtype(dt)]
                        defaults[field] = Field(default=False)
                    else:
                        annotations[field] = Annotated[Any, NumpyDtype(dt)]
                        defaults[field] = Field(default=None)

            if row_type is None:
                model_fields = {}
                for name, ann in annotations.items():
                    model_fields[name] = (ann, defaults.get(name, ...))
                row_type = create_model("InferredRowModel", __base__=BaseModel, **model_fields)

            tbl = cls(row_type)

            tbl._cols = b2_arrays
            tbl._n_rows = loaded_nrows
            tbl._capacity = loaded_nrows
            return tbl





if __name__ == "__main__":
    import numpy as np
    import time
    import random

    # Generación masiva actualizada

    n_rows = 10_000_000
    data_masiva = []
    for i in range(n_rows):
        data_masiva.append([
            i,
            complex(i, i * 0.1),  # Valor complejo en lugar del string f"User_{i}"
            random.random() * 100,
            i % 2 == 0
        ])

    tabla = CTable(RowModel)
    tabla2=CTable(RowModel)
    start = time.perf_counter()
    tabla.extend(data_masiva)
    stop = time.perf_counter()
    print(f"Tiempo extend: {stop - start:.4f} s")
    print(len(tabla))

    start = time.perf_counter()
    tabla2.extend(tabla)
    stop = time.perf_counter()
    print(f"Tiempo extend CTable: {stop - start:.4f} s")
    print(len(tabla2))


    numeros = random.sample(range(0, 100_000), 50_000)
    start = time.perf_counter()
    tabla.delete(numeros)
    stop = time.perf_counter()
    print(f"Tiempo delete: {stop - start:.4f} s")
    print(len(tabla))

    start = time.perf_counter()
    tabla._compact()
    stop = time.perf_counter()
    print(f"Tiempo compact: {stop - start:.4f} s")
    print(len(tabla))



