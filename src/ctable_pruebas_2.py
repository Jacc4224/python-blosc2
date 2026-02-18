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
from typing import TYPE_CHECKING, Annotated, Any, Generic, TypeVar, List

import numpy as np
from line_profiler import profile
from numpy.ma.core import append
from pydantic import BaseModel, Field, create_model, ValidationError

import blosc2
from blosc2 import concat, compute_chunks_blocks, where

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
        while cont < self._n_rows:          #Cambiar por nueva sintaxis más rápida
            if self._valid_rows[pos]:
                cont += 1
            pos += 1

        #validar que el dáto no se vaya a poner en última posición

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

        filter = expr_result.compute() if isinstance(expr_result, blosc2.LazyExpr) else expr_result

        target_len = len(self._valid_rows)

        if len(filter) > target_len:
            filter = filter[:target_len]
        elif len(filter) < target_len:
            padding = blosc2.zeros(target_len, dtype=np.bool_)
            padding[:len(filter)] = filter[:]
            filter = padding

        filter = (filter & self._valid_rows).compute()
        new_nrows = blosc2.count_nonzero(filter)

        retval = CTable(self._row_type, expected_size=target_len)

        for k, v in retval._cols.items():
            v[:] = self._cols[k][:]

        retval._valid_rows = filter
        retval._n_rows = int(new_nrows)
        retval._compact()


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
        ...

    @classmethod
    def load(cls, urlpath: str, group: str = "table", row_type: type[RowT] | None = None) -> CTable:
        ...





if __name__ == "__main__":
    import numpy as np
    import time
    import random

    # Generación masiva actualizada

    n_rows = 10_000_000
    data_masiva = []
    print(f"Creando {n_rows} filas.")

    start = time.perf_counter()
    for i in range(n_rows):
        data_masiva.append([
            i,
            complex(i, i * 0.1),
            random.random() * 100,
            i % 2 == 0
        ])
    stop = time.perf_counter()
    print(f"Tiempo creación de datos: {stop - start:.4f} s")
    print("------------------------------------------------------")

    tabla = CTable(RowModel)
    tabla2=CTable(RowModel)
    start = time.perf_counter()
    tabla.extend(data_masiva)
    stop = time.perf_counter()
    print(f"Tiempo extend: {stop - start:.4f} s")
    print("------------------------------------------------------")

    start = time.perf_counter()
    tabla2.extend(tabla)
    stop = time.perf_counter()
    print(f"Tiempo extend CTable: {stop - start:.4f} s")
    print("------------------------------------------------------")



    numeros = random.sample(range(0, 100_000), 50_000)
    start = time.perf_counter()
    tabla.delete(numeros)
    stop = time.perf_counter()
    print(f"Tiempo delete: {stop - start:.4f} s")
    print("------------------------------------------------------")


    start = time.perf_counter()
    tabla._compact()
    stop = time.perf_counter()
    print(f"Tiempo compact: {stop - start:.4f} s")
    print("------------------------------------------------------")



    filtro = ((tabla['id'] <=1_000) & (tabla['score'] > 80)).compute()
    start = time.perf_counter()
    tabla2 = tabla.filter(filtro)
    stop = time.perf_counter()
    print(f"Tiempo filter: {stop - start:.4f} s")
    print("------------------------------------------------------")



    # En el main
    total_comprimido = sum(col.cbytes for col in tabla._cols.values()) + tabla._valid_rows.cbytes
    total_sin_comprimir = sum(col.nbytes for col in tabla._cols.values()) + tabla._valid_rows.nbytes

    print(f"Comprimido: {total_comprimido / 1024 ** 2:.2f} MB")
    print(f"Sin comprimir: {total_sin_comprimir / 1024 ** 2:.2f} MB")
    print(f"Ratio: {total_comprimido / total_sin_comprimir:.2%}")
