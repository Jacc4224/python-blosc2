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
from numpy.ma.core import append
from pydantic import BaseModel, Field, create_model, ValidationError

import blosc2
from blosc2 import concat
from blosc2.lazyexpr import get_chunk

""" Imports extra """
import time
import random
from line_profiler import profile


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

    def __init__(self, row_type: type[RowT], new_data = None, key: str = None) -> None:
        self._row_type = row_type
        self._cols: dict[str, blosc2.NDArray] = {}
        self._capacity: int = 1
        self._n_rows: int = 0
        self._col_widths: dict[str, int] = {}
        self._col_names = []
        self.row = _RowIndexer(self)
        self._key: str = key
        self._key_set: set[Any] = set()


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

            self._cols[name] = blosc2.zeros(shape=1, dtype=dt)

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
        for i in range(self._n_rows):
            for name in self._cols.keys():
                retval.append(f"{self._cols[name][i]:^{self._col_widths[name]}}")
                retval.append(f" |")
            retval.append("\n")
            for i in range(cont):
                retval.append("-")
            retval.append("\n")
        return "".join(retval)

    def __len__(self):
        return self._n_rows

    def head(self, n: int = 5) -> CTable:
        if not isinstance(n, int):
            raise TypeError("n must be an integer")

        start = 0
        end = min(n, self._n_rows)

        if n <= 0:
            return self.__class__(self._row_type)

        new_table = self.__class__(self._row_type)
        col_names = getattr(self, "_col_names", list(self._cols.keys()))

        count = end - start
        new_table._n_rows = count
        new_table._capacity = count

        for name in col_names:
            source_slice = self._cols[name][start:end]
            new_table._cols[name].resize((count,))
            new_table._cols[name][:] = source_slice

        if self._key is not None:
            new_table._key = self._key
            keys_slice = self._cols[self._key][start:end]
            new_table._key_set = set(keys_slice)

        return new_table

    def tail(self, n: int = 5) -> CTable:
        if not isinstance(n, int):
            raise TypeError("n must be an integer")
        start = max(0, self._n_rows - n)
        end = self._n_rows
        if n <= 0:
            return self.__class__(self._row_type)
        new_table = self.__class__(self._row_type)
        col_names = getattr(self, "_col_names", list(self._cols.keys()))
        count = end - start
        new_table._n_rows = count
        new_table._capacity = count
        for name in col_names:
            source_slice = self._cols[name][start:end]
            new_table._cols[name].resize((count,))
            new_table._cols[name][:] = source_slice
        if self._key is not None:
            new_table._key = self._key
            keys_slice = self._cols[self._key][start:end]
            new_table._key_set = set(keys_slice)
        return new_table

    def __getitem__(self, s: str):
        return self._cols[s] if s in self._cols else None

    def __getattr__(self, s: str):
        return self._cols[s] if s in self._cols else super().__getattribute__(s)


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
        key_value = None
        col_values = list(self._cols.values())
        col_names = self._col_names

        if isinstance(data, dict):
            raise TypeError("Dictionaries are not supported in append.")

        if is_list and len(data) != len(col_values):
            raise ValueError(f"Expected {len(col_values)} values, received {len(data)}")

        if self._key is not None:
            if is_list:
                try:
                    key_idx = col_names.index(self._key)
                    key_value = data[key_idx]
                except ValueError:
                    raise KeyError(f"Key column '{self._key}' does not exist.")
            else:
                try:
                    key_value = data[self._key]
                except (IndexError, KeyError, ValueError):
                    raise KeyError(f"Input data does not contain the key field '{self._key}'")

            if key_value in self._key_set:
                raise KeyError(f"Key '{key_value}' already exists in column '{self._key}'")

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

        self._capacity += 1
        for col_array in col_values:
            col_array.resize((self._capacity,))


        if is_list:
            for i, col_array in enumerate(col_values):
                col_array[self._n_rows] = data[i]
        else:
            for name, col_array in self._cols.items():
                col_array[self._n_rows] = data[name]


        if self._key is not None:
            self._key_set.add(key_value)

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
            LS = list(pos)

            for k, v in self._cols.items():
                sust = v[:]
                sust[LS] = False
                sust = np.flatnonzero(sust)
                self._cols[k] = blosc2.asarray(sust)

            self._capacity = self._capacity - len(pos)

            for col_array in self._cols.values():
                col_array.resize((self._capacity,))
            self._n_rows = self._n_rows - len(pos)

        else:
            raise TypeError("Position must be an integer or a list of integers")

    def extend(self, data: list | CTable | Any) -> None:
        if isinstance(data, dict):
            raise TypeError("Dictionaries are not supported in extend.")

        current_col_names = getattr(self, "_col_names", list(self._cols.keys()))
        current_col_values = [self._cols[name] for name in current_col_names]

        columns_to_insert = []
        new_nrows = 0

        if hasattr(data, "_cols") and hasattr(data, "nrows"):
            if len(data._cols) != len(current_col_names):
                raise ValueError(f"Input CTable has {len(data._cols)} columns, expected {len(current_col_names)}")

            for name in current_col_names:
                if name not in data._cols:
                    raise ValueError(f"Input CTable missing column '{name}'")

                if data._cols[name].dtype != self._cols[name].dtype:
                    raise TypeError(
                        f"Column '{name}' dtype mismatch. Source: {data._cols[name].dtype}, Target: {self._cols[name].dtype}")

                columns_to_insert.append(data._cols[name])

            new_nrows = data.nrows


        elif isinstance(data, (list, tuple, np.ndarray)):
            if len(data) == 0:
                return

            if isinstance(data, np.ndarray) and data.dtype.names is not None:
                columns_to_insert = []
                for name in current_col_names:
                    if name not in data.dtype.names:
                        raise ValueError(f"Input structured array is missing required field '{name}'")
                    columns_to_insert.append(data[name])
                new_nrows = len(data)

            elif isinstance(data[0], (np.void, np.record)):
                try:
                    structured_source = np.array(data, dtype=data[0].dtype)
                    columns_to_insert = []
                    for name in current_col_names:
                        if name not in structured_source.dtype.names:
                            raise ValueError(f"Input data is missing required field '{name}'")
                        columns_to_insert.append(structured_source[name])
                    new_nrows = len(data)

                except Exception:
                    columns_to_insert = []
                    for name in current_col_names:
                        columns_to_insert.append([row[name] for row in data])
                    new_nrows = len(data)

            else:
                first_row = data[0]
                if len(first_row) != len(current_col_names):
                    raise ValueError(
                        f"Rows must have {len(current_col_names)} items, but first row has {len(first_row)}")
                try:
                    columns_to_insert = list(zip(*data, strict=True))

                except TypeError:
                    columns_to_insert = list(zip(*data))

                except ValueError:
                    raise ValueError("Inconsistent row lengths in input data.")

                except Exception:
                    raise ValueError("Error transposing data. Ensure all rows are iterable.")

                if len(columns_to_insert) != len(current_col_names):
                    raise ValueError(f"Data has {len(columns_to_insert)} columns, expected {len(current_col_names)}.")

                new_nrows = len(data)

        else:
            raise TypeError("Data format not supported in extend.")

        if new_nrows == 0:
            return

        processed_cols = []
        for i, raw_col in enumerate(columns_to_insert):
            target_dtype = current_col_values[i].dtype
            try:
                if isinstance(raw_col, blosc2.NDArray) and raw_col.dtype == target_dtype:
                    b2_arr = raw_col
                else:
                    b2_arr = blosc2.asarray(raw_col, dtype=target_dtype)

                processed_cols.append(b2_arr)
            except Exception as e:
                raise TypeError(f"Column {i} ('{current_col_names[i]}') conversion error: {e}")




        for i, name in enumerate(current_col_names):
            target_array = self._cols[name]
            source_array = processed_cols[i]
            if self._n_rows == 0:
                self._cols[name] = source_array
            else:
                self._cols[name] = blosc2.concat([target_array, source_array], axis=0)

            #target_array.resize((self._capacity,))
            #target_array[old_nrows:self._n_rows] = source_array[:]
        old_nrows = self._n_rows
        self._n_rows += new_nrows
        self._capacity = self._n_rows

    @profile
    def filter(self, expr_result) -> CTable:
        filtro = None
        if not (isinstance(expr_result, (blosc2.NDArray, blosc2.LazyExpr)) and expr_result.dtype == np.bool_):
            raise TypeError(
                f"Expected a boolean 'blosc2.NDArray' or 'blosc2.LazyExpr', "
                f"but got type '{type(expr_result).__name__}' "
                f"with dtype '{getattr(expr_result, 'dtype', 'N/A')}'."
            )
        if isinstance(expr_result, blosc2.LazyExpr):
            filtro = expr_result.compute()
        elif isinstance(expr_result, blosc2.NDArray) or (expr_result.dtype != np.bool_):
            filtro = expr_result

        filtro = filtro[:]

        retval = CTable(self._row_type)
        n_true = np.count_nonzero(filtro)
        retval._n_rows = n_true
        retval._capacity = n_true

        if filtro is not None and len(filtro) >= self._n_rows:
            for k in self._cols.keys():
                retval._cols[k] = self._cols[k][filtro]
        else:
            raise ValueError(
                f"Filter length ({len(filtro)}) does not match the number of rows ({self._n_rows})."
            )
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
    n_rows = 200_001
    print(f"Generando {n_rows} filas de prueba con complejos...")

    # Generación masiva actualizada
    data_masiva = []
    for i in range(n_rows):
        data_masiva.append([
            i,
            complex(i, i * 0.1),  # Valor complejo en lugar del string f"User_{i}"
            random.random() * 100,
            i % 2 == 0
        ])

    # Nota: Asegúrate de que RowModel esté actualizado para tener un campo complejo
    # compatible (ej. c_val = Col(dtype=np.complex128) o similar)

    # Instanciamos la tabla vacía
    tabla_test = CTable(RowModel)
    tabla_test.extend(data_masiva)

    print("Comenzando prueba de rendimiento 1...")

    inicio = time.perf_counter()

    tabla_test.extend(data_masiva)

    fin = time.perf_counter()

    tiempo_total = fin - inicio
    velocidad = n_rows / tiempo_total if tiempo_total > 0 else 0

    print(f"\n=== RESULTADOS ===")
    print(f"Filas insertadas: {n_rows:,}")
    print(f"Tiempo total:     {tiempo_total:.4f} segundos")
    print(f"Velocidad:        {velocidad:,.0f} filas/segundo")
    print(len(tabla_test))


    a = (tabla_test["id"]>100)

    inicio = time.perf_counter()

    filtrado = tabla_test.filter(a)

    fin = time.perf_counter()

    tiempo_total = fin - inicio

    print(f"\n=== RESULTADOS ===")
    print(f"Tiempo total:     {tiempo_total:.4f} segundos")


    print(tabla_test["id"].shape)
    chunk = tabla_test["id"].get_chunk(0)
    decompressed_chunk = blosc2.decompress(chunk)
    np_array_chunk = np.frombuffer(decompressed_chunk, dtype=np.int64)


