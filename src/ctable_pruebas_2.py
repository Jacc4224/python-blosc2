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
from pydantic import BaseModel, Field, create_model, ValidationError

import blosc2
from blosc2 import concat

""" Imports extra """


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
    id: Annotated[int, NumpyDtype(np.int16)] = Field(ge=0)
    name: Annotated[str, MaxLen(10)] = Field(default="unknown")
    # name: Annotated[bytes, MaxLen(10)] = Field(default=b"unknown")
    score: Annotated[float, NumpyDtype(np.float32)] = Field(ge=0, le=100)
    active: Annotated[bool, NumpyDtype(np.bool_)] = True

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

    def head(self, head: int = 1) -> None:
        if not isinstance(head, int):
            raise TypeError("tail must be an integer")

        start = 0
        end = min(self._n_rows, head)

        row_to_add = self.row[start:end]

        retval = CTable(self._row_type, row_to_add)
        return retval

    def tail(self, tail: int = 1) -> None:
        if not isinstance(tail, int):
            raise TypeError("tail must be an integer")

        end = self._n_rows
        start = max(0, self._n_rows-tail)

        row_to_add = self.row[start:end]

        retval = CTable(self._row_type, row_to_add)
        return retval

    def __getitem__(self, s: str):
        return self._cols[s] if s in self._cols else None

    def __getattr__(self, s: str):
        return self[s] if s in self._cols else super().__getattribute__(s)


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
        if isinstance(pos, list):
            if not isinstance(pos[0], int):
                raise TypeError("Position must be an integer or a list of integers")
            desp = 1
            for i in range(min(pos), self._n_rows-len(pos)):
                for v in self._cols.values():
                    while(i+desp in pos):
                        desp += 1
                    v[i] = v[i + desp]
            self._capacity = self._capacity - len(pos)
            for col_array in self._cols.values():
                col_array.resize((self._capacity,))
            self._n_rows = self._n_rows - len(pos)
        elif isinstance(pos, int):
            if pos > self._n_rows:
                raise IndexError("Index out of range")
            elif pos < 0:
                pos = self._n_rows + pos

            for i in range(pos, self._n_rows-1):
                for v in self._cols.values():
                    v[i] = v[i+1]
            self._capacity = self._capacity -1
            for col_array in self._cols.values():
                col_array.resize((self._capacity,))
            self._n_rows = self._n_rows-1

        else:
            raise TypeError("Position must be an integer or a list of integers")

    def extend(self, data: list | CTable | Any) -> None:
        if isinstance(data, dict):
            raise TypeError("Dictionaries are not supported in extend.")

        current_col_names = getattr(self, "_col_names", list(self._cols.keys()))
        current_col_values = [self._cols[name] for name in current_col_names]

        new_b2_cols = []
        new_nrows = 0

        if hasattr(data, "_cols") and hasattr(data, "nrows"):
            if len(data._cols) != len(current_col_names):
                raise ValueError(f"Input CTable has {len(data._cols)} columns, expected {len(current_col_names)}")

            for name in current_col_names:
                if name not in data._cols:
                    raise ValueError(f"Input CTable missing column '{name}'")
                new_b2_cols.append(data._cols[name])

            new_nrows = data.nrows

        elif isinstance(data, (list, tuple)):
            if len(data) != len(current_col_names):
                raise ValueError(f"Expected {len(current_col_names)} columns, received {len(data)}")

            for i, col_data in enumerate(data):
                target_dtype = current_col_values[i].dtype
                try:
                    b2_arr = blosc2.asarray(col_data, dtype=target_dtype)
                except Exception as e:
                    raise TypeError(f"Column {i} ('{current_col_names[i]}') conversion error: {e}")

                if i == 0:
                    new_nrows = b2_arr.shape[0]
                elif b2_arr.shape[0] != new_nrows:
                    raise ValueError("All new columns must have the same length.")

                new_b2_cols.append(b2_arr)
        else:
            raise TypeError("Data format not supported in extend (expected list of columns or CTable).")

        if new_nrows == 0:
            return

        if self._key is not None:
            key_idx = current_col_names.index(self._key)
            new_key_col = new_b2_cols[key_idx]
            new_keys_np = new_key_col[:]

            if len(set(new_keys_np)) != len(new_keys_np):
                raise KeyError(f"Input data contains duplicate keys in column '{self._key}'")

            if not self._key_set.isdisjoint(new_keys_np):
                raise KeyError(f"Input data contains keys that already exist in the table.")

        old_nrows = self._n_rows
        self._n_rows += new_nrows
        self._capacity = self._n_rows

        for i, name in enumerate(current_col_names):
            target_array = self._cols[name]
            source_array = new_b2_cols[i]

            target_array.resize((self._capacity,))
            target_array[old_nrows:self._n_rows] = source_array[:]

        if self._key is not None:
            self._key_set.update(new_keys_np)

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


        retval = CTable(self._row_type)
        if filtro is not None and len(filtro) >= self._n_rows:
            for i in range(self._n_rows):
                if filtro[i]:
                    retval.append(self.row[i])
        else:
            raise ValueError(
                f"Filter length ({len(filtro)}) does not match the number of rows ({self._n_rows})."
            )
        return retval

    def _run_row_logic(self, ind: int | slice | str) -> RowT | list[RowT]| None:
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
                index: int = ind
            elif -self._n_rows <= ind < 0:
                index: int = self._n_rows + ind
            else:
                raise IndexError("list index out of range")

            data = {}
            for name, col in self._cols.items():
                arr_val = col[index]
                data[name] = arr_val[()]
            '''if hasattr(arr_val, 'item'):
                    data[name] = arr_val.item()
                else:
                    data[name] = arr_val'''
            return self._row_type(**data)


        if isinstance(ind, slice):
            indices = range(*ind.indices(self._n_rows))
            return [self._run_row_logic(i) for i in indices]

        raise  TypeError(
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
    ...