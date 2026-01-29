#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
from __future__ import annotations
from collections.abc import Iterable

from dataclasses import Field
from os import wait
from typing import TYPE_CHECKING, Annotated, Any, Generic, TypeVar, Tuple

import numpy as np
from numpy.ma.core import resize
from pydantic import BaseModel, Field, create_model, ValidationError
from pydantic.v1.class_validators import Validator

import blosc2

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


class ColumnTable_B2(Generic[RowT]):

    def __init__(self, row_type: type[RowT], new_data: dict[str, Any] | Iterable[dict[str,Any]] | RowT = None, key: str = None) -> None:
        self._row_type = row_type
        self._cols: dict[str, blosc2.NDArray] = {}
        self._capacity: int = 1
        self._n_rows: int = 0
        self._col_widths: dict[str, int] = {}
        self.row = _RowIndexer(self)
        self._key: str = key


        for name, field in row_type.model_fields.items():
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
            if isinstance(new_data, Iterable) and not isinstance(new_data, dict):
                self.extend(new_data)
            else:
                self.append(new_data)

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

        retval = ColumnTable_B2(self._row_type, row_to_add)
        return retval

    def tail(self, tail: int = 1) -> None:
        if not isinstance(tail, int):
            raise TypeError("tail must be an integer")

        end = self._n_rows
        start = max(0, self._n_rows-tail)

        row_to_add = self.row[start:end]

        retval = ColumnTable_B2(self._row_type, row_to_add)
        return retval

    def __getitem__(self, s: str):
        return self._cols[s] if s in self._cols else None

    def __getattr__(self, s: str):
        return self[s] if s in self._cols else super().__getattribute__(s)

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

    def append(self, data: dict[str, Any] | RowT) -> None:
        try:
            row = data if isinstance(data, self._row_type) else self._row_type(**data)
        except (TypeError, ValidationError) as e:
            raise TypeError(
                f"Data provided does not match the expected row schema '{self._row_type.__name__}'.\n"
                f"Details: {e}"
            ) from e

        if self._n_rows > 0:
            if self._key is not None:
                key = data[self._key]
                for v in self._cols[self._key]:
                    if v == key:
                        raise KeyError(f"Key already exists in column {self._key} with value {data[self._key]}")

            self._capacity += 1
            for col_array in self._cols.values():
                col_array.resize((self._capacity,))

        for k, v in row.model_dump().items():
            self._cols[k][self._n_rows] = v

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

    def _appendExtend(self, data: dict[str, Any] | RowT) -> None:
        try:
            row = data if isinstance(data, self._row_type) else self._row_type(**data)
        except (TypeError, ValidationError) as e:
            raise TypeError(
                f"Data provided does not match the expected row schema '{self._row_type.__name__}'.\n"
                f"Details: {e}"
            ) from e

        if self._n_rows > 0:
            if self._key is not None:
                key = data[self._key]
                for v in self._cols[self._key]:
                    if v == key:
                        raise KeyError(f"Key already exists in column {self._key} with value {data[self._key]}")

        # if error the table should be resized, otherwise empty or default rows.
        # continue with extend and resize one row?
        # stop extend and rsize all empty rows?

        for k, v in row.model_dump().items():
            self._cols[k][self._n_rows] = v

        self._n_rows += 1

    def extend(self, rows: Iterable[dict[str, Any] | RowT] | ColumnTable_B2) -> None:
        if not (isinstance(rows, Iterable) or isinstance(rows, dict) or isinstance(rows, ColumnTable_B2)):
            raise TypeError("Expected an iterable of rows or ColumnTable_B2.")
        if isinstance(rows, ColumnTable_B2):
            if rows._row_type != self._row_type:
                raise TypeError("Tables are not the same type.")

            self._capacity = self._capacity + rows.nrows
            for col_array in self._cols.values():
                col_array.resize((self._capacity,))

            for k in rows._cols.keys():
                self._cols[k][self._n_rows:] = rows._cols[k][:]
            self._n_rows += rows.nrows

        else:
            if self._n_rows > 0:
                self._capacity += len(rows)
            else:
                self._capacity += len(rows)-1

            for col_array in self._cols.values():
                col_array.resize((self._capacity,))

            for r in rows:
                self._appendExtend(r)

    def filter(self, expr_result) -> ColumnTable_B2:
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


        retval = ColumnTable_B2(self._row_type)
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
    def load(cls, urlpath: str, group: str = "table", row_type: type[RowT] | None = None) -> ColumnTable_B2:
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

    #We create our Blosc2 CTable
    #Capacity atribute set al 512 by default


    data_1 = {"id": 0, "name": "Alice", "score": 91.5}
    data_2 = [
            {"id": 1, "name": "bob", "score": 88.0, "active": False},
            {"id": 2, "score": 88.0, "active": False},  # missing field
            {"id": 3, "name": "carol", "score": 73.25},
            {"id": 4, "name": "Alex", "score": 3.0, "active": True},

            ]
    data_3 = {"id": 4, "name": "Jorge", "score": 42.0}

    data_4 = [{"id": 0, "name": "Alice", "score": 91.5},
                 {"id": 1, "name": "Bob", "score": 88.0, "active": False},
                 {"id": 2, "score": 88.0, "active": False},
                 {"id": 3, "name": "Carol", "score": 73.25},
                 {"id": 4, "name": "Alex", "score": 3.0, "active": True},
                 {"id": 5, "name": "Jorge", "score": 42.0},
                 {"id": 6, "name": "Diana", "score": 95.5, "active": True},
                 {"id": 7, "name": "Elena", "score": 67.25},
                 {"id": 8, "score": 82.0, "active": True},
                 {"id": 9, "name": "Franco", "score": 76.5, "active": False},
                 {"id": 10, "name": "Gabriela", "score": 89.0},
                 {"id": 11, "name": "Héctor", "score": 55.75, "active": True},
                 {"id": 12, "score": 91.25, "active": False},
                 {"id": 13, "name": "Isabel", "score": 84.0},
                 {"id": 14, "name": "Javier", "score": 72.5, "active": True},
                 {"id": 15, "name": "Karen", "score": 93.0},
                 {"id": 16, "name": "Luis", "score": 68.75, "active": False},
                 {"id": 17, "score": 87.5, "active": True},
                 {"id": 18, "name": "Marta", "score": 79.25},
                 {"id": 19, "name": "Nicolás", "score": 85.5, "active": False},
                 {"id": 20, "name": "Olivia", "score": 92.0},
                 {"id": 21, "name": "Pablo", "score": 61.0, "active": True},
                 {"id": 22, "score": 80.75, "active": False},
                 {"id": 23, "name": "Quentin", "score": 75.25},
                 {"id": 24, "name": "Rosa", "score": 88.5, "active": True},
                 {"id": 25, "name": "Santiago", "score": 69.0},
                 {"id": 26, "name": "Teresa", "score": 94.75, "active": False},
                 {"id": 27, "score": 83.0, "active": True},
                 {"id": 28, "name": "Ulises", "score": 77.5},
                 {"id": 29, "name": "Valeria", "score": 86.25, "active": False},
                 {"id": 30, "name": "Wanda", "score": 90.0},
                 {"id": 31, "name": "Xavier", "score": 64.75, "active": True},
                 {"id": 32, "score": 81.5, "active": False},
                 {"id": 33, "name": "Yolanda", "score": 74.0},
                 {"id": 34, "name": "Zara", "score": 87.25, "active": True},
                 {"id": 35, "name": "Andrés", "score": 70.5},
                 {"id": 36, "name": "Beatriz", "score": 93.5, "active": False},
                 {"id": 37, "score": 85.75, "active": True},
                 {"id": 38, "name": "Carlos", "score": 78.0},
                 {"id": 39, "name": "Daniela", "score": 89.5, "active": False},
                 {"id": 40, "name": "Enrique", "score": 66.25},
                 {"id": 41, "name": "Fernanda", "score": 92.5, "active": True},
                 {"id": 42, "score": 84.0, "active": False},
                 {"id": 43, "name": "Gustavo", "score": 71.75},
                 {"id": 44, "name": "Herminia", "score": 88.25, "active": True},
                 {"id": 45, "name": "Ignacio", "score": 63.5},
                 {"id": 46, "name": "Justina", "score": 95.0, "active": False},
                 {"id": 47, "score": 86.5, "active": True},
                 {"id": 48, "name": "Kevin", "score": 76.75},
                 {"id": 49, "name": "Laura", "score": 90.5, "active": False}
                ]

    data_5 = {"id": 4, "name": "Jorge"}


    # Lets try indexing  new elements with key="id"
    tableb2 = ColumnTable_B2(RowModel, data_1, key="id")
    tableb2.extend(data_2)
    # tableb2.append(data_3) # key="id" in data_3 is the same as in data_i: KeyError



    tableb2 = ColumnTable_B2(RowModel, data_4)



    # Append error example
    # tableb2.append({"a": 6})

    # Lets see the full table
    print(f"Tabla:\n{tableb2} \n\n")


    ##############################################################################
    # Save (using treestore)
    tableb2.save(urlpath="people.b2z")

    # Load back
    loaded = ColumnTable_B2.load(urlpath="people.b2z")

    """
        The Columns are shuffled, not the same order as before save
    """
    ##############################################################################

    # We make a filter expresion
    exp = ((tableb2["score"] > 50) & (tableb2["active"] == True))
    exp_no_bool = (tableb2.active + 1)
    verdad = exp.compute()


    # Filter from lazy expresion and from bool NDArray, both with the same outcome
    tableb2.filter(exp)
    prnt = tableb2.filter(verdad)
    print(prnt)


    # Smaller filter size error examlpe
    # arr = blosc2.asarray(np.array([True, False, True, False]))
    # tableb2.filter(arr)

    # Not same dtype error example
    #tableb2.filter(exp_no_bool)

    # Not NDArray error example
    #tableb2.filter([True, False, True, False])

    print('\n\n')


    # The following expresions are equivalent

    tableb2.row[:3]    # and other slice combinations such as 0:3, :3:1
    filas = tableb2.row["1:3"]


    print(tableb2["name"].dtype)


    # Head and tail return a new table with the first and last n rows respectivley
    tableb2.tail(5)
    tableb2.head(5)




    table_extension_fail = ColumnTable_B2(RowModel2, data_5, key="id")

    # tableb2.extend(table_extension_fail)
    tableb2.extend(tableb2.head(3))

    tableb2.delete([1, 2, 3, 5, 7, 8])
    tableb2.delete(0)

    print(tableb2)



    """ Esto no funciona """


    '''target_name = blosc2.full(shape=(tableb2.nrows,), fill_value="unknown", dtype="<U32")

    lzExp = (tableb2["name"] == "unknown")
    cmptExp = lzExp.compute()'''