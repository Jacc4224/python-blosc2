#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
from __future__ import annotations

from dataclasses import Field
from os import wait
from typing import TYPE_CHECKING, Annotated, Any, Generic, TypeVar

import numpy as np
from pydantic import BaseModel, Field, create_model

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


class ColumnTable(Generic[RowT]):

    #_row_type -->
    #_cols --> diccionario de listas.


    def __init__(self, row_type: type[RowT]):
        self._row_type = row_type
        self._cols: dict[str, list[Any]] = {f: [] for f in row_type.model_fields}

    def __str__(self):
        retval = []
        #arrays = self.to_numpy()
        cont = 0
        for name in self._cols.keys():
            retval.append(f"{name:<{15}} |")
            cont += 1
        retval.append("\n")
        for i in range(cont):
            retval.append("-----------------")
        retval.append("\n")

        for i in range(len(next(iter(self._cols.values())))):
            for name in self._cols.keys():
                retval.append(f"{self._cols[name][i]:<{15}}")
                retval.append(f" |")
            retval.append("\n")
            for i in range(cont):
                retval.append("-----------------")
            retval.append("\n")
        return "".join(retval)


    def __len__(self) -> int:
        return self.nrows

    def __getitem__(self, s: str):
        arrays = self.to_numpy()
        return arrays[s] if s in arrays.keys() else None

    def __getattr__(self, s: str):
        arrays = self.to_numpy()
        return arrays[s] if s in arrays.keys() else None


    def append(self, data: dict[str, Any] | RowT) -> None:

        row = data if isinstance(data, self._row_type) else self._row_type(**data)
        for k, v in row.model_dump().items():
            self._cols[k].append(v)

    def extend(self, rows: Iterable[dict[str, Any] | RowT]) -> None:
        for r in rows:      #Falta comprobar que sean válidos o delegar a append?
            self.append(r)


    def filter(self, ls: list[bool]) -> ColumnTable:
        retval = ColumnTable(self._row_type)
        for i in range(len(ls)):
            if ls[i]:
                retval.append(self.row(i))
        return retval




    def row(self, index: int) -> RowT | None:
        num_rows = len(next(iter(self._cols.values()))) if self._cols else 0
        if not (0 <= index < num_rows):
            return None

        data = {
            name: self._cols[name][index]
            for name in self._cols.keys()
        }

        return self._row_type(**data)

    @property
    def nrows(self) -> int:
        return len(next(iter(self._cols.values()))) if self._cols else 0

    def to_numpy(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, values in self._cols.items():
            field_info = self._row_type.model_fields[name]
            numpy_dtype = None
            max_length = None
            for md in getattr(field_info, "metadata", ()):
                if isinstance(md, NumpyDtype):
                    numpy_dtype = md.dtype
                if isinstance(md, MaxLen):
                    max_length = md.length

            base_ann = field_info.annotation
            if base_ann in (str, bytes):
                if max_length is None:
                    raise ValueError(f"Missing MaxLen for column '{name}'")
                dtype = f"U{max_length}" if base_ann is str else f"S{max_length}"
                out[name] = np.array(values, dtype=dtype)
                continue

            out[name] = np.array(values, dtype=numpy_dtype)
        return out

    def save(self, urlpath: str, group: str = "table") -> None:
        """
        Persist columns into a single TreeStore container.
        Each column is stored under a group / colname.
        """
        # mode='w' creates/overwrites; use 'a' to append/replace columns.
        with blosc2.TreeStore(urlpath, mode="w") as ts:
            arrays = self.to_numpy()
            for name, arr in arrays.items():
                node_path = f"{group}/{name}"
                # Store as compressed NDArray inside the tree
                print(f"Storing {name} with shape {arr.shape} and dtype {arr.dtype} in {node_path}")
                ts[node_path] = arr

    @classmethod
    def load(cls, urlpath: str, group: str = "table", row_type: type[RowT] | None = None) -> ColumnTable:  # noqa: C901
        """
        If `row_type` is provided, behave as before. If `row_type` is None,
        infer a BaseModel from the stored arrays' dtypes and construct a model
        using `create_model`.
        """
        with blosc2.TreeStore(urlpath, mode="r") as ts:
            # discover stored column node paths under the group
            keys = list(ts.keys()) if hasattr(ts, "keys") else []
            prefix = f"/{group}/"
            field_names = sorted(k[len(prefix) :] for k in keys if k.startswith(prefix))

            # read arrays and infer annotations/metadata
            arrays: dict[str, np.ndarray] = {}
            annotations: dict[str, Any] = {}
            defaults: dict[str, Any] = {}

            for field in field_names:
                node_path = f"{group}/{field}"
                nda = ts[node_path]
                arr = np.asarray(nda)
                arrays[field] = arr

                dt = arr.dtype
                kind = dt.kind

                if kind == "U":
                    # numpy 'U' itemsize is bytes; deduce char length using U1 itemsize
                    char_size = np.dtype("U1").itemsize
                    max_len = dt.itemsize // char_size
                    annotations[field] = Annotated[str, MaxLen(int(max_len))]
                    defaults[field] = Field(default="")
                elif kind == "S":
                    max_len = dt.itemsize
                    annotations[field] = Annotated[bytes, MaxLen(int(max_len))]
                    defaults[field] = Field(default=b"")
                elif kind in ("i", "u"):  # signed/unsigned ints
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
                    # fallback: keep dtype metadata and Any annotation
                    annotations[field] = Annotated[Any, NumpyDtype(dt)]
                    defaults[field] = Field(default=None)

            # if a row_type was supplied, prefer it; otherwise build an inferred model
            if row_type is None:
                model_fields = {}
                for name, ann in annotations.items():
                    model_fields[name] = (ann, defaults[name])
                row_type = create_model("InferredRowModel", __base__=BaseModel, **model_fields)  # type: ignore

            # instantiate table and populate columns
            tbl = cls(row_type)
            for name, arr in arrays.items():
                # convert bytes fields to python bytes if needed; keep lists
                tbl._cols[name] = arr.tolist()
        return tbl








class ColumnTable_B2(Generic[RowT]):

    def __init__(self, row_type: type[RowT], capacity: int = 512):
        self._row_type = row_type
        self._cols: dict[str, blosc2.NDArray] = {}
        self._capacity: int = capacity
        self._n_rows: int = 0
        self._col_widths: dict[str, int] = {}


        for name, field in row_type.model_fields.items():
            origin = getattr(field.annotation, "__origin__", field.annotation)

            if origin == str or field.annotation == str:
                max_len = 32  # Default si no hay MaxLen
                if hasattr(field.annotation, "__metadata__"):
                    for meta in field.annotation.__metadata__:
                        if isinstance(meta, MaxLen):
                            max_len = meta.max_length
                            break
                dt = np.dtype(f"U{max_len}")
                display_width = max(10, min(max_len, 50))

            elif origin == bytes or field.annotation == bytes:
                max_len = 32
                if hasattr(field.annotation, "__metadata__"):
                    for meta in field.annotation.__metadata__:
                        if isinstance(meta, MaxLen):
                            max_len = meta.max_length
                            break
                dt = np.dtype(f"S{max_len}")
                display_width = max(10, min(max_len, 50))

            elif origin == int or field.annotation == int:
                dt = np.int64
                display_width = 12  # Suficiente para enteros estándar

            elif origin == float or field.annotation == float:
                dt = np.float64
                display_width = 15  # Espacio para decimales y notación científica

            elif origin == bool or field.annotation == bool:
                dt = np.bool_
                display_width = 6  # "True" / "False" caben en 5-6

            elif origin == complex or field.annotation == complex:
                dt = np.complex128
                display_width = 25  # (1.23+4.56j) suele ser largo

            else:
                dt = np.object_
                display_width = 20

            final_width = max(len(name), display_width)
            self._col_widths[name] = final_width

            self._cols[name] = blosc2.zeros(shape=capacity, dtype=dt)

    def __str__(self):
        # Versión simplificada del print
        retval = []
        # arrays = self.to_numpy()
        cont = 0
        for name in self._cols.keys():
            retval.append(f"{name:^{self._col_widths[name]}} |")
            cont += self._col_widths[name]+2
        retval.append("\n")
        for i in range(cont):
            retval.append("-")
        retval.append("\n")

        for i in range(self._n_rows):
            for name in self._cols.keys():
                retval.append(f"{self._cols[name][i]:^{self._col_widths[name]}}")
                retval.append(f" |")
            retval.append("\n")
            for i in range(cont):
                retval.append("-")
            retval.append("\n")
        return "".join(retval)

    def __getitem__(self, s: str):
        return self._cols[s] if s in self._cols else None

    def __getattr__(self, s: str):
        return self[s] if s in self._cols else super().__getattribute__(s)

    @property
    def nrows(self) -> int:
        return self._n_rows

    def append(self, data: dict[str, Any] | RowT) -> None:
        #Falta comprobar que el tipo de datos de RowT coincide con el del diccionario

        if self._n_rows >= self._capacity:
            self._capacity *= 2
            for col_array in self._cols.values():
                col_array.resize((self._capacity,))

        row = data if isinstance(data, self._row_type) else self._row_type(**data)

        for k, v in row.model_dump().items():
            self._cols[k][self._n_rows] = v

        self._n_rows += 1

    def extend(self, rows: Iterable[dict[str, Any] | RowT]) -> None:
        for r in rows:
            self.append(r)

    def row(self, index: int) -> RowT | None:
        num_rows = self._n_rows
        if not (0 <= index < num_rows):
            return None

        data = {
            name: self._cols[name][index]
            for name in self._cols.keys()
        }

        return self._row_type(**data)



    def filter(self, expr_result) -> ColumnTable_B2:
        filtro = None
        if isinstance(expr_result, blosc2.LazyExpr):
            filtro = expr_result.compute()
        elif isinstance(expr_result, blosc2.NDArray):
            filtro = expr_result
        else:
            raise TypeError(f"El tipo {type(expr_result)} no es válido. Se esperaba blosc2.LazyExpr.")

        retval = ColumnTable_B2(self._row_type, self._capacity)
        if filtro is not None and len(filtro) >= self._n_rows:
            for i in range(self._n_rows):
                if filtro[i]:
                    retval.append(self.row(i))
        return retval




    def row(self, index: int) -> RowT | None:
        if not (0 <= index < self.nrows):
            return None

        data = {}
        for name, schunk in self._cols.items():
            arr_val = schunk[index]  # Esto descomprime el chunk afectado
            data[name] = arr_val[0] if isinstance(arr_val, (np.ndarray, blosc2.NDArray)) else arr_val

        return self._row_type(**data)


    def save(self, urlpath: str, group: str = "table") -> None:
        """
        Persist columns into a single TreeStore container.
        Each column is stored under a group / colname.
        """
        # mode='w' creates/overwrites; use 'a' to append/replace columns.
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

            tbl = cls(row_type, capacity=max(1, loaded_nrows))

            tbl._cols = b2_arrays
            tbl._n_rows = loaded_nrows
            tbl._capacity = loaded_nrows
            return tbl


if __name__ == "__main__":
    '''
    table = ColumnTable(RowModel)
    table.append({"id": 0, "name": "Alice", "score": 91.5})
    table.extend(
        [
            # {"id": 1, "name": "bob", "score": 88.0, "active": False},
            {"id": 1, "score": 88.0, "active": False},  # missing field
            {"id": 2, "name": "carol", "score": 73.25},
        ]
    )
    print("Rows:", table.nrows)
    ###########################################################################
    ###########################################################################
    ###########################################################################


    
    print("\n############ Segunda prueba #############\n")

    print(f"names con getitem: {table["name"]}")
    print(f"names con getattr: {table.name}")
    print(f"Tabla: \n{table}")


    bol_vec = (table.id == 1) | (table.name == "Alice")
    print(f"names: {bol_vec}")
    print(f"fila 2: {table.row(2)}")
    print(f"len(tabla): {len(table)}")
    print(f"Filtro: \n{table.filter(bol_vec)}")
    '''
    #Creamos nuestra tabla de prueba
    tableb2 = ColumnTable_B2(RowModel)

    #Probamos a introducir datos de diferentes forman
    tableb2.append({"id": 0, "name": "Alice", "score": 91.5})

    tableb2.extend(
        [
            {"id": 1, "name": "bob", "score": 88.0, "active": False},
            {"id": 3, "score": 88.0, "active": False},  # missing field
            {"id": 2, "name": "carol", "score": 73.25},
            {"id": 4, "name": "Alex", "score": 3.0, "active": True},

        ]
    )

    #preparamos una comparación
    exp = ((tableb2["id"] == 1))
    type(exp)
    verdad = exp.compute()

    print(f"Tabla:\n{tableb2}")
    print(f"names: {verdad[:tableb2.nrows]}")
    print("\n\n")


    tableb2.save(urlpath="people.b2z")

    # Load back
    loaded = ColumnTable_B2.load(urlpath="people.b2z")
    print("\n\n", loaded)

    print(type([True, False, True, False]))
    tableb2.filter(exp)
    tableb2.filter(verdad)
    #tableb2.filter([True, False, True, False]) #ejemplo de excepción

    #print(tableb2.filter(verdad))


