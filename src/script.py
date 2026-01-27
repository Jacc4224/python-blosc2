#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
from __future__ import annotations

from dataclasses import Field
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

    def __init__(self, row_type: type[RowT]):
        self._row_type = row_type
        self._cols: dict[str, blosc2.NDArray] = {}
        shape = (512,512)

        for name, field in row_type.model_fields.items():
            origin = getattr(field.annotation, "__origin__", field.annotation)

            if origin == str or field.annotation == str:
                max_len = 32
                if hasattr(field.annotation, "__metadata__"):
                    for meta in field.annotation.__metadata__:
                        if isinstance(meta, MaxLen):
                            max_len = meta.max_length
                            break
                dt = np.dtype(f"U{max_len}")

            elif origin == bytes or field.annotation == bytes:
                max_len = 32
                if hasattr(field.annotation, "__metadata__"):
                    for meta in field.annotation.__metadata__:
                        if isinstance(meta, MaxLen):
                            max_len = meta.max_length
                            break
                dt = np.dtype(f"S{max_len}")

            elif origin == int or field.annotation == int:
                dt = np.int64

            elif origin == float or field.annotation == float:
                dt = np.float64

            elif origin == bool or field.annotation == bool:
                dt = np.bool_

            elif origin == complex or field.annotation == complex:
                dt = np.complex128

            else:
                dt = np.object_

            self._cols[name] = blosc2.zeros(shape=shape, dtype=dt)

    def append(self, data: dict[str, Any] | RowT) -> None:
        ...

    def extend(self, rows: Iterable[dict[str, Any] | RowT]) -> None:
       ...

    def __getitem__(self, s: str):
        ...

    def __getattr__(self, s: str):
        return self[s] if s in self._cols else super().__getattribute__(s)

    def filter(self, expr_result) -> ColumnTable_B2:
        """Filtra usando el resultado de una expresión Blosc2"""
        # Materializamos la máscara a numpy bool para iterar índices
        # (Esto es rápido porque es solo 1 bit por fila)
        if hasattr(expr_result, "to_numpy"):
            mask = expr_result.to_numpy()
        else:
            mask = np.array(expr_result, dtype=bool)

        indices = np.where(mask)[0]

        retval = ColumnTable_B2(self._row_type)
        # Copiamos fila a fila (se podría optimizar copiando chunks enteros)
        for i in indices:
            retval.append(self.get_row(int(i)))  # int(i) para evitar tipos numpy
        return retval

    def get_row(self, index: int) -> RowT | None:
        if not (0 <= index < self.nrows):
            return None

        data = {}
        for name, schunk in self._cols.items():
            # Descomprimir SOLO ese dato
            # schunk[index] devuelve un array 0-D, sacamos el escalar con .item()
            arr_val = schunk[index]  # Esto descomprime el chunk afectado
            data[name] = arr_val[0] if isinstance(arr_val, (np.ndarray, blosc2.NDArray)) else arr_val

        return self._row_type(**data)

    @property
    def nrows(self) -> int:
        if not self._cols: return 0
        return next(iter(self._cols.values())).nitems

    def save(self, urlpath: str, group: str = "table") -> None:
        """Guarda los SChunks en disco via TreeStore"""
        with blosc2.TreeStore(urlpath, mode="w") as ts:
            for name, schunk in self._cols.items():
                node_path = f"{group}/{name}"
                print(f"Storing {name} ({schunk.nitems} items) in {node_path}")
                # SChunk se puede convertir a NDArray persistente en disco
                ts[node_path] = schunk.to_ndarray()

    @classmethod
    def load(cls, urlpath: str, group: str = "table", row_type: type[RowT] | None = None) -> ColumnTable:
        """Carga datos desde disco directamente a memoria comprimida (SChunk)"""
        with blosc2.TreeStore(urlpath, mode="r") as ts:
            # 1. Descubrir columnas
            prefix = f"/{group}/"
            # Truco para listar claves en TreeStore actual
            keys = [k for k in ts.root.iter_children() if k.startswith(group)]
            # O usar nombres conocidos si row_type existe.
            # Simplificación: asumimos que las claves son los nombres de archivo

            # (Aquí iría la lógica de inferencia de modelo si row_type es None,
            #  copiada de tu código original. Por brevedad, asumo row_type dado
            #  o infiero básico).

            if row_type is None:
                # ... Inferencia (ver código original) ...
                pass

            tbl = cls(row_type)

            # 2. Cargar datos
            for name in row_type.model_fields.keys():
                node_path = f"{group}/{name}"
                if node_path not in ts: continue

                # Leemos el NDArray del disco
                stored_arr = ts[node_path]

                # CONVERSIÓN CRÍTICA: Disk NDArray -> RAM SChunk
                # Creamos un SChunk en memoria con los mismos datos
                # Esto carga y re-comprime en RAM (o copia si es compatible)
                schunk = tbl._cols[name]

                # Opción eficiente: copiar buffers.
                # Opción fácil: leer todo a numpy y meterlo en SChunk
                # Para datasets enormes, habría que hacerlo por bloques.
                full_data = stored_arr[:]  # Descomprime todo a RAM temporalmente
                schunk.append_data(full_data)  # Re-inserta en SChunk

        return tbl

    def __str__(self):
        # Versión simplificada del print
        if self.nrows == 0: return "Empty Table"
        col_names = list(self._cols.keys())
        header = " | ".join(f"{c:^10}" for c in col_names)
        res = [header, "-" * len(header)]

        # Imprimir primeras 10 filas
        limit = min(10, self.nrows)
        for i in range(limit):
            row = [str(self._cols[c][i][0]) for c in col_names]  # [0] por dimensión extra
            res.append(" | ".join(f"{r:^10}" for r in row))

        return "\n".join(res)








if __name__ == "__main__":
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

    # Persist
    table.save(urlpath="people.b2z")

    # Load back
    loaded = ColumnTable.load(urlpath="people.b2z")
    print("Loaded rows:", loaded.nrows)
    print("Loaded name column:", loaded.to_numpy()["name"])
    print("Loaded score column:", loaded.to_numpy()["score"])
    print("Loaded score column:", loaded.to_numpy()["active"])



    print("Segunda prueba")
    #print(table)

    print(f"names: {table["name"]}")
    print(f"names: {table.juanjo}")


    bol_vec = (table.id == 1) | (table.name == "Alice")
    print(f"names: {bol_vec}")

    print(f"fila 2: {table.row(2)}")

    print(f"len(tabla): {len(table)}")

    print(f"Filtro: \n{table.filter(bol_vec)}")







