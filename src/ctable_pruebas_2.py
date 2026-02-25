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






if __name__ == "__main__":
    import numpy as np
    import time
    import blosc2.ctable
    import random

    # Generación masiva actualizada

    n_rows = 1000
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

    tabla = blosc2.CTable(RowModel)
    tabla2=blosc2.CTable(RowModel)
    start = time.perf_counter()
    tabla.extend(data_masiva)
    stop = time.perf_counter()
    print(f"Tiempo extend: {stop - start:.4f} s")
    print("------------------------------------------------------")


    tabla2=blosc2.CTable(RowModel)
    start = time.perf_counter()
    tabla2.append(data_masiva[0])
    stop = time.perf_counter()
    print(f"Tiempo append: {stop - start:.4f} s")
    print("------------------------------------------------------")

    start = time.perf_counter()
    tabla2.extend(tabla)
    stop = time.perf_counter()
    print(f"Tiempo extend CTable: {stop - start:.4f} s")
    print("------------------------------------------------------")


    numeros = random.sample(range(0, n_rows), n_rows//2)
    n = blosc2.count_nonzero(tabla._valid_rows)
    start = time.perf_counter()
    tabla.delete(numeros)
    stop = time.perf_counter()
    print(f"Tiempo delete: {stop - start:.4f} s")
    print(f"Longitud despues de delete: {blosc2.count_nonzero(tabla._valid_rows)}")
    print(f"Longitud mostrada: {tabla.nrows}")
    print(f"Valor esperado: {500000}")
    print("------------------------------------------------------")


    filtro = ((tabla['id'] <=1_000_000) & (tabla['score'] > 80)).compute()
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
    print(f"Ratio: {total_sin_comprimir/total_comprimido:.2}x")




    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print("COMPARACIÓN NUEVO HEAD Y TAIL")

    print("------------------------------------------------------")
    print("------------------------------------------------------")

    n = 1000
    start = time.perf_counter()
    h = tabla.tail(n)
    stop = time.perf_counter()
    print(f"Tiempo Tail: {stop - start:.4f} s")
    print(len(h))
    print(blosc2.count_nonzero(h._valid_rows))
    print("------------------------------------------------------")

    start = time.perf_counter()
    h = tabla.head(n)
    stop = time.perf_counter()
    print(f"Tiempo Head: {stop - start:.4f} s")
    print(len(h))
    print(blosc2.count_nonzero(h._valid_rows))
    print("------------------------------------------------------")
