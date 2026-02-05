import time
import numpy as np
import blosc2
from pydantic import BaseModel, Field
from typing import Annotated

# --- 1. Definir el Modelo ---
class RowModel(BaseModel):
    id: int = Field(ge=0)
    name: bytes = Field(default=b"unknown", max_length=10)
    score: float

# --- 2. Parámetros ---
N = 100_000
row_data = {"id": 1, "name": b"benchmark", "score": 3.14}

print(f"=== BENCHMARK: Ingestión Iterativa ({N} filas) ===\n")

# ==========================================
# TEST PANDAS (Baseline)
# ==========================================
import pandas as pd

print("--- 1. PANDAS (Lista -> DataFrame) ---")
t0 = time.time()

buffer_list = []
for _ in range(N):
    buffer_list.append(row_data)

df = pd.DataFrame(buffer_list)
t_pandas = time.time() - t0

print(f"Tiempo Total: {t_pandas:.4f} s")
mem_pandas = df.memory_usage(deep=True).sum() / (1024**2)
print(f"Memoria RAM:  {mem_pandas:.2f} MB")


# ==========================================
# TEST BLOSC2 (Estrategia 1: extend() con lista)
# ==========================================
print("\n--- 2. BLOSC2 (extend con lista de dicts) ---")
t0 = time.time()

# Acumular en lista de diccionarios
buffer_list_2 = []
for _ in range(N):
    buffer_list_2.append(row_data)

# Crear CTable vacía e insertar todo de golpe
ctable = blosc2.CTable(RowModel)
ctable.extend(buffer_list_2)

t_blosc_extend = time.time() - t0
print(f"Tiempo Total: {t_blosc_extend:.4f} s")

mem_blosc_extend = sum(col.schunk.nbytes for col in ctable._cols.values()) / (1024**2)
print(f"Memoria (Compr): {mem_blosc_extend:.2f} MB")


# ==========================================
# TEST BLOSC2 (Estrategia 2: Append iterativo + resize geométrico)
# ==========================================
# Esta es la estrategia OPTIMIZADA que discutimos (resize geométrico)
print("\n--- 3. BLOSC2 (append iterativo optimizado) ---")
t0 = time.time()

ctable2 = blosc2.CTable(RowModel)

# Simulamos el bucle iterativo "uno a uno"
# En tu código optimizado, esto sería tu método append() con resize geométrico
for _ in range(N):
    ctable2.append(row_data)

t_blosc_append = time.time() - t0
print(f"Tiempo Total: {t_blosc_append:.4f} s")

mem_blosc_append = sum(col.schunk.nbytes for col in ctable2._cols.values()) / (1024**2)
print(f"Memoria (Compr): {mem_blosc_append:.2f} MB")


# ==========================================
# CONCLUSIONES
# ==========================================
print("\n--- RESUMEN ---")
print(f"Pandas (lista->df):       {t_pandas:.4f} s")
print(f"Blosc2 (extend):          {t_blosc_extend:.4f} s ({t_pandas/t_blosc_extend:.2f}x {'más rápido' if t_blosc_extend < t_pandas else 'más lento'})")
print(f"Blosc2 (append iterativo):{t_blosc_append:.4f} s ({t_pandas/t_blosc_append:.2f}x {'más rápido' if t_blosc_append < t_pandas else 'más lento'})")

print(f"\nCompresión Blosc2 vs Pandas: {mem_blosc_extend / mem_pandas * 100:.2f}% del tamaño")
