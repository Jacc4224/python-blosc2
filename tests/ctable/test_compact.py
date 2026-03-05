import pytest
import numpy as np
from blosc2 import CTable
from pydantic import BaseModel, Field
from typing import Annotated


# --- Setup básico de modelos para los tests ---
class NumpyDtype:
    def __init__(self, dtype):
        self.dtype = dtype


class RowModel(BaseModel):
    id: Annotated[int, NumpyDtype(np.int64)] = Field(ge=0)
    score: Annotated[float, NumpyDtype(np.float64)] = Field(ge=0, le=100)


def generate_test_data(n_rows: int) -> list:
    return [(i, float(i)) for i in range(n_rows)]


# -------------------------------------------------------------------
# TESTS ESPECÍFICOS PARA COMPACT()
# -------------------------------------------------------------------

def test_compact_empty_table():
    """Prueba compact() en una tabla completamente vacía (sin datos)."""
    table = CTable(RowModel, expected_size=100)
    initial_capacity = len(table._valid_rows)

    assert len(table) == 0

    # No debería lanzar ningún error
    table.compact()

    # La capacidad puede haberse reducido drásticamente, pero la tabla lógica debe seguir vacía
    assert len(table) == 0
    # Verificamos que si se añaden datos después, funciona correctamente
    table.append((1, 10.0))
    assert len(table) == 1
    assert table.id[0] == 1


def test_compact_full_table():
    """Prueba compact() en una tabla completamente llena (sin agujeros ni espacio libre)."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    assert len(table) == 50
    initial_capacity = len(table._valid_rows)

    # No debería lanzar ningún error ni cambiar el estado lógico
    table.compact()

    assert len(table) == 50
    # La capacidad no debería haber cambiado porque ya estaba llena
    assert len(table._valid_rows) == initial_capacity

    # Verificamos integridad de datos
    assert table.id[0] == 0
    assert table.id[-1] == 49


def test_compact_already_compacted_table():
    """Prueba compact() en una tabla que tiene espacio libre pero ningún agujero (datos contiguos)."""
    data = generate_test_data(20)
    # Expected_size grande para asegurar que hay espacio libre al final
    table = CTable(RowModel, new_data=data, expected_size=100)

    assert len(table) == 20

    # Ejecutamos compact. Como los datos ya están contiguos, la tabla podría reducir
    # su tamaño por el while de < len//2, pero no debería fallar.
    table.compact()

    assert len(table) == 20

    # Verificamos que los datos siguen en su sitio
    for i in range(20):
        assert table.id[i] == i

    # Validamos que todos los True están seguidos al principio
    mask = table._valid_rows[:len(table._valid_rows)]
    assert np.all(mask[:20] == True)
    if len(mask) > 20:
        assert np.all(mask[20:] == False)


def test_compact_with_holes():
    """Prueba compact() en una tabla con alta fragmentación (agujeros)."""
    data = generate_test_data(30)
    table = CTable(RowModel, new_data=data, expected_size=50)

    # Borramos de forma dispersa: dejamos solo [0, 5, 10, 15, 20, 25]
    to_delete = [i for i in range(30) if i % 5 != 0]
    table.delete(to_delete)

    assert len(table) == 6

    # Ejecutamos compact
    table.compact()

    assert len(table) == 6

    # Verificamos que los datos correctos sobrevivieron y se movieron al principio
    expected_ids = [0, 5, 10, 15, 20, 25]
    for i, exp_id in enumerate(expected_ids):
        # A través de la vista lógica (Column wrapper)
        assert table.id[i] == exp_id
        # A través del array físico de blosc2 (para asegurar que compact funcionó)
        assert table._cols["id"][i] == exp_id

    # Verificamos la máscara física: los primeros 6 deben ser True, el resto False
    mask = table._valid_rows[:len(table._valid_rows)]
    assert np.all(mask[:6] == True)
    if len(mask) > 6:
        assert np.all(mask[6:] == False)


def test_compact_all_deleted():
    """Prueba compact() en una tabla donde se han borrado absolutamente todas las filas."""
    data = generate_test_data(20)
    table = CTable(RowModel, new_data=data, expected_size=20)

    # Borramos todo
    table.delete(list(range(20)))
    assert len(table) == 0

    # Debería manejar arreglos vacíos correctamente
    table.compact()

    assert len(table) == 0

    # Comprobamos que podemos volver a escribir en ella
    table.append((99, 99.0))
    assert len(table) == 1
    assert table.id[0] == 99


def test_compact_multiple_times():
    """Llamar a compact() varias veces seguidas no debe corromper los datos ni crashear."""
    data = generate_test_data(10)
    table = CTable(RowModel, new_data=data, expected_size=20)

    table.delete([1, 3, 5, 7, 9])  # Quedan 5 elementos

    # Compactar 3 veces seguidas
    table.compact()
    table.compact()
    table.compact()

    assert len(table) == 5
    assert list(table.id) == [0, 2, 4, 6, 8]
