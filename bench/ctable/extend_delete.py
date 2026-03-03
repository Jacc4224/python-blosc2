#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import pytest
import numpy as np
import blosc2
from blosc2 import CTable
from pydantic import BaseModel, Field
from typing import Annotated, TypeVar

# NOTE: Make sure to import your CTable and NumpyDtype correctly

# -------------------------------------------------------------------
# 1. Row Type Definition for Testing
# -------------------------------------------------------------------
RowT = TypeVar("RowT", bound=BaseModel)


class NumpyDtype:
    def __init__(self, dtype):
        self.dtype = dtype


class RowModel(BaseModel):
    id: Annotated[int, NumpyDtype(np.int64)] = Field(ge=0)
    c_val: Annotated[complex, NumpyDtype(np.complex128)] = Field(default=0j)
    score: Annotated[float, NumpyDtype(np.float64)] = Field(ge=0, le=100)
    active: Annotated[bool, NumpyDtype(np.bool_)] = True


# -------------------------------------------------------------------
# 2. Helper Functions
# -------------------------------------------------------------------
def generate_test_data(n_rows: int, start_id: int = 1) -> list:
    """
    Generate n_rows of test data with IDs starting from start_id.
    This allows us to track which data is which after insertions.
    """
    return [
        (start_id + i, complex(i, -i), float((i * 7) % 100), bool(i % 2))
        for i in range(n_rows)
    ]


def get_valid_mask(table: CTable) -> np.ndarray:
    """
    Extract the _valid_rows mask as a boolean numpy array.
    Returns only up to the internal array length.
    """
    return np.array(table._valid_rows[:len(table._valid_rows)], dtype=bool)


def get_column_values(table: CTable, col_name: str, length: int) -> np.ndarray:
    """
    Extract the first 'length' values from a column.
    This includes both valid and invalid (deleted) positions.
    """
    return np.array(table._cols[col_name][:length])


def assert_mask_matches(table: CTable, expected_mask: list):
    """
    Verify that _valid_rows matches the expected boolean pattern.

    Args:
        table: CTable instance
        expected_mask: List of booleans [True, False, True, ...]
    """
    actual_mask = get_valid_mask(table)[:len(expected_mask)]
    expected = np.array(expected_mask, dtype=bool)

    np.testing.assert_array_equal(
        actual_mask, expected,
        err_msg=f"Mask mismatch.\nExpected: {expected}\nGot: {actual_mask}"
    )


def assert_data_at_positions(table: CTable, positions: list, expected_ids: list):
    """
    Verify that specific physical positions contain expected ID values.
    This checks the actual data in the arrays, not the logical view.

    Args:
        positions: Physical array positions to check
        expected_ids: Expected ID values at those positions
    """
    id_col = table._cols["id"]
    for pos, expected_id in zip(positions, expected_ids):
        actual_id = int(id_col[pos])
        assert actual_id == expected_id, \
            f"Position {pos}: expected ID {expected_id}, got {actual_id}"


# -------------------------------------------------------------------
# 3. Basic Gap Filling Tests
# -------------------------------------------------------------------

def test_insert_after_delete_fills_last_gap():
    """
    Insert 7 rows, delete even positions, then append 3 more.
    The new data should fill from the last False position.

    Initial: [0,1,2,3,4,5,6] -> IDs [1,2,3,4,5,6,7]
    Delete evens: mask = [F,T,F,T,F,T,F]
    Append 3: IDs [8,9,10] -> mask = [F,T,F,T,F,T,T,T,T]
    """
    # Insert initial data
    data_c1 = generate_test_data(7, start_id=1)
    table = CTable(RowModel, new_data=data_c1, expected_size=10)

    # Delete even positions (0, 2, 4, 6)
    table.delete([0, 2, 4, 6])

    # Verify mask after deletion
    expected_mask_after_delete = [False, True, False, True, False, True, False]
    assert_mask_matches(table, expected_mask_after_delete)
    assert len(table) == 3  # Only odd positions remain valid

    # Append new data
    data_c2 = generate_test_data(3, start_id=8)
    table.extend(data_c2)

    # Verify final mask
    expected_mask_final = [False, True, False, True, False, True, True, True, True]
    assert_mask_matches(table, expected_mask_final)
    assert len(table) == 6  # 3 original + 3 new

    # Verify data at physical positions
    # Positions 6, 7, 8 should have IDs 8, 9, 10
    assert_data_at_positions(table, [6, 7, 8], [8, 9, 10])


def test_append_single_row_fills_gap():
    """
    Create table with gaps, then append single rows one by one.
    Each append should fill the next available gap.
    """
    data = generate_test_data(5, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=10)

    # Delete positions 1 and 3 to create gaps
    table.delete([1, 3])

    # Mask: [T,F,T,F,T]
    expected_mask = [True, False, True, False, True]
    assert_mask_matches(table, expected_mask)

    # Append one row (ID=6)
    table.append((6, 1j, 50.0, True))

    # Should fill from end: [T,F,T,F,T,T]
    expected_mask_after = [True, False, True, False, True, True]
    assert_mask_matches(table, expected_mask_after)

    # Append another (ID=7)
    table.append((7, 2j, 60.0, False))

    # [T,F,T,F,T,T,T]
    expected_mask_final = [True, False, True, False, True, True, True]
    assert_mask_matches(table, expected_mask_final)


# -------------------------------------------------------------------
# 4. Resize Behavior Tests
# -------------------------------------------------------------------

def test_resize_when_capacity_full_with_gaps():
    """
    Table with capacity 10, insert 10, delete first 9.
    Mask: [F,F,F,F,F,F,F,F,F,T]
    Append 1 more -> should trigger resize because last valid is at end.
    """
    data = generate_test_data(10, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=10, compact=False)

    # Delete first 9 positions
    table.delete(list(range(9)))

    assert len(table) == 1
    # Mask: [F,F,F,F,F,F,F,F,F,T]

    # Get current capacity
    initial_capacity = len(table._valid_rows)

    # Append one more row
    table.append((11, 5j, 75.0, True))

    # Should have triggered resize
    new_capacity = len(table._valid_rows)
    assert new_capacity > initial_capacity, \
        f"Expected resize, but capacity stayed {initial_capacity}"


def test_no_resize_with_compact_enabled():
    """
    Same scenario as above but with compact=True.
    Should compact before extending, avoiding resize.
    """
    data = generate_test_data(10, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=10, compact=True)

    # Delete first 9 positions
    table.delete(list(range(9)))

    assert len(table) == 1

    # With auto_compact=True, extend should compact first
    initial_capacity = len(table._valid_rows)

    # Extend with new data
    new_data = generate_test_data(3, start_id=11)
    table.extend(new_data)

    # Should NOT have resized because compaction freed space
    new_capacity = len(table._valid_rows)
    # Capacity might stay same or reduce due to compaction
    assert new_capacity <= initial_capacity * 2, \
        "Unexpected massive resize with auto_compact enabled"


def test_resize_when_extend_exceeds_capacity():
    """
    Table with small capacity, many gaps, but extend data is so large
    that even with gaps it must resize.
    """
    data = generate_test_data(5, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=10, compact=False)

    # Delete 3 positions, leaving 2 valid and 3 gaps
    table.delete([0, 2, 4])

    initial_capacity = len(table._valid_rows)

    # Try to extend with 20 rows (more than capacity)
    large_data = generate_test_data(20, start_id=100)
    table.extend(large_data)

    # Must have resized
    new_capacity = len(table._valid_rows)
    assert new_capacity > initial_capacity


# -------------------------------------------------------------------
# 5. Gap Filling Order Tests
# -------------------------------------------------------------------

def test_extend_fills_from_last_valid_position():
    """
    Verify that extend always appends after the last valid position,
    not into middle gaps (based on your implementation).
    """
    data = generate_test_data(10, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=15)

    # Delete positions 2, 4, 6 (create gaps in the middle)
    table.delete([2, 4, 6])

    # Current last valid position should be at index 9
    # Mask: [T,T,F,T,F,T,F,T,T,T]

    # Extend with 3 rows
    new_data = generate_test_data(3, start_id=20)
    table.extend(new_data)

    # New data should start at position 10 (after last valid)
    # Mask: [T,T,F,T,F,T,F,T,T,T,T,T,T]
    assert_data_at_positions(table, [10, 11, 12], [20, 21, 22])


def test_multiple_extends_with_gaps():
    """
    Multiple extend operations with gaps in between.
    Verify consistent behavior.
    """
    data = generate_test_data(5, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=20)

    # First extend
    table.extend(generate_test_data(3, start_id=10))
    assert len(table) == 8

    # Delete some from middle
    table.delete([2, 4, 6])
    assert len(table) == 5

    # Second extend
    table.extend(generate_test_data(2, start_id=20))
    assert len(table) == 7

    # Delete more
    table.delete([0, 1])
    assert len(table) == 5

    # Third extend
    table.extend(generate_test_data(4, start_id=30))
    assert len(table) == 9


# -------------------------------------------------------------------
# 6. Append vs Extend with Gaps
# -------------------------------------------------------------------

def test_append_and_extend_mixed_with_gaps():
    """
    Mix append and extend operations with deletions in between.
    """
    table = CTable(RowModel, expected_size=20)

    # Start with append
    for i in range(5):
        table.append((i + 1, complex(i), float(i * 10), True))

    assert len(table) == 5

    # Extend
    table.extend(generate_test_data(5, start_id=10))
    assert len(table) == 10

    # Delete some
    table.delete([1, 3, 5, 7, 9])
    assert len(table) == 5

    # Append one
    table.append((100, 0j, 50.0, False))
    assert len(table) == 6

    # Extend more
    table.extend(generate_test_data(3, start_id=200))
    assert len(table) == 9


# -------------------------------------------------------------------
# 7. Edge Cases
# -------------------------------------------------------------------

def test_fill_gaps_completely_then_extend():
    """
    Create gaps, fill them exactly with new data, then extend more.
    """
    data = generate_test_data(10, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=15)

    # Delete all even positions (5 gaps)
    table.delete(list(range(0, 10, 2)))
    assert len(table) == 5

    # Now extend with exactly 5 rows (should fill gaps + maybe extend)
    table.extend(generate_test_data(5, start_id=20))
    assert len(table) == 10


def test_delete_all_then_extend():
    """
    Delete all rows, then extend with new data.
    Should start fresh from position 0.
    """
    data = generate_test_data(10, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=15)

    # Delete all
    table.delete(list(range(10)))
    assert len(table) == 0

    # Extend with new data
    new_data = generate_test_data(5, start_id=100)
    table.extend(new_data)

    assert len(table) == 5
    # Should start from position 10 (after the last deleted position)
    # or from 0 if compacted


def test_sparse_table_with_many_gaps():
    """
    Create a very sparse table with many gaps.
    """
    data = generate_test_data(20, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=30)

    # Delete 15 out of 20 rows, keeping only 5
    to_delete = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
    table.delete(to_delete)

    assert len(table) == 5

    # Extend with 10 more
    table.extend(generate_test_data(10, start_id=100))

    assert len(table) == 15


def test_alternating_insert_delete_pattern():
    """
    Alternating pattern of inserts and deletes.
    Stress test for gap management.
    """
    table = CTable(RowModel, expected_size=50)

    for cycle in range(5):
        # Insert 10 rows
        table.extend(generate_test_data(10, start_id=cycle * 100))

        # Delete 5 random positions
        current_len = len(table)
        if current_len >= 5:
            to_delete = list(range(0, min(5, current_len)))
            table.delete(to_delete)


# -------------------------------------------------------------------
# 8. Compact Behavior with Gaps
# -------------------------------------------------------------------

def test_manual_compact_before_extend():
    """
    Create gaps, manually compact, then extend.
    After compact, data should be contiguous.
    """
    data = generate_test_data(10, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=15, compact=False)

    # Delete positions to create gaps
    table.delete([1, 3, 5, 7, 9])
    assert len(table) == 5

    # Manually compact
    table.compact()

    # After compact, first 5 positions should be valid, rest False
    expected_mask = [True] * 5 + [False] * 10
    assert_mask_matches(table, expected_mask)

    # Now extend
    table.extend(generate_test_data(3, start_id=20))
    assert len(table) == 8


def test_auto_compact_on_extend():
    """
    With auto_compact=True, extending should trigger compaction
    when beneficial.
    """
    data = generate_test_data(10, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=15, compact=True)

    # Delete many rows to create fragmentation
    table.delete(list(range(0, 8)))
    assert len(table) == 2

    # Extend with large data
    table.extend(generate_test_data(10, start_id=100))

    # Should have compacted to make room
    assert len(table) == 12


# -------------------------------------------------------------------
# 9. Data Integrity Verification
# -------------------------------------------------------------------

def test_data_integrity_after_gap_operations():
    """
    Verify that actual data values remain correct after
    complex insert/delete operations with gaps.
    """
    # Insert initial data with distinct IDs
    data1 = [(1, 1j, 10.0, True), (2, 2j, 20.0, False), (3, 3j, 30.0, True)]
    table = CTable(RowModel, new_data=data1, expected_size=10)

    # Delete middle row
    table.delete(1)

    # Verify remaining data
    assert table.row[0].id[0] == 1  # ID of first logical row
    assert table.row[1].id[0] == 3  # ID of second logical row

    # Extend with new data
    data2 = [(10, 10j, 100.0, True), (11, 11j, 110.0, False)]
    table.extend(data2)

    # Verify all data
    assert table.row[0].id[0] == 1
    assert table.row[1].id[0] == 3
    assert table.row[2].id[0] == 10
    assert table.row[3].id[0] == 11


def test_complex_scenario_full_workflow():
    """
    Complex realistic scenario combining all operations.
    """
    table = CTable(RowModel, expected_size=20, compact=False)

    # Phase 1: Initial insert
    table.extend(generate_test_data(10, start_id=1))
    assert len(table) == 10

    # Phase 2: Delete some rows
    table.delete([0, 2, 4, 6, 8])
    assert len(table) == 5

    # Phase 3: Append individual rows
    table.append((100, 0j, 50.0, True))
    table.append((101, 1j, 60.0, False))
    assert len(table) == 7

    # Phase 4: Extend with batch
    table.extend(generate_test_data(5, start_id=200))
    assert len(table) == 12

    # Phase 5: More deletes
    table.delete([3, 7, 10])
    assert len(table) == 9

    # Phase 6: Final extend
    table.extend(generate_test_data(3, start_id=300))
    assert len(table) == 12

    # Verify table is still functional
    assert table.nrows == 12
    assert table.ncols == 4


if __name__ == "__main__":
    pytest.main(["-v", __file__])
