#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sudoku with Pygame UI, logical generator/solver, difficulty modes,
timer + best times, unlimited mistake counter, and ML docking via SudokuEnv.

Author: You
License: MIT
"""

from __future__ import annotations
import sys
import os
import json
import time
import random
from copy import deepcopy
from typing import List, Tuple, Optional, Dict, Set, Any

import numpy as np

# ----------------------------
# Config & Constants
# ----------------------------

GRID = 9
BOX = 3
DIGITS = set(range(1, 10))

# UI config
WIDTH, HEIGHT = 720, 840  # bottom area for timer and stats
CELL_SIZE = WIDTH // GRID
MARGIN = 2  # optional margins inside cells for drawing
BG_COLOR = (250, 250, 250)
LINE_COLOR = (30, 30, 30)
SUBLINE_COLOR = (120, 120, 120)
FIXED_NUM_COLOR = (20, 20, 20)
USER_NUM_COLOR = (20, 70, 220)
WRONG_NUM_COLOR = (220, 30, 30)
SELECT_COLOR = (210, 230, 255)
HIGHLIGHT_SAME_COLOR = (235, 245, 255)
NOTE_COLOR = (120, 120, 120)
CONFLICT_BG_COLOR = (255, 230, 230)

# Font sizes
NUMBER_FONT_SIZE = int(CELL_SIZE * 0.55)
NOTE_FONT_SIZE = int(CELL_SIZE * 0.2)
UI_FONT_SIZE = 22
TITLE_FONT_SIZE = 28

# Best times storage
STATS_FILE = "sudoku_stats.json"

# Difficulty strategy profiles
DIFFICULTY_PROFILES = {
    "easy": {
        "target_givens_min": 36,
        "target_givens_max": 45,
        "strategies": [
            "naked_singles",
            "hidden_singles",
        ],
        "max_attempts": 2000,
    },
    "medium": {
        "target_givens_min": 28,
        "target_givens_max": 35,
        "strategies": [
            "naked_singles",
            "hidden_singles",
            "naked_pairs_triples",
            "hidden_pairs",
            "pointing_pairs",
            "claiming_pairs",
        ],
        "max_attempts": 4000,
    },
    "hard": {
        "target_givens_min": 22,
        "target_givens_max": 28,
        "strategies": [
            "naked_singles",
            "hidden_singles",
            "naked_pairs_triples",
            "hidden_pairs",
            "pointing_pairs",
            "claiming_pairs",
            # Hard constraint uses all implemented logic. This can still be very challenging.
        ],
        "max_attempts": 8000,
    },
}

# ----------------------------
# Utility
# ----------------------------

def deep_copy_grid(grid: List[List[int]]) -> List[List[int]]:
    return [row[:] for row in grid]

def coords_to_box(r: int, c: int) -> Tuple[int, int]:
    return (r // BOX, c // BOX)

def unit_indices() -> Dict[str, List[List[Tuple[int, int]]]]:
    """Return row, col, box index sets for iteration."""
    rows = [[(r, c) for c in range(GRID)] for r in range(GRID)]
    cols = [[(r, c) for r in range(GRID)] for c in range(GRID)]
    boxes = []
    for br in range(BOX):
        for bc in range(BOX):
            boxes.append([(r, c) for r in range(br * BOX, (br + 1) * BOX)
                                   for c in range(bc * BOX, (bc + 1) * BOX)])
    return {"rows": rows, "cols": cols, "boxes": boxes}

UNITS = unit_indices()

# ----------------------------
# Core Sudoku Board Model
# ----------------------------

class SudokuBoard:
    """
    Holds grid, fixed cells, solution, notes, and candidates.
    Provides validation, mutation, and solving helpers.
    """
    def __init__(self, puzzle: List[List[int]], solution: List[List[int]]):
        self.grid = deep_copy_grid(puzzle)  # current state
        self.solution = deep_copy_grid(solution)  # ground truth
        self.fixed = [[puzzle[r][c] != 0 for c in range(GRID)] for r in range(GRID)]
        # notes[r][c] is a set of candidates 1..9
        self.notes: List[List[Set[int]]] = [[set() for _ in range(GRID)] for _ in range(GRID)]
        # candidate grid (updated on demand)
        self.candidates: List[List[Set[int]]] = [[set() for _ in range(GRID)] for _ in range(GRID)]
        self.update_all_candidates()

    def clone(self) -> "SudokuBoard":
        clone = SudokuBoard(self.grid, self.solution)
        clone.fixed = [row[:] for row in self.fixed]
        clone.notes = [ [s.copy() for s in row] for row in self.notes ]
        clone.candidates = [ [s.copy() for s in row] for row in self.candidates ]
        return clone

    def is_solved(self) -> bool:
        for r in range(GRID):
            for c in range(GRID):
                if self.grid[r][c] == 0:
                    return False
        return self.is_valid_complete()

    def is_valid_complete(self) -> bool:
        # Checks each row/col/box contains 1..9
        for r in range(GRID):
            row = [self.grid[r][c] for c in range(GRID)]
            if set(row) != DIGITS:
                return False
        for c in range(GRID):
            col = [self.grid[r][c] for r in range(GRID)]
            if set(col) != DIGITS:
                return False
        for br in range(BOX):
            for bc in range(BOX):
                block = [self.grid[r][c] for r in range(br*BOX, (br+1)*BOX)
                                          for c in range(bc*BOX, (bc+1)*BOX)]
                if set(block) != DIGITS:
                    return False
        return True

    def check_legal_placement(self, r: int, c: int, val: int) -> bool:
        # Check row/col/box constraints relative to current grid
        if val == 0:
            return True
        for cc in range(GRID):
            if cc != c and self.grid[r][cc] == val:
                return False
        for rr in range(GRID):
            if rr != r and self.grid[rr][c] == val:
                return False
        br, bc = coords_to_box(r, c)
        for rr in range(br * BOX, (br + 1) * BOX):
            for cc in range(bc * BOX, (bc + 1) * BOX):
                if (rr != r or cc != c) and self.grid[rr][cc] == val:
                    return False
        return True

    def update_all_candidates(self) -> None:
        for r in range(GRID):
            for c in range(GRID):
                if self.grid[r][c] != 0:
                    self.candidates[r][c] = set()
                else:
                    possible = set(DIGITS)
                    # remove seen row/col/box
                    for cc in range(GRID):
                        v = self.grid[r][cc]
                        if v != 0 and v in possible:
                            possible.remove(v)
                    for rr in range(GRID):
                        v = self.grid[rr][c]
                        if v != 0 and v in possible:
                            possible.remove(v)
                    br, bc = coords_to_box(r, c)
                    for rr in range(br * BOX, (br + 1) * BOX):
                        for cc in range(bc * BOX, (bc + 1) * BOX):
                            v = self.grid[rr][cc]
                            if v != 0 and v in possible:
                                possible.remove(v)
                    self.candidates[r][c] = possible

    def place(self, r: int, c: int, val: int) -> bool:
        """
        Place a value if the cell is editable and update candidates. Returns True if placed.
        """
        if self.fixed[r][c]:
            return False
        self.grid[r][c] = val
        # Clear notes if placed
        if val != 0:
            self.notes[r][c].clear()
        self.update_all_candidates()
        return True

    def clear_cell(self, r: int, c: int) -> bool:
        if self.fixed[r][c]:
            return False
        self.grid[r][c] = 0
        self.notes[r][c].clear()
        self.update_all_candidates()
        return True

    def add_note(self, r: int, c: int, val: int) -> bool:
        if self.fixed[r][c] or self.grid[r][c] != 0:
            return False
        if val in DIGITS:
            if val in self.notes[r][c]:
                self.notes[r][c].remove(val)
            else:
                self.notes[r][c].add(val)
            return True
        return False

    def conflicting_cells(self) -> Set[Tuple[int, int]]:
        """
        Return set of cells that violate Sudoku constraints (duplicates row/col/box).
        Useful to color background.
        """
        conflicts: Set[Tuple[int, int]] = set()
        # Rows
        for r in range(GRID):
            counts: Dict[int, List[int]] = {}
            for c in range(GRID):
                v = self.grid[r][c]
                if v != 0:
                    counts.setdefault(v, []).append(c)
            for v, locs in counts.items():
                if len(locs) > 1:
                    for c in locs:
                        conflicts.add((r, c))
        # Cols
        for c in range(GRID):
            counts: Dict[int, List[int]] = {}
            for r in range(GRID):
                v = self.grid[r][c]
                if v != 0:
                    counts.setdefault(v, []).append(r)
            for v, locs in counts.items():
                if len(locs) > 1:
                    for r in locs:
                        conflicts.add((r, c))
        # Boxes
        for br in range(BOX):
            for bc in range(BOX):
                counts: Dict[int, List[Tuple[int, int]]] = {}
                for r in range(br * BOX, (br + 1) * BOX):
                    for c in range(bc * BOX, (bc + 1) * BOX):
                        v = self.grid[r][c]
                        if v != 0:
                            counts.setdefault(v, []).append((r, c))
                for v, locs in counts.items():
                    if len(locs) > 1:
                        for (r, c) in locs:
                            conflicts.add((r, c))
        return conflicts

# ----------------------------
# Backtracking Solver / Counter for Uniqueness
# ----------------------------

def find_empty(grid: List[List[int]]) -> Optional[Tuple[int, int]]:
    for r in range(GRID):
        for c in range(GRID):
            if grid[r][c] == 0:
                return (r, c)
    return None

def valid_for(grid: List[List[int]], r: int, c: int, v: int) -> bool:
    for cc in range(GRID):
        if cc != c and grid[r][cc] == v:
            return False
    for rr in range(GRID):
        if rr != r and grid[rr][c] == v:
            return False
    br, bc = coords_to_box(r, c)
    for rr in range(br*BOX, (br+1)*BOX):
        for cc in range(bc*BOX, (bc+1)*BOX):
            if (rr != r or cc != c) and grid[rr][cc] == v:
                return False
    return True

def solve_backtracking(grid: List[List[int]]) -> bool:
    empty = find_empty(grid)
    if not empty:
        return True
    r, c = empty
    for v in range(1, 10):
        if valid_for(grid, r, c, v):
            grid[r][c] = v
            if solve_backtracking(grid):
                return True
            grid[r][c] = 0
    return False

def count_solutions(grid: List[List[int]], cap: int = 2) -> int:
    """Count solutions up to 'cap'."""
    # Copy grid to avoid mutating caller
    g = deep_copy_grid(grid)
    count = 0

    def backtrack() -> bool:
        nonlocal count, g
        if count >= cap:
            return True
        empty = find_empty(g)
        if not empty:
            count += 1
            return False  # continue looking for more
        r, c = empty
        for v in range(1, 10):
            if valid_for(g, r, c, v):
                g[r][c] = v
                stop = backtrack()
                g[r][c] = 0
                if stop:
                    return True
        return False

    backtrack()
    return count

# ----------------------------
# Logical Solver (Human Strategies)
# ----------------------------

class LogicSolver:
    """
    Apply a set of logical strategies step-by-step until solved or stuck.
    Strategies: naked singles, hidden singles, naked pairs/triples,
    hidden pairs, pointing pairs, claiming pairs.
    """
    def __init__(self, allowed: List[str]):
        self.allowed = set(allowed)

    def solve_step(self, board: SudokuBoard) -> bool:
        """
        Apply one step of any allowed strategy.
        Return True if something changed; False if no progress.
        """
        board.update_all_candidates()

        # Strategy order (roughly easy -> hard)
        if "naked_singles" in self.allowed:
            if self._naked_singles(board):
                return True
        if "hidden_singles" in self.allowed:
            if self._hidden_singles(board):
                return True
        if "naked_pairs_triples" in self.allowed:
            if self._naked_pairs_triples(board):
                return True
        if "hidden_pairs" in self.allowed:
            if self._hidden_pairs(board):
                return True
        if "pointing_pairs" in self.allowed:
            if self._pointing_pairs(board):
                return True
        if "claiming_pairs" in self.allowed:
            if self._claiming_pairs(board):
                return True

        return False

    def solve_fully(self, board: SudokuBoard, max_steps: int = 10000) -> bool:
        """
        Repeatedly call solve_step until no more progress or solved.
        Returns True if solved, False if stuck.
        """
        steps = 0
        while steps < max_steps:
            if board.is_solved():
                return True
            changed = self.solve_step(board)
            if not changed:
                break
            steps += 1
        return board.is_solved()

    # ---- Strategies ----

    def _naked_singles(self, board: SudokuBoard) -> bool:
        changed = False
        for r in range(GRID):
            for c in range(GRID):
                if board.grid[r][c] == 0:
                    cand = board.candidates[r][c]
                    if len(cand) == 1:
                        v = next(iter(cand))
                        board.place(r, c, v)
                        changed = True
        return changed

    def _hidden_singles(self, board: SudokuBoard) -> bool:
        changed = False
        # Rows
        for r in range(GRID):
            candidate_positions: Dict[int, List[int]] = {d: [] for d in DIGITS}
            for c in range(GRID):
                if board.grid[r][c] == 0:
                    for v in board.candidates[r][c]:
                        candidate_positions[v].append(c)
            for v, cols in candidate_positions.items():
                if len(cols) == 1:
                    c = cols[0]
                    board.place(r, c, v)
                    changed = True
        # Cols
        for c in range(GRID):
            candidate_positions: Dict[int, List[int]] = {d: [] for d in DIGITS}
            for r in range(GRID):
                if board.grid[r][c] == 0:
                    for v in board.candidates[r][c]:
                        candidate_positions[v].append(r)
            for v, rows in candidate_positions.items():
                if len(rows) == 1:
                    r = rows[0]
                    board.place(r, c, v)
                    changed = True
        # Boxes
        for br in range(BOX):
            for bc in range(BOX):
                candidate_positions: Dict[int, List[Tuple[int, int]]] = {d: [] for d in DIGITS}
                for r in range(br*BOX, (br+1)*BOX):
                    for c in range(bc*BOX, (bc+1)*BOX):
                        if board.grid[r][c] == 0:
                            for v in board.candidates[r][c]:
                                candidate_positions[v].append((r, c))
                for v, locs in candidate_positions.items():
                    if len(locs) == 1:
                        r, c = locs[0]
                        board.place(r, c, v)
                        changed = True
        return changed

    def _naked_pairs_triples(self, board: SudokuBoard) -> bool:
        """
        Find naked pairs and triples in each unit and eliminate from others.
        """
        changed = False
        # Rows
        for r in range(GRID):
            changed |= self._naked_subset_in_unit([(r, c) for c in range(GRID)], board, sizes=(2, 3))
        # Cols
        for c in range(GRID):
            changed |= self._naked_subset_in_unit([(r, c) for r in range(GRID)], board, sizes=(2, 3))
        # Boxes
        for br in range(BOX):
            for bc in range(BOX):
                unit = [(r, c) for r in range(br*BOX, (br+1)*BOX) for c in range(bc*BOX, (bc+1)*BOX)]
                changed |= self._naked_subset_in_unit(unit, board, sizes=(2, 3))
        return changed

    def _naked_subset_in_unit(self, unit: List[Tuple[int, int]], board: SudokuBoard, sizes=(2, 3)) -> bool:
        changed = False
        # Map frozenset(cands) -> list of cells
        cand_map: Dict[frozenset, List[Tuple[int, int]]] = {}
        for (r, c) in unit:
            if board.grid[r][c] == 0 and 1 <= len(board.candidates[r][c]) <= max(sizes):
                fs = frozenset(board.candidates[r][c])
                cand_map.setdefault(fs, []).append((r, c))
        for fs, cells in cand_map.items():
            if len(fs) in sizes and len(cells) == len(fs):
                # eliminate these candidates from other cells in unit
                for (r, c) in unit:
                    if (r, c) not in cells and board.grid[r][c] == 0:
                        before = len(board.candidates[r][c])
                        board.candidates[r][c] -= set(fs)
                        if len(board.candidates[r][c]) != before:
                            changed = True
                # place singles created
                for (r, c) in unit:
                    if board.grid[r][c] == 0 and len(board.candidates[r][c]) == 1:
                        v = next(iter(board.candidates[r][c]))
                        board.place(r, c, v)
                        changed = True
        return changed

    def _hidden_pairs(self, board: SudokuBoard) -> bool:
        """
        In each unit, find pairs of candidates that appear exactly twice across the unit,
        and restrict those two cells to those two candidates.
        """
        changed = False
        # Units: rows, cols, boxes
        for unit in UNITS["rows"] + UNITS["cols"] + UNITS["boxes"]:
            # count occurrences
            occ: Dict[int, List[Tuple[int, int]]] = {d: [] for d in DIGITS}
            for (r, c) in unit:
                if board.grid[r][c] == 0:
                    for v in board.candidates[r][c]:
                        occ[v].append((r, c))
            # find pairs
            pairs: List[Tuple[int, int]] = []
            for a in range(1, 10):
                if len(occ[a]) == 2:
                    for b in range(a+1, 10):
                        if len(occ[b]) == 2 and set(occ[a]) == set(occ[b]):
                            pairs.append((a, b))
            # apply restrictions
            for (a, b) in pairs:
                targets = occ[a]  # same as occ[b]
                for (r, c) in targets:
                    before = board.candidates[r][c].copy()
                    board.candidates[r][c] &= {a, b}
                    if board.candidates[r][c] != before:
                        changed = True
                    if len(board.candidates[r][c]) == 1:
                        v = next(iter(board.candidates[r][c]))
                        board.place(r, c, v)
                        changed = True
        return changed

    def _pointing_pairs(self, board: SudokuBoard) -> bool:
        """
        In a box, if a candidate is confined to a single row or column, eliminate from that row/col outside the box.
        """
        changed = False
        for br in range(BOX):
            for bc in range(BOX):
                unit = [(r, c) for r in range(br*BOX, (br+1)*BOX) for c in range(bc*BOX, (bc+1)*BOX)]
                # map candidate -> rows/cols inside box
                for v in range(1, 10):
                    cells = [(r, c) for (r, c) in unit if board.grid[r][c] == 0 and v in board.candidates[r][c]]
                    if not cells:
                        continue
                    rows = {r for (r, c) in cells}
                    cols = {c for (r, c) in cells}
                    # confined to a single row in this box
                    if len(rows) == 1:
                        r = next(iter(rows))
                        # eliminate v from row r outside this box
                        for c in range(GRID):
                            if c < bc*BOX or c >= (bc+1)*BOX:
                                if board.grid[r][c] == 0 and v in board.candidates[r][c]:
                                    board.candidates[r][c].discard(v)
                                    changed = True
                    # confined to a single column in this box
                    if len(cols) == 1:
                        c = next(iter(cols))
                        for r in range(GRID):
                            if r < br*BOX or r >= (br+1)*BOX:
                                if board.grid[r][c] == 0 and v in board.candidates[r][c]:
                                    board.candidates[r][c].discard(v)
                                    changed = True
        # fill any singles created
        for r in range(GRID):
            for c in range(GRID):
                if board.grid[r][c] == 0 and len(board.candidates[r][c]) == 1:
                    v = next(iter(board.candidates[r][c]))
                    board.place(r, c, v)
                    changed = True
        return changed

    def _claiming_pairs(self, board: SudokuBoard) -> bool:
        """
        In a row/col, if a candidate's occurrences are confined to a single box,
        eliminate from the rest of the box.
        """
        changed = False
        # Rows
        for r in range(GRID):
            for v in range(1, 10):
                cols = [c for c in range(GRID) if board.grid[r][c] == 0 and v in board.candidates[r][c]]
                if not cols:
                    continue
                boxes = {c // BOX for c in cols}
                if len(boxes) == 1:
                    bc = next(iter(boxes))
                    # eliminate v in that box but not row r
                    for rr in range((r//BOX)*BOX, (r//BOX)*BOX + BOX):
                        for cc in range(bc*BOX, (bc+1)*BOX):
                            if rr != r and board.grid[rr][cc] == 0 and v in board.candidates[rr][cc]:
                                board.candidates[rr][cc].discard(v)
                                changed = True
        # Cols
        for c in range(GRID):
            for v in range(1, 10):
                rows = [r for r in range(GRID) if board.grid[r][c] == 0 and v in board.candidates[r][c]]
                if not rows:
                    continue
                boxes = {r // BOX for r in rows}
                if len(boxes) == 1:
                    br = next(iter(boxes))
                    for rr in range(br*BOX, (br+1)*BOX):
                        for cc in range((c//BOX)*BOX, (c//BOX)*BOX + BOX):
                            if cc != c and board.grid[rr][cc] == 0 and v in board.candidates[rr][cc]:
                                board.candidates[rr][cc].discard(v)
                                changed = True
        # fill singles created
        for r in range(GRID):
            for c in range(GRID):
                if board.grid[r][c] == 0 and len(board.candidates[r][c]) == 1:
                    v = next(iter(board.candidates[r][c]))
                    board.place(r, c, v)
                    changed = True
        return changed

# ----------------------------
# Puzzle Generator
# ----------------------------

def generate_full_solution(seed: Optional[int] = None) -> List[List[int]]:
    """Generate a fully solved grid via randomized backtracking."""
    if seed is not None:
        random.seed(seed)

    grid = [[0]*GRID for _ in range(GRID)]
    digits = list(range(1, 10))

    def backtrack() -> bool:
        empty = find_empty(grid)
        if not empty:
            return True
        r, c = empty
        random.shuffle(digits)
        for v in digits:
            if valid_for(grid, r, c, v):
                grid[r][c] = v
                if backtrack():
                    return True
                grid[r][c] = 0
        return False

    backtrack()
    return grid

def generate_puzzle(difficulty: str, rng_seed: Optional[int] = None) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Create a puzzle and its solution under the difficulty profile:
      - Unique solution (checked via count_solutions)
      - Solvable by allowed logical strategies (no guessing)
      - Target clue count depends on difficulty profile
    """
    profile = DIFFICULTY_PROFILES[difficulty]
    target_min = profile["target_givens_min"]
    target_max = profile["target_givens_max"]
    strategies = profile["strategies"]
    max_attempts = profile["max_attempts"]

    if rng_seed is not None:
        random.seed(rng_seed)

    # Start from a random solved grid
    solution = generate_full_solution()

    # Start puzzle as full solution, then remove clues while preserving constraints
    puzzle = deep_copy_grid(solution)

    # Generate a list of positions to try removing, randomized
    positions = [(r, c) for r in range(GRID) for c in range(GRID)]
    random.shuffle(positions)

    attempts = 0
    # Greedy removal: try to remove if uniqueness + logical solvability preserved
    for (r, c) in positions:
        if attempts >= max_attempts:
            break
        if puzzle[r][c] == 0:
            continue
        saved = puzzle[r][c]
        puzzle[r][c] = 0

        # Check unique
        if count_solutions(puzzle, cap=2) != 1:
            # not unique; revert
            puzzle[r][c] = saved
            attempts += 1
            continue

        # Check logic solvability under allowed strategies
        test_board = SudokuBoard(puzzle, solution)
        solver = LogicSolver(allowed=strategies)
        if not solver.solve_fully(test_board):
            # not solvable by our strategies; revert
            puzzle[r][c] = saved
            attempts += 1
            continue

        attempts += 1

        # Stop removing if we already hit target givens
        givens = sum(1 for rr in range(GRID) for cc in range(GRID) if puzzle[rr][cc] != 0)
        if givens <= target_min:
            break

    # If too many givens remain (> target_max), try extra removals with additional passes
    # but keep uniqueness + logical solvability
    if sum(1 for rr in range(GRID) for cc in range(GRID) if puzzle[rr][cc] != 0) > target_max:
        more_positions = [(r, c) for r in range(GRID) for c in range(GRID) if puzzle[r][c] != 0]
        random.shuffle(more_positions)
        for (r, c) in more_positions:
            if attempts >= max_attempts:
                break
            saved = puzzle[r][c]
            puzzle[r][c] = 0
            if count_solutions(puzzle, cap=2) != 1:
                puzzle[r][c] = saved
                attempts += 1
                continue
            test_board = SudokuBoard(puzzle, solution)
            solver = LogicSolver(allowed=strategies)
            if not solver.solve_fully(test_board):
                puzzle[r][c] = saved
                attempts += 1
                continue
            attempts += 1
            givens = sum(1 for rr in range(GRID) for cc in range(GRID) if puzzle[rr][cc] != 0)
            if givens <= target_min:
                break

    # Final verification
    if count_solutions(puzzle, cap=2) != 1:
        # regenerate if something went wrong
        return generate_puzzle(difficulty, rng_seed=None)

    # Guarantee logical solvability
    tb = SudokuBoard(puzzle, solution)
    sol = LogicSolver(allowed=strategies)
    if not sol.solve_fully(tb):
        return generate_puzzle(difficulty, rng_seed=None)

    return puzzle, solution

# ----------------------------
# Best Times Storage
# ----------------------------

def load_stats() -> Dict[str, Any]:
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_stats(stats: Dict[str, Any]) -> None:
    try:
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    except Exception:
        pass

def update_best_time(difficulty: str, elapsed_seconds: int) -> Optional[int]:
    stats = load_stats()
    best_times = stats.get("best_times", {})
    current_best = best_times.get(difficulty)
    if (current_best is None) or (elapsed_seconds < current_best):
        best_times[difficulty] = elapsed_seconds
        stats["best_times"] = best_times
        save_stats(stats)
        return elapsed_seconds
    return None

def get_best_time(difficulty: str) -> Optional[int]:
    stats = load_stats()
    best_times = stats.get("best_times", {})
    return best_times.get(difficulty)

# ----------------------------
# UI Engine with Pygame
# ----------------------------

class Game:
    """
    Manage UI state and render with Pygame.
    """
    def __init__(self, difficulty: str = "medium"):
        import pygame  # Late import so module can be imported in notebooks headless

        self.pygame = pygame
        self.screen = None
        self.clock = None
        self.font_num = None
        self.font_note = None
        self.font_ui = None
        self.font_title = None

        # State
        self.difficulty = difficulty
        self.puzzle, self.solution = generate_puzzle(self.difficulty)
        self.board = SudokuBoard(self.puzzle, self.solution)
        self.selected: Optional[Tuple[int, int]] = None
        self.notes_mode: bool = False
        self.mistakes: int = 0
        self.start_time: float = time.time()
        self.solved: bool = False
        self.last_finish_time: Optional[int] = None

    # ---- ML docking helpers ----

    def get_mistakes(self) -> int:
        return self.mistakes

    def get_elapsed_time(self) -> int:
        return int(time.time() - self.start_time)

    def get_state(self) -> np.ndarray:
        return np.array(self.board.grid, dtype=np.int32)

    def reset(self, difficulty: Optional[str] = None) -> None:
        if difficulty is not None:
            self.difficulty = difficulty
        self.puzzle, self.solution = generate_puzzle(self.difficulty)
        self.board = SudokuBoard(self.puzzle, self.solution)
        self.selected = None
        self.notes_mode = False
        self.mistakes = 0
        self.start_time = time.time()
        self.solved = False
        self.last_finish_time = None

    # ---- Pygame ----

    def init_pygame(self):
        pygame = self.pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Sudoku")
        self.clock = pygame.time.Clock()

        self.font_num = pygame.font.SysFont("arial", NUMBER_FONT_SIZE, bold=True)
        self.font_note = pygame.font.SysFont("arial", NOTE_FONT_SIZE)
        self.font_ui = pygame.font.SysFont("arial", UI_FONT_SIZE)
        self.font_title = pygame.font.SysFont("arial", TITLE_FONT_SIZE, bold=True)

    def draw(self, finalize: bool = True):
        pygame = self.pygame
        self.screen.fill(BG_COLOR)

        # Highlight same numbers as selected
        if self.selected:
            sr, sc = self.selected
            selected_val = self.board.grid[sr][sc]
            # Shade selected cell
            pygame.draw.rect(self.screen, SELECT_COLOR,
                             (sc*CELL_SIZE, sr*CELL_SIZE, CELL_SIZE, CELL_SIZE))
            # Shade same number cells (not selected)
            if selected_val != 0:
                for r in range(GRID):
                    for c in range(GRID):
                        if (r, c) != (sr, sc) and self.board.grid[r][c] == selected_val:
                            pygame.draw.rect(self.screen, HIGHLIGHT_SAME_COLOR,
                                             (c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Shade conflict cells
        conflicts = self.board.conflicting_cells()
        for (r, c) in conflicts:
            pygame.draw.rect(self.screen, CONFLICT_BG_COLOR,
                             (c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Grid lines
        for i in range(GRID + 1):
            thick = 3 if i % BOX == 0 else 1
            pygame.draw.line(self.screen, LINE_COLOR, (0, i*CELL_SIZE), (WIDTH, i*CELL_SIZE), thick)
            pygame.draw.line(self.screen, LINE_COLOR, (i*CELL_SIZE, 0), (i*CELL_SIZE, WIDTH), thick)

        # Numbers and notes
        for r in range(GRID):
            for c in range(GRID):
                val = self.board.grid[r][c]
                if val != 0:
                    x = c * CELL_SIZE + CELL_SIZE // 2
                    y = r * CELL_SIZE + CELL_SIZE // 2
                    # Wrong numbers always red if not matching solution
                    color = FIXED_NUM_COLOR if self.board.fixed[r][c] else (WRONG_NUM_COLOR if val != self.solution[r][c] else USER_NUM_COLOR)
                    text = self.font_num.render(str(val), True, color)
                    rect = text.get_rect(center=(x, y))
                    self.screen.blit(text, rect)
                else:
                    # render notes (3x3 mini-grid)
                    if self.board.notes[r][c]:
                        for n in range(1, 10):
                            if n in self.board.notes[r][c]:
                                nx = c * CELL_SIZE + int(( (n - 1) % 3 + 0.5 ) * (CELL_SIZE / 3))
                                ny = r * CELL_SIZE + int(( (n - 1) // 3 + 0.5 ) * (CELL_SIZE / 3))
                                t = self.font_note.render(str(n), True, NOTE_COLOR)
                                rect = t.get_rect(center=(nx, ny))
                                self.screen.blit(t, rect)

        # --- Footer UI area (REPLACE EXISTING FOOTER BLOCK WITH THIS) ---
        footer_top = WIDTH
        footer_h = HEIGHT - WIDTH
        pygame.draw.rect(self.screen, (240, 240, 240), (0, footer_top, WIDTH, footer_h))

        y = footer_top + 8
        # Title
        title = self.font_title.render(f"Sudoku - {self.difficulty.capitalize()}", True, (20, 20, 20))
        self.screen.blit(title, (12, y))

        # Info line (time, mistakes, best, notes)
        y += 34
        elapsed = self.get_elapsed_time() if not self.solved else (self.last_finish_time or 0)
        info_parts = [f"Time: {elapsed}s", f"Mistakes: {self.mistakes}"]
        bt = get_best_time(self.difficulty)
        if bt is not None:
            info_parts.append(f"Best: {bt}s")
        info_parts.append(f"Notes: {'ON' if self.notes_mode else 'OFF'}")
        info_text = "   |   ".join(info_parts)
        self.screen.blit(self.font_ui.render(info_text, True, (55, 55, 55)), (12, y))

        # Controls (2 lines, compact)
        y += 24
        controls_lines = [
            "Arrows move | Click select | 1–9 place | 0/Del clear | N notes | C clear notes | Shift+H hint",
            "E/M/H new | R restart | S show solution | Q quit | (These actions will ask for confirmation)"
        ]
        for line in controls_lines:
            if y + 20 > HEIGHT - 8:  # stay within footer
                break
            self.screen.blit(self.font_ui.render(line, True, (55, 55, 55)), (12, y))
            y += 20
        # --- End Footer UI area ---
        
        if finalize:
            pygame.display.flip()

    def confirm_action_ui(self, message: str) -> bool:
        pygame = self.pygame

        # Translucent full-screen overlay
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))

        # Modal rectangle centered over the grid area
        w, h = int(WIDTH * 0.8), 180
        x, y = (WIDTH - w) // 2, (WIDTH - h) // 2
        rect = pygame.Rect(x, y, w, h)

        # Prepare message wrapping
        def wrap_lines(text, font, max_width):
            words = text.split()
            lines = []
            line = ""
            for word in words:
                test = (line + " " + word).strip()
                if font.render(test, True, (0,0,0)).get_width() <= max_width:
                    line = test
                else:
                    if line:
                        lines.append(line)
                    line = word
            if line:
                lines.append(line)
            return lines

        title = self.font_title.render("Confirm", True, (20, 20, 20))
        msg_lines = wrap_lines(message, self.font_ui, w - 40)

        # Buttons (clickable regions)
        btn_yes_text = "[Y] Yes"
        btn_no_text = "[N] No"
        yes_surface = self.font_ui.render(btn_yes_text, True, (255, 255, 255))
        no_surface = self.font_ui.render(btn_no_text, True, (255, 255, 255))

        yes_rect = pygame.Rect(x + w - 200, y + h - 50, 90, 32)
        no_rect  = pygame.Rect(x + w - 100, y + h - 50, 80, 32)

        while True:
            # Draw current game underneath, then overlay + modal
            if self.clock:
                self.clock.tick(60)

            self.draw(finalize=False)
            self.screen.blit(overlay, (0, 0))
            pygame.draw.rect(self.screen, (245, 245, 245), rect, border_radius=8)
            pygame.draw.rect(self.screen, (70, 70, 70), rect, 2, border_radius=8)

            # Title
            self.screen.blit(title, (x + 16, y + 12))

            # Message lines
            yy = y + 56
            for ln in msg_lines:
                t = self.font_ui.render(ln, True, (30, 30, 30))
                self.screen.blit(t, (x + 20, yy))
                yy += 24

            # Buttons
            pygame.draw.rect(self.screen, (50, 150, 80), yes_rect, border_radius=6)
            pygame.draw.rect(self.screen, (170, 60, 60), no_rect, border_radius=6)
            self.screen.blit(yes_surface, yes_surface.get_rect(center=yes_rect.center))
            self.screen.blit(no_surface,  no_surface.get_rect(center=no_rect.center))

            pygame.display.flip()

            # Modal event loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_y, pygame.K_RETURN):
                        return True
                    if event.key in (pygame.K_n, pygame.K_ESCAPE):
                        return False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if yes_rect.collidepoint(mx, my):
                        return True
                    if no_rect.collidepoint(mx, my):
                        return False

    
    def handle_mouse(self, pos: Tuple[int, int]):
        x, y = pos
        if x < WIDTH and y < WIDTH:
            c = x // CELL_SIZE
            r = y // CELL_SIZE
            self.selected = (r, c)

    def handle_key(self, key: int, mods: int):
        pygame = self.pygame
        if self.selected is None:
            return
        r, c = self.selected
        if key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
            if key == pygame.K_UP:
                r = (r - 1) % GRID
            elif key == pygame.K_DOWN:
                r = (r + 1) % GRID
            elif key == pygame.K_LEFT:
                c = (c - 1) % GRID
            elif key == pygame.K_RIGHT:
                c = (c + 1) % GRID
            self.selected = (r, c)
            return
        elif key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5,
                   pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
            val = key - pygame.K_0
            if self.notes_mode:
                self.board.add_note(r, c, val)
            else:
                # place and count mistakes if not matching solution
                if not self.board.fixed[r][c]:
                    prev = self.board.grid[r][c]
                    placed = self.board.place(r, c, val)
                    if placed:
                        if val != self.solution[r][c]:
                            self.mistakes += 1
                        # if user replaced a wrong num with right, we don't decrement mistakes (unlimited count)
                        # Check solved
                        if np.array_equal(np.array(self.board.grid), np.array(self.solution)):
                            self.solved = True
                            self.last_finish_time = self.get_elapsed_time()
                            # update best time
                            update_best_time(self.difficulty, self.last_finish_time)
        elif key in [pygame.K_0, pygame.K_DELETE, pygame.K_BACKSPACE]:
            if not self.board.fixed[r][c]:
                self.board.clear_cell(r, c)
        elif key == pygame.K_c:
            # clear notes in cell
            if not self.board.fixed[r][c] and self.board.grid[r][c] == 0:
                self.board.notes[r][c].clear()

    # Undo/redo via history
    def push_state(self):
        # keep small history (optional)
        pass

    # Simple hint: place a logically deducible single if available (naked/hidden)
    def take_hint(self):
        tmp = self.board.clone()
        solver = LogicSolver(allowed=[
            "naked_singles", "hidden_singles",
            "naked_pairs_triples", "hidden_pairs",
            "pointing_pairs", "claiming_pairs"
        ])
        # Try to find a single next move
        before = deepcopy(tmp.grid)
        if solver._naked_singles(tmp) or solver._hidden_singles(tmp):
            # find difference and apply first found
            for r in range(GRID):
                for c in range(GRID):
                    if self.board.grid[r][c] == 0 and tmp.grid[r][c] != 0:
                        # Apply correct digit (doesn't count as mistake)
                        self.board.place(r, c, tmp.grid[r][c])
                        return True
        return False

    def run(self):
        pygame = self.pygame
        self.init_pygame()
        running = True

        # UI toggles and extra keys outside selected cell context
        while running:
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse(pygame.mouse.get_pos())
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_n:
                        self.notes_mode = not self.notes_mode
                    elif event.key == pygame.K_q:
                        if self.confirm_action_ui("Quit the game? Unsaved progress will be lost."):
                            running = False

                    elif event.key == pygame.K_r:
                        if self.confirm_action_ui(f"Restart the current {self.difficulty.upper()} puzzle?"):
                            self.reset(self.difficulty)

                    elif event.key == pygame.K_e:
                        if self.confirm_action_ui("Start a NEW EASY puzzle?"):
                            self.reset("easy")

                    elif event.key == pygame.K_m:
                        if self.confirm_action_ui("Start a NEW MEDIUM puzzle?"):
                            self.reset("medium")

                    elif event.key == pygame.K_h:
                        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                            # Shift+H => hint (no confirmation)
                            self.take_hint()
                        else:
                            if self.confirm_action_ui("Start a NEW HARD puzzle?"):
                                self.reset("hard")

                    elif event.key == pygame.K_s:
                        if self.confirm_action_ui("Reveal the solution now? This ends the game."):
                            self.board.grid = deep_copy_grid(self.solution)
                            self.solved = True
                            self.last_finish_time = self.get_elapsed_time()
                    elif event.key == pygame.K_u:
                        # (placeholder for undo) - left for extension
                        pass
                    elif event.key == pygame.K_y:
                        # (placeholder for redo)
                        pass
                    else:
                        self.handle_key(event.key, event.mod)
            self.draw()

        pygame.quit()

# ----------------------------
# ML Docking: RL-friendly Environment
# ----------------------------

class SudokuEnv:
    """
    A minimal environment interface for ML/RL projects.

    - reset(difficulty) -> observation (9x9 numpy int32)
    - get_state() -> observation (9x9)
    - get_candidates() -> 9x9x10 boolean mask (index 1..9 for convenience)
    - step(action) where action = (r, c, v) with 0<=r,c<9 and 1<=v<=9:
        * Applies placement if cell not fixed.
        * reward:
            +1 if (r,c) was empty and v == solution[r][c]
            -1 if v != solution[r][c] (mistake increments)
            0 otherwise (illegal move or placing same correct value again)
        * done = True if solved
        * info = {"mistakes": int, "valid": bool, "was_empty": bool}
    """
    def __init__(self, difficulty: str = "medium"):
        self.difficulty = difficulty
        self.puzzle, self.solution = generate_puzzle(difficulty)
        self.board = SudokuBoard(self.puzzle, self.solution)
        self.mistakes = 0
        self.start_time = time.time()

    def reset(self, difficulty: Optional[str] = None) -> np.ndarray:
        if difficulty is not None:
            self.difficulty = difficulty
        self.puzzle, self.solution = generate_puzzle(self.difficulty)
        self.board = SudokuBoard(self.puzzle, self.solution)
        self.mistakes = 0
        self.start_time = time.time()
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return np.array(self.board.grid, dtype=np.int32)

    def get_candidates(self) -> np.ndarray:
        self.board.update_all_candidates()
        # Return a [9,9,10] boolean array; index 0 unused; cands[r][c][v] = 1 if v candidate
        cands = np.zeros((GRID, GRID, 10), dtype=np.bool_)
        for r in range(GRID):
            for c in range(GRID):
                for v in self.board.candidates[r][c]:
                    cands[r, c, v] = True
        return cands

    def step(self, action: Tuple[int, int, int]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        r, c, v = action
        reward = 0.0
        done = False
        valid = False
        was_empty = False

        if not (0 <= r < GRID and 0 <= c < GRID and 1 <= v <= 9):
            return self.get_state(), 0.0, False, {"mistakes": self.mistakes, "valid": False, "was_empty": False}

        if self.board.fixed[r][c]:
            # cannot change fixed cells
            return self.get_state(), 0.0, False, {"mistakes": self.mistakes, "valid": False, "was_empty": False}

        was_empty = (self.board.grid[r][c] == 0)
        if was_empty:
            placed = self.board.place(r, c, v)
            valid = placed
            if placed:
                if v == self.solution[r][c]:
                    reward = 1.0
                else:
                    reward = -1.0
                    self.mistakes += 1
        else:
            # Overwriting existing non-fixed input allowed; reward based on match
            placed = self.board.place(r, c, v)
            valid = placed
            if placed:
                if v == self.solution[r][c]:
                    reward = 0.0  # neutral for corrections (customize as needed)
                else:
                    reward = -1.0
                    self.mistakes += 1

        if np.array_equal(np.array(self.board.grid), np.array(self.solution)):
            done = True

        return self.get_state(), reward, done, {"mistakes": self.mistakes, "valid": valid, "was_empty": was_empty}

# ----------------------------
# Entry Point
# ----------------------------

def main():
    # Simple CLI: python sudoku_pygame.py [easy|medium|hard]
    diff = "medium"
    if len(sys.argv) >= 2 and sys.argv[1].lower() in ("easy", "medium", "hard"):
        diff = sys.argv[1].lower()
    try:
        game = Game(difficulty=diff)
        game.run()
    except ImportError as e:
        print("Pygame not installed. Install with: pip install pygame")
        raise

if __name__ == "__main__":
    main()

"""
USAGE & CONTROLS (UI):
- Run: python sudoku_pygame.py [easy|medium|hard]
- Click a cell to select it.
- Type 1..9 to place, 0/Del/Backspace to clear.
- Press N to toggle Notes (pencil marks). With Notes ON, typing 1..9 toggles that note in the selected cell.
- Press C to clear notes in the selected cell.
- Press E/M/H to start a new puzzle for Easy/Medium/Hard.
- Press R to restart a new puzzle at the current difficulty.
- Press Shift+H to take a logical hint (if available).
- Press S to fill the full solution (ends the game).
- Mistake Counter: increases whenever you place a digit that doesn't match the solution.
- Timer and Best Time are shown at bottom.

ML DOCKING:
- Import SudokuEnv from this file in Jupyter:
    from sudoku_pygame import SudokuEnv
    env = SudokuEnv('hard')
    obs = env.reset()
    # Take an action (r,c,v)
    obs, reward, done, info = env.step((0,0,5))
- Access candidates with env.get_candidates()
- The environment supplies:
    * Unique solution puzzles
    * Solvable by logical strategies (no guessing)
    * Mistake count in info['mistakes'] for reinforcement signals
"""
