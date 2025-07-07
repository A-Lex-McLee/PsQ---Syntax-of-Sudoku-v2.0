#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SUDOKU GRIDs 

The Syntax of Sudoku Grids, Part A: Local Infrastructure 
 
Created in Spring 2020, revised 2024/25

@author: alexanderpfaff

"""

from __future__ import annotations
from itertools import permutations, combinations, product
from typing import Optional, TypeVar, List, Tuple, Set, Dict, Collection, Iterator, Union
from random import randint, shuffle, choice, sample
from math import factorial, sqrt
from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Callable
#from enum import Enum
#import collections.abc.Collection
#import FunX  as fx
import numpy as np

# from tkinter import END as tk_END
# from tkinter import Text as tk_Text

from tqdm import tqdm
#import sqlite3


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 


V = TypeVar('V', str, int) 





@dataclass
class Cell: 
    """
    Minimal unit of a grid structure; contains the actual value and 
    the grid coordinates: 
        run -- running number 
        row -- row number 
        col -- column number 
        box -- box number 
        box_row -- wrapper structure comprising rows of box size 
        box_col -- wrapper structure comprising columns of box size 
        
    """
    run: int 
    row: int
    col: int 
    box_row: int 
    box_col: int 
    box: int 
    val: V = 0
    
    
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

# Functional parameters  ==> to be passed in as arguments to certain
#                             methods of the Grid class
#                             (e.g. gridOut, )
#                        ==> they return the value of the eponymous attribute
#                             of some cell

row: int     = lambda c : c.row
col: int     = lambda c : c.col
box: int     = lambda c : c.box
box_row: int = lambda c : c.box_row
box_col: int = lambda c : c.box_col



# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

class Grid:
    """ 
        Provides a (baseNumber x baseNumber) X (baseNumber x baseNumber) Sudoku Grid,
        default setting: baseNumber = 3 ==> classical 9 X 9 Sdoku grid (recommended!);
        The constructor generates a base grid (baseGrid), which is effectively
        a coordinate system of Cell objects with the coordinates:
           -- run: running number (1 to baseNumber**4) from top-left to bottom-right,
           -- col: column number (1 to baseNumber**2) from left to right,
           -- row: row number (1 to baseNumber**2) top-down,
           -- box: box number (1 to baseNumber**2) from top-left to bottom-right,
           -- val: the actual value in a valid Sudoku grid; default setting: 0
 
           -- box_col: number of column with box-width (1 to baseNumber), left to right,
           -- box_row: number of row with box-width (1 to baseNumber), top-down.
           
           An illustration; the following grid: 
               
                *************************************
                * 2 | 6 | 4 * 3 | 8 | 9 * 5 | 1 | 7 *
                *-----------*-----------*-----------*
                * 5 | 1 | 7 * 6 | 4 | 2 * 9 | 8 | 3 *
                *-----------*-----------*-----------*
                * 3 | 8 | 9 * 7 | 5 | 1 * 4 | 6 | 2 *
                *************************************
                * 4 | 2 | 6 * 5 | 1 | 7 * 3 | 9 | 8 *
                *-----------*-----------*-----------*
                * 9 | 3 | 8 * 2 | 6 | 4 * 1 | 7 | 5 *
                *-----------*-----------*-----------*
                * 1 | 7 | 5 * 8 | 9 | 3 * 6 | 2 | 4 *
                *************************************
                * 6 | 4 | 2 * 1 | 3 | 8 * 7 | 5 | 9 *
                *-----------*-----------*-----------*
                * 7 | 5 | 3 * 9 | 2 | 6 * 8 | 4 | 1 *
                *-----------*-----------*-----------*
                * 8 | 9 | 1 * 4 | 7 | 5 * 2 | 3 | 6 *
                *************************************
                
               has a.o. the following coordinates (matrix notation): 
                   
                row 1:      [2, 6, 4, 3, 8, 9, 5, 1, 7]  
                
                col 2:      [6, 1, 8, 2, 3, 7, 4, 5, 9]T
                
                box 3:      [5, 1, 7,
                             9, 8, 3,
                             4, 6, 2]
                
                box_row 3: [[6, 4, 2, 1, 3, 8, 7, 5, 9]
                            [7, 5, 3, 9, 2, 6, 8, 4, 1]
                            [8, 9, 1, 4, 7, 5, 2, 3, 6]]
                
                col_row 2: [[3, 6, 7, 5, 2, 8, 1, 9, 4]T
                            [8, 4, 5, 1, 6, 9, 3, 2, 7]T
                            [9, 2, 1, 7, 4, 3, 8, 6, 5]T]
                    
                         
           The constructor initializes a grid with coordinates calculated from {baseNumer}
           with values set to zero. 
           The class provides a method to initialize the values via insertion 
           where the inserted object must be a sequential container object (tuple, list, string)
           containing legal value types, viz. int or str. 
           
           Further functionalities are included, such as methods for
             -- visualizing (= printing out) the grid,
             -- generating random grids, 
             -- geometric manipulation of the grid,
             -- permuting the grid coordinates,
                  notably, for generating the full permutation series of a given grid 
                  resulting in a collection of {baseNumber}!^8 * 2 grid permutations
                  (for baseNumber=3 <=> classical Sodoku grid, this means 3.359.232 grids!)
             -- alphabetizing the grid <==> de-/encoding the grid alphabetically.              
    """
    
    """ Default setting classical 9 X 9 Sudoku at the class level -> class methods"""
    _BASE_NUMBER: int = 3
    _DIMENSION: int = 9
    _SIZE: int = 81

    def __init__(self, baseNumber: int = 3):
        self.BASE_NUMBER: int = baseNumber
        self.DIMENSION: int = baseNumber**2
        self.SIZE: int = baseNumber**4
        self.zeroSeq: Tuple[0] = tuple(0 
                                       for i in range(self.SIZE))        
        """ Algorithm to calculate the grid coordinates. """
        self.baseGrid: Tuple[Cell] = tuple(
            Cell(run=(self.DIMENSION)*out + self.BASE_NUMBER*mid + inn + 1,
                  row=out + 1,
                  col=self.BASE_NUMBER*mid + inn + 1,
                  box_row=out//self.BASE_NUMBER + 1,
                  box_col=mid + 1,
                  box=(out//self.BASE_NUMBER)*self.BASE_NUMBER + mid + 1) 
             for out in range(self.DIMENSION)
             for mid in range(self.BASE_NUMBER)
             for inn in range(self.BASE_NUMBER) 
            )
        
    def __str__(self) -> str:
        return f'Sudoku_Grid:  ({self.BASE_NUMBER} x {self.BASE_NUMBER})  X  ({self.BASE_NUMBER} x {self.BASE_NUMBER})'

    def __repr__(self) -> str:
        return "<class 'Grid'>"

     
# * * * * * * * * * * * * * *  INVENTORY  * * * * * * * * * * * * * * * * * * * 

    """ 0.    INSERT a GRID """

    def insert(self, seq: Tuple[V]) -> None:
        """
        inserts a sequence of characters (numbers, letters ...) and 
        translates it into a grid structure.
        
        @requires seq is an ordered iterable (list, tuple, str)!  
        @requires len(seq) == self.BASE_NUMBER**4
        """
        if len(seq) != len(self.baseGrid):
            raise ValueError(f"The sequence submitted does not contain the required number of cells: {len(self.baseGrid)}")
        for i in range(len(self.baseGrid)):
            self.baseGrid[i].val = seq[i]

    def setZero(self) -> None:
        """
        Empties the grid by setting all values to zero
        """
        self.insert(self.zeroSeq)


    """ A.    VISUAL """

    def showFrame(self) -> None:
        """prints the current grid (= baseGrid) as a sequence of coordinates """
        for cell in self.baseGrid:
            outStr: str = f'run: {cell.run:2};   '  \
                + f'row: {cell.row};   '            \
                + f'col: {cell.col};   '            \
                + f'box: {cell.box};   '            \
                + f'box_col: {cell.box_col};   '    \
                + f'box_row: {cell.box_row};   '    \
                + f'value: {cell.val} '    
            print(outStr)

    def showGrid(self) -> None:      
        """prints out the current grid as Sudoku grid """
        grdLen: int = (self.DIMENSION * 2 + self.BASE_NUMBER*2 + 1)
        print()
        print("-" * grdLen)
        for i in range(self.DIMENSION):
            print("|", end=" ")
            for u in range(self.DIMENSION):
                print(self.baseGrid[self.DIMENSION*i+u].val, end=" ")
                if (self.DIMENSION*i+u) % self.BASE_NUMBER == self.BASE_NUMBER - 1: #2:
                    print("|", end=" ")
            print()
            if (i+1) % self.BASE_NUMBER == 0:
                print("-" * grdLen)



    """ B.    VALUES / Value TUPLES by COORDINATE """
    
    def getRun(self, r: int, c: int) -> int:
        """ calculates the running number from row number r and column number c 
            
            @requires: 1 <= r,c <= 9
        """
        return (r-1) * self.DIMENSION + c
        
    def getCell_fromRun(self, r: int) -> Cell:
        """ returns the cell with running number r """ 
        if r < 1 or r > self.BASE_NUMBER**4:
            raise ValueError(f"Invalid running number (choose 1 - {self.BASE_NUMBER**4})")
        return deepcopy(self.baseGrid[r-1])
        
        
    def getRelativeBox(self, center: int) -> np.ndarray:
        """
            returns a BASE_NUMBER X BASE_NUMBER value box 
            with running number {center} as the center cell
            (-> column/row overflow) 
        

        Parameters
        ----------
        center : int
            the center cell of the prospective BASE_NUMBER X BASE_NUMBER box

        Returns
        -------
        TYPE : numpy.ndarray (shape: BASE_NUMBER, BASE_NUMBER)
             'relative' box, i.e. not necessarily a valid grid-box.
        """
        out: List[V] = []
        for relativeRow in range(self.BASE_NUMBER):
            rTmp = ( ( (center-1) - self.DIMENSION) // self.DIMENSION + relativeRow)
            _row: int = rTmp % self.DIMENSION + 1

            for relativeCol in range(self.BASE_NUMBER):                
                cTmp = ((center - 2) % self.DIMENSION + relativeCol) 
                _col: int = cTmp % self.DIMENSION + 1
                
                _run: int = self.getRun(_row, _col)
                cell: Cell = self.getCell_fromRun(_run)
                out.append(cell.val)
                    
        boxAsArray: np.ndarray = np.asarray(out)
        return boxAsArray.reshape([self.BASE_NUMBER, self.BASE_NUMBER]) 



    def gridRow(self, r: int) -> Tuple[int]:
        """ returns the values in row r as tuple """ 
        if r < 1 or r > self.DIMENSION:
            raise ValueError(f"Invalid row number (choose 1 - {self.DIMENSION})")
        return tuple(cell.val
                for cell in self.baseGrid
                if cell.row == r)

    def gridCol(self, c: int) -> Tuple[int]:
        """ returns the values in column c as tuple """ 
        if c < 1 or c > self.DIMENSION:
            raise ValueError(f"Invalid column number (choose 1 - {self.DIMENSION})")
        return tuple(cell.val
                for cell in self.baseGrid
                if cell.col == c)

    def gridBox(self, b: int) -> Tuple[int]:
        """ returns the values in box b as tuple """ 
        if b < 1 or b > self.DIMENSION:
            raise ValueError(f"Invalid box number (choose 1 - {self.DIMENSION})")
        return tuple(cell.val
                for cell in self.baseGrid
                if cell.box == b)
        
    def gridOut(self, coord: Callable[Cell, int] = row) -> Tuple[int]:
        """ 
        returns the <entire> current grid as a tuple; 
        legal arguments: 
            -- row (default): sorted row-wise (= by running number)
                                      ==> de facto IS the grid,
            -- col:           sorted column-wise (= top-down),
            -- box:           sorted box-wise (box-internally: row-wise)
        
        >> NB: if row is a valid grid,
        >>      -->  col produces a valid grid 
        >>      -->  box does not (normally!?) produce a valid grid
        """
        out: list[int] = []
        for i in range(1, self.DIMENSION + 1):
            for cell in self.baseGrid:
                if coord(cell) == i:
                    out.append(cell.val)   
        return tuple(out)


    def grid_toArray(self, shape=(81,)) -> np.ndarray:
        """ 
        returns the  current grid as a numpy array; 

        """

        return np.array(self.gridOut()).reshape(shape)





    """ C.    RANDOM GRID GENERATION 
          ==> with subsequent insertion 
                (i.e. the current grid will be overwritten; 
                 for grid generation without insertion, 
                 see external functions  @generateRndGrid_cell,
                                         @generateRndGrid_deep  )
    """
    
    def generateGrid_flat(self) -> None: 
        """ ==> calls external function  @generateRndGrid_cell
 
            Generates a grid assignment with a valid digit distribution at random,
            and inserts it into the current Grid object.
            Sloppy version, does not use backtracking  
            --> can result in a quick output (or not), but it is possible
            that it does not produce any output at all because the procedure 
            >> runs out of options << as it were. 
            (for posssible fix, @see generateRndGrid_cell)        
        """
        sequence: Tuple[int] = generateRndGrid_cell(self.BASE_NUMBER)
        self.insert(sequence)

    def generateGrid_deep(self) -> None: 
        """ ==> calls external function  @generateRndGrid_deep
        
            Generates a grid assignment with a valid digit distribution at random,
            and inserts it into the current Grid object.
            Systematic semi-recursive version that implements backtracking 
            --> will always return a valid solution;
            but may take a bit more time to do so. 
        """
        sequence: Tuple[int] = generateRndGrid_deep(self.BASE_NUMBER)
        self.insert(sequence)




    """ D.    CHECK the GRID   """

    # looks good so far, but re-check
    def gridCheckZero(self) -> bool:
        """
        checks whether the current grid contains no more than one occurrence of every value 
        in every row, every column and every box, respectively. 
        Notably, it returns true even if the respective value occurs zero times 
        in any row, column or box. 
        """
        values: list[int] = list(range(1, self.DIMENSION+1))
        for val in values:
            
            col_null = self.gridCol(val).count(0)
            row_null = self.gridRow(val).count(0)
            box_null = self.gridBox(val).count(0)
            
            col_set = set(self.gridCol(val)) - {0}
            row_set = set(self.gridRow(val)) - {0}
            box_set = set(self.gridBox(val)) - {0}
            
            if len(col_set) != self.DIMENSION - col_null:
                return False
            if len(row_set) != self.DIMENSION - row_null:
                return False
            if len(box_set) != self.DIMENSION - box_null:
                return False
            
        return True           
            
    def gridCheck(self) -> bool:
        """
        checks whether the current grid is valid;
        in a a valid grid, every row, every colum and every box
        contains every character EXACTLY ONCE each 
        """
        values: Set[V] = set(self.gridOut())
        if len(values) != self.DIMENSION:
          #  raise ValueError(f"Number of values not equal to {self.DIMENSION}")
            return False
        for val in values:
            for coord in range(1,self.DIMENSION+1):
                if self.gridBox(coord).count(val) != 1:
                    return False
                if self.gridCol(coord).count(val) != 1:
                    return False
                if self.gridRow(coord).count(val) != 1:
                    return False
        return True           


    # double-check if valid
    def gridCheck_count(self) -> bool:
        """
        checks whether the current grid is valid (classical 9 x 9 only);
        in a a valid grid, every value occurs exactly 9 times, and
        every row, every colum and every box
        contains every character EXACTLY ONCE each
        """
        _grid =  self.gridOut()
        values: Set[int] = set(_grid)
        if len(values) != self.DIMENSION:
            raise ValueError(f"Number of values not equal to {self.DIMENSION}")
        for val in values:
            if _grid.count(val) != 9:
                return False
            if set(self.gridBox(val)) != values:
                return False
            if set(self.gridCol(val)) != values:
                return False
            if set(self.gridRow(val)) != values:
                return False
        return True           


    def gridCheck_numpy(self) -> bool:
        """
        Fast NumPy-based check for a 9x9 Sudoku grid:
        - Checks that each digit 1–9 appears exactly 9 times globally
        - Ensures each row, column, and box has unique values 1–9
        """
        
        _grid =  np.array(self.gridOut()).reshape(9, 9)
    
        unique, counts = np.unique(_grid, return_counts=True)   # Global count check
        if len(unique) != 9 or not np.all(counts == 9):
            return False
    
        expected = set(range(1, 10))
    
        for i in range(9):                                      # Row and column uniqueness
            if set(_grid[i, :]) != expected:
                return False
            if set(_grid[:, i]) != expected:
                return False
        
        for r in range(0, 9, 3):                                # Box uniqueness
            for c in range(0, 9, 3):
                box = _grid[r:r+3, c:c+3].flatten()
                if set(box) != expected:
                    return False
        return True




    
    @classmethod
    def checkGrid(cls, grid: np.ndarray) -> bool:
        # Ensure the grid is a 1D array of size 81
        if grid.size != cls._DIMENSION * cls._DIMENSION:
            print(f"Invalid grid size: Expected {cls._DIMENSION * cls._DIMENSION} elements.")
            return False
        
        # Reshape the grid into a 9x9 2D array
        _grid = np.array(grid).reshape(cls._DIMENSION, cls._DIMENSION)
        
        # Ensure each number from 1 to 9 appears exactly once in every row, column, and 3x3 subgrid
        expected_set = set(range(1, cls._DIMENSION + 1))  # Set of valid Sudoku values (1-9)
        
        # Check row and column uniqueness
        for i in range(cls._DIMENSION):
            if set(_grid[i, :]) != expected_set:  # Check the i-th row
                return False
            if set(_grid[:, i]) != expected_set:  # Check the i-th column
                return False
        
        # Check each 3x3 subgrid
        for r in range(0, cls._DIMENSION, cls._BASE_NUMBER):
            for c in range(0, cls._DIMENSION, cls._BASE_NUMBER):
                box = _grid[r:r+cls._BASE_NUMBER, c:c+cls._BASE_NUMBER].flatten()  # Extract 3x3 subgrid
                if set(box) != expected_set:  # Ensure the 3x3 subgrid has all digits 1-9
                    return False
        
        return True





    """ E.    SYMMETRY / GEOMETRY"""

    def rotate(self) -> None:
        """ 
            rotate the current grid 90°clockwise
            Procedure: insert reversed column values into the corresponding row.
        """
        rotation: Tuple[int] = [reverseCol_toRow
                                for coordRange in range(1,self.DIMENSION + 1)
                                for reverseCol_toRow in [self.gridCol(coordRange)[-reverseIdx]
                                                         for reverseIdx in range(1,self.DIMENSION + 1)]]
        self.insert(rotation)


    def diaflect(self) -> None:
        """ 
            reflects the grid diagonally along top-left <=> bottom-right axis;
            (= transpose in matrix terms).
        """
        self.insert(self.gridOut(col))
        

    def seqOverflow(self, coord: int = row) -> Tuple[int]:
        """
            Sequential overflow;
            moves running number += 1, with final+1 to initial position
        """
        movedSeq: list[int] = list(self.gridOut(coord))
        movedSeq.insert(0, movedSeq.pop(self.BASE_NUMBER**4-1))
        return tuple(movedSeq)
    
    
    
    def scanGrid(self) -> Tuple[Tuple[int, np.ndarray, int]]:
        """
        Scans the current grid box-wise (BASE_NUMBER X BASE_NUMBER) without overflow; 
        compares the sum of the current box to the currently legal sum of boxes ('kleiner Gauss')

        Returns
        -------
        scannedGrid : TYPE
        """
        legalBoxSum: int = (self.DIMENSION * (self.DIMENSION +1)) // 2
        scannedGrid: List[Tuple[int, np.ndarray, int]] = []
        
        _runs = np.arange(1, self.DIMENSION**2 + 1).reshape(self.DIMENSION, self.DIMENSION)
        _runs = _runs[1 : self.DIMENSION - (self.BASE_NUMBER // 2),     # array of running numbers
                      1 : self.DIMENSION - (self.BASE_NUMBER // 2)]
        
        for _run in _runs.flatten():
            _box = self.getRelativeBox(_run) 
            currentBoxSum: int = _box.sum()  
            out: Tuple[int, np.ndarray, int] = (_run, _box, legalBoxSum-currentBoxSum)
            scannedGrid.append(out)
        return tuple(scannedGrid)



    def scanGridOverflow(self) -> Tuple[Tuple[int, np.ndarray, int]]:
        """
        Scans the current grid box-wise (BASE_NUMBER X BASE_NUMBER) along the row-axis
        WITH row overflow; 
        compares the sum of the current box to the currently legal sum of boxes ('kleiner Gauss')

        Returns
        -------
        scannedGrid :    Tuple[Tuple[int, np.ndarray, int]]
            DESCRIPTION:  Tuple<running number, relative Bos, sum difference>

        """
        legalBoxSum: int = (self.DIMENSION * (self.DIMENSION +1)) // 2
        scannedGrid: List[Tuple[int, np.ndarray, int]] = []
        
        for _run in range(1, self.DIMENSION**2 + 1):
            _box = self.getRelativeBox(_run) 
            currentBoxSum: int = sum(sum(_box))                
            out: Tuple[int, np.ndarray, int] = (_run, _box, legalBoxSum-currentBoxSum)
            scannedGrid.append(out)            
        return tuple(scannedGrid)




    """ F.    PERMUTE (parts of) the GRID """

    def permuteCols_numpy(self, grid: np.ndarray = None) -> list[np.ndarray]:
        """
        Given a 9x9 Sudoku grid, returns a list of all 1296 column permutations
        that preserve Sudoku structure using NumPy operations.
        
        Parameters:
        - grid: np.ndarray of shape (9, 9)
        
        Returns:
        - list of np.ndarray, each of shape (9, 9)
        """
        if grid is None:
            grid = np.array(self.gridOut())
        grid.reshape(9, 9)  
        
        all_permuted_grids = []
        
        # All permutations of [0, 1, 2]
        group_perms = list(permutations([0, 1, 2]))
    
        # Step 1: Permute the 3 column *blocks*
        for box_perm in permutations([0, 1, 2]):
            box_indices = [i*3 for i in box_perm]  # starting indices of each col block
            box_cols = [list(range(i, i+3)) for i in box_indices]
            
            # Step 2: Permute columns within each block
            for p1 in group_perms:
                for p2 in group_perms:
                    for p3 in group_perms:
                        full_col_order = [box_cols[0][i] for i in p1] + \
                                         [box_cols[1][i] for i in p2] + \
                                         [box_cols[2][i] for i in p3]
                        
                        permuted_grid = grid[:, full_col_order]
                        all_permuted_grids.append(permuted_grid.copy())
        
        return all_permuted_grids


    def permuteRows_numpy(self, grid: np.ndarray = None) -> list[np.ndarray]:
        """
        Given a 9x9 Sudoku grid, returns a list of all 1296 row permutations
        that preserve Sudoku structure using NumPy operations.
        
        Parameters:
        - grid: np.ndarray of shape (9, 9)
        
        Returns:
        - list of np.ndarray, each of shape (9, 9)
        """
        if grid is None:
            grid = np.array(self.gridOut())
        grid.reshape(9, 9)  
        
        all_permuted_grids = []
        
        # All permutations of [0, 1, 2]
        group_perms = list(permutations([0, 1, 2]))
    
        # Step 1: Permute the 3 rows *blocks*
        for box_perm in permutations([0, 1, 2]):
            box_indices = [i*3 for i in box_perm]  # starting indices of each col block
            box_rows = [list(range(i, i+3)) for i in box_indices]
            
            # Step 2: Permute rows within each block
            for p1 in group_perms:
                for p2 in group_perms:
                    for p3 in group_perms:
                        full_row_order = [box_rows[0][i] for i in p1] + \
                                         [box_rows[1][i] for i in p2] + \
                                         [box_rows[2][i] for i in p3]
                        
                        permuted_grid = grid[full_row_order, :]
                        all_permuted_grids.append(permuted_grid.copy())
        
        return all_permuted_grids


 #   def permuteGrids(self, grid: np.ndarray) -> xxx   # to class method
    def permuteGrids(self, toCollection: bool = True, 
                              toDB: bool = False, db_name: str = None) -> GridCollection: 
        """
        Returns a list of all 1296 (column) X 1296 (row) = 1679616 permutations,
        for a given input grid (self.gridOut() as default).
        
        Parameters:
        - grid: np.ndarray of shape (9, 9)
        
        Returns:
        - list of np.ndarray, each of shape (9, 9)
        """
        from PsQ_GridCollection import GridCollection

        grid = np.array(self.gridOut()).reshape(9, 9)
        all_permutations = []
        
        # start with column permutations
        column_permutations: list[np.ndarray] = self.permuteCols_numpy(grid)
        for colPerm in tqdm(column_permutations): 
            all_permutations += self.permuteRows_numpy(colPerm)
            
        out_grids: np.ndarray = np.array(all_permutations).reshape(-1, self.SIZE)
        
        if toDB:
            pass
        
        if toCollection:
            return GridCollection(out_grids)
        else:
            return out_grids




#    def fullPermutation(self, grid: np.ndarray)   # for class method!"!!
    def fullPermutation(self, toCollection: bool = True, 
                              toDB: bool = False, db_name: str = None) -> GridCollection: 
        """
        Returns a list of all 1296 (column) X 1296 (row) permutations X 2
        (adding mirror symmetry) ==> = 1679616 X 2 = 3359232 grids in total
        for a given input grid (self.gridOut() as default).
        
        Parameters:
        - grid: np.ndarray of shape (9, 9)
        
        Returns:
        - list of np.ndarray, each of shape (9, 9)
        """
        from PsQ_GridCollection import GridCollection
        # grid = np.array(self.gridOut()).reshape(self.DIMENSION, self.DIMENSION)        
        all_permutations: list[np.ndarray] = []

        for perm in tqdm(self.permuteGrids(False, False)):            
            all_permutations.append(perm)
            all_permutations.append(perm.T.copy())
        
        out_grids: np.ndarray = np.array(all_permutations).reshape(-1, self.SIZE)
        
        if toDB:
            GridCollection.save_toSQL(out_grids, db_name)
        
        if toCollection:
            return GridCollection(out_grids)
        else:
            return out_grids






# ****************************   special for gui   **************************************
    def _permuteCoord(self, pos: int = 1) -> Tuple[Tuple[int]]:
        """
        _aux_method@permutePos
        
        produces permutations of coordinates within the given positional 
        dimension; possible values for pos: 1 - self.BASE_NUMBER
        """
        coordinateSpan: List[int] = list(range(1,self.BASE_NUMBER + 1))
        lowestIndex: int = (pos-1) * self.BASE_NUMBER
        coordinateRange: list[int] = [lowestIndex + i
                           for i in coordinateSpan]
        return tuple(permutations(coordinateRange))

    def _permuteDiff(self, pos: int = 1):
        """
        _aux_method@permutePos
        
        calculates the differences between old and new positions, and produces
        permutations of the respective differences
        """
        out: List[Tuple[int]] = []
        per_mutations: Tuple[Tuple[int]] = self._permuteCoord(pos=pos)
        base: Tuple[int] = per_mutations[0]
        for perm in per_mutations:
            out.append(tuple(perm[i] - base[i]
                             for i in range(len(perm))))
        return out

    def permutePos(self, pos: int = 1, grdCoord: Callable[Cell, int] = col) -> Tuple[Tuple[int]]:
        """
        creates the self.BASE_NUMBER! permutations of grdCoord in the curent grid
        within the positional dimension pos


        Parameters
        ----------
        pos : int, optional
                position of the higher unit of the {grdCoord} coordinate, i.e.
                    * for row & col ==> box_row & box_col, respectively;
                        thus legal values are 1, 2 .. BASE_NUMBER        
                    * for box_row & box_col ==> grid.DIMENSION;
                        thus the only legal value is 1        
        grdCoord : Callable[Cell, int], optional
                    Functional parameter to specify the coordinate to be permuted:
                    row, col (= default), box_row, box_col. 

        Returns :  
        -------
        TYPE:  Tuple[Tuple[int]]
            contains the grid permutations for {grd} as tuples.
        """
        out = []
        per_mutations = self._permuteCoord(pos=pos)
        permute_Diffs = self._permuteDiff(pos=pos)
        base = per_mutations[0]
        for perm in permute_Diffs:
            tempGrid = list(self.gridOut()) 
            for cells in self.baseGrid:
                for i in range(self.BASE_NUMBER):
                    if grdCoord(cells) == base[i]:
                        if grdCoord == col:
                            diff = perm[i] * self.BASE_NUMBER**0
                        if grdCoord == box_col:
                            diff = perm[i] * self.BASE_NUMBER**1
                        if grdCoord == row:
                            diff = perm[i] * self.BASE_NUMBER**2
                        if grdCoord == box_row:
                            diff = perm[i] * self.BASE_NUMBER**3
                        ix = cells.run + diff
                        tempGrid[ix-1] = cells.val
            out.append(tuple(tempGrid))
        return tuple(out)

# ******************************************************************************





    def intStr_toGrid(self, grid_str: str, insert: bool = True) -> np.ndarray:
        """Converts a grid string to an np.ndarray of integers"""
        if insert:
            self.insert(grid_str)
        else:
            return np.fromiter((int(ch) 
                                for ch in grid_str), dtype=int)


    def intGrid_toString(self, grid: np.ndarray = None) -> str:
        if grid is None:
            grid: np.ndarray = np.array(self.gridOut())
        grid = grid.reshape(self.SIZE, )
        gridString: str = "".join(map(str, grid))
        return gridString







    def recode(self, re_code: Collection[V], grid: np.ndarray = None, 
               insert: bool = True) -> Optional[np.ndarray]:
        """
        'translates' a given grid via the key {encode}, which must be a valid 
        Sudoku sequence of size {DIMENSION} (= 9 for classical Sudoku). 
        The key substitutes the first row and forms the basis for an encoder 
        with gridRow_1[n] -> encode[n]; for example:
            
            given grid row 1:   [1, 2, 3, 4, 5, 6, 7, 8, 9]
            re_code:            [9, 8, 7, 6, 5, 4, 3, 2, 1]
            
          ==>  recoder:         1 -> 9
                                2 -> 8
                                3 -> 7
                                4 -> 6
                                5 -> 5
                                6 -> 4
                                7 -> 3
                                8 -> 2
                                9 -> 1
        
        Finally, the grid values are substituted by this encoder.
        Works from int to int, int to str, str to str, and str to int. 

        Parameters
        ----------
        encode : Collection[V]
            DESCRIPTION.
        gridTuple : Optional[Tuple[V]], optional
            DESCRIPTION. The default is None.
        insert : bool, optional (the default is True).
            DESCRIPTION:  If True, the re-coded grid will be inserted into this grid.

        Returns
        -------
        Optional[Tuple[V]]
            DESCRIPTION.

        """
        if grid is None:
            grid: np.ndarray = np.array(self.gridOut())
            
        assert len(re_code) == self.DIMENSION

        recoder: Dict[V, V] = {grid.flat[idx] : re_code[idx]
                               for idx in range(self.DIMENSION)}
        lookup = np.arange(10)
                
        for old_val, new_val in recoder.items():
            lookup[old_val] = new_val
        newGrid: np.ndarray = lookup[grid]
        if insert:
            self.insert(newGrid)
        else:
            return newGrid






    def recodeAlphaNum(self, encode: Collection[V], gridTuple: Optional[Tuple[V]] = None,
                  insert: bool = True) -> Optional[Tuple[V]]:
        """
        'translates' a given grid via the key {encode}, which must be a valid 
        Sudoku sequence of size {DIMENSION} (= 9 for classical Sudoku). 
        The key substitutes the first row and forms the basis for an encoder 
        with gridRow_1[n] -> encode[n]; for example:
            
            given grid row 1:   [1, 2, 3, 4, 5, 6, 7, 8, 9]
            encode:             [A, B, C, D, E, F, G, H, I]
            
          ==>  newEncoder:      1 -> A
                                2 -> B
                                3 -> C
                                4 -> D
                                5 -> E
                                6 -> F
                                7 -> G
                                8 -> H
                                9 -> I
        
        Finally, the grid values are substituted by this encoder.
        Works from int to int, int to str, str to str, and str to int. 

        Parameters
        ----------
        encode : Collection[V]
            DESCRIPTION.
        gridTuple : Optional[Tuple[V]], optional
            DESCRIPTION. The default is None.
        insert : bool, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        Optional[Tuple[V]]
            DESCRIPTION.

        """
        if gridTuple is None:
            gridTuple = self.gridOut()
        assert len(set(encode)) == self.DIMENSION, \
        f"{self.DIMENSION} keys required; {len(set(encode))} were given!"
        assert not (0 in gridTuple), "Grid not initialized!"
        assert issubclass(type(gridTuple[0]), (str, int)) 
        sourceType: type = type(gridTuple[0])        
        assert [type(x) is sourceType and str(x).isalnum() 
                for x in gridTuple]
        newEncoder: Dict[V, V] = {gridTuple[idx] : encode[idx] 
                                  for idx in range(len(encode))}
        out: List[V] = []
        for key in gridTuple:
            out.append(newEncoder[key])
        if not insert:
            return tuple(out)
        self.insert(out)






# 9**81 = 
    # 196627050475552913618075908526912116283103450944214766927315415537966391196809 (len 78)

# faculty(81) / (faculty(9)**9) = 
    # 53130688706387570345024083447116905297676780137518287775972350551392256        (len 71)


    


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

def generateRndGrid_cell(baseNumber: int, attempts: int = 1000, test_runs: int = 4000, 
                        prnt: bool = False) -> Tuple[int]:
    """
    Generates a grid assignment with a valid digit distribution at random.
    Sloppy / brute force version, does not use backtracking  
    --> can result in a quick output (or not), but it is possible
    that it does not produce any output at all because the procedure 
    >> runs out of options << as it were. Posssible fix:
        -- increase the {test_runs} parameter : how many grid permutations 
                                                are tested in one go?
        -- increase the {attempts} parameter  : how many permutation series
                                                are run?

    Parameters
    ----------
    attempts : int,  optional
        DESCRIPTION: Number of times the aux method is called; 
                     the default is 1000.
    test_runs : int, optional
        DESCRIPTION: Number of times the aux function may fail at inserting 
                     a random number into the grid.
                     The default is 4000.
    prnt : bool, optional:  if True: the solution will be printed on screen
        DESCRIPTION. The default is True.

    Returns
    -------
    None
    """
    
    for i in range(attempts):
        testGrid: Grid = Grid(baseNumber)
        if _rnd_single(testGrid, test_runs = test_runs):
            if prnt == True:
                print(i * test_runs, " attempts!")
                testGrid.showGrid()
            return testGrid.gridOut()
    return tuple()

def _rnd_single(testGrid: Grid, test_runs: int) -> bool:
    """ @generateRndGrid_cell_aux """
    run: int = 0
    while run < (testGrid.BASE_NUMBER**4):
        row: int = testGrid.baseGrid[run].row
        col: int = testGrid.baseGrid[run].col
        box: int = testGrid.baseGrid[run].box
        val: int = randint(1, testGrid.DIMENSION)
        if val in testGrid.gridRow(row) or \
           val in testGrid.gridCol(col) or \
           val in testGrid.gridBox(box):
            test_runs -= 1                
        else:
            testGrid.baseGrid[run].val = val
            run += 1
        if test_runs < 0:
            return False
    return True

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *             

def generateRndGrid_deep(baseNumber: int) -> Tuple[int]:
    """
        Generates a grid assignment with a valid digit distribution at random,

        Systematic semi-recursive version that implements backtracking 
        --> will always return a valid solution;
        but may take a bit more time to do so. 
    
    Parameters
    ----------
    baseNumber : int
        BASE NUMBER of grid.

    Returns
    -------
    Tuple[int]
        valid Sudoku Grid.
    """
    sequence: Tuple[int] = LinkedSequence(baseNumber)
    return sequence.sequence

class Node:
    def __init__(self, baseNumber: int, parent: Node, 
                 valSequence: Tuple[int] = []):
        self.level: int = 1 if not parent else parent.level + 1     ####
        self.parent: Node = parent
        self.child: Node = None 
        self.valueSequence: Tuple[int] = valSequence
        self.dimension: int = baseNumber**2
        self.sequenceSoFar = self._getCurrentSequence() 
        self.possibleValues: iter[Tuple[int]] = self._getValues()
        
    def _getValues(self) -> iter[Tuple[int]]: 
        possibleValues: List[int] = list(self.valueSequence) if self.valueSequence else list(range(1, self.dimension+1))
        for i in range(randint(0, 42)):
            shuffle(possibleValues)
        return permutations(possibleValues)

    def _getCurrentSequence(self) -> Tuple[int]:
        sequence: list[int] = []
        node: Node = self
        while node != None:
            tmpSeq = list(node.valueSequence)
            tmpSeq.reverse()
            _ = [sequence.insert(0, val) 
                 for val in tmpSeq]
            node = node.parent
        return tuple(sequence)




class LinkedSequence:
    def __init__(self, baseNumber: int) -> None:
        self.BASE_NUMBER = baseNumber
        self.DIMENSION = baseNumber**2
        rootValue: List[int] = list(range(1, self.DIMENSION+1))
        for shuffling in  range(randint(0, factorial(self.DIMENSION)-1)):
            shuffle(rootValue)
        self.sequence: Tuple[int] 
        self.root: Node = Node(self.BASE_NUMBER, parent=None, valSequence=rootValue)
        self.initializeNode(self.root) 
        
    
    def initializeNode(self, node: Node):
        print(f"initialize @Level 1:  {node.valueSequence}")
        while node.level < self.DIMENSION:      ####            
            node = self.nextNode(node)
            self.sequence = node.sequenceSoFar
        
    def nextNode(self, node: Node) -> Node:         
        testGrid: Grid = Grid(self.BASE_NUMBER) 
        while True:
            try:
                candidateSeq: tuple[int] = next(node.possibleValues)
                testSequence: List[int] = list(testGrid.zeroSeq)
                for idx in range(len(node.sequenceSoFar)):
                    val: int = node.sequenceSoFar[idx]
                    testSequence[idx] = val                 
                addIdx: int = testSequence.index(0)
                for idx in range(len(candidateSeq)):
                    testSequence[idx + addIdx] = candidateSeq[idx]
                testGrid.insert(testSequence)
                if testGrid.gridCheckZero():
                    nextKid: Node = Node(self.BASE_NUMBER, node, candidateSeq)
                    node.child = nextKid                     
                    print(f"success    @Level {nextKid.level}:  {candidateSeq}")
                    return nextKid
            except StopIteration:
                return node.parent

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#
#                               Test grids for illustration

grd1 = (7, 2, 4, 9, 1, 3, 5, 6, 8,
        5, 1, 9, 6, 8, 7, 3, 4, 2,
        3, 8, 6, 2, 5, 4, 1, 9, 7,
        2, 3, 1, 4, 7, 9, 6, 8, 5,
        4, 6, 7, 5, 3, 8, 2, 1, 9,
        8, 9, 5, 1, 6, 2, 7, 3, 4,
        1, 7, 8, 3, 4, 5, 9, 2, 6,
        9, 4, 3, 7, 2, 6, 8, 5, 1,
        6, 5, 2, 8, 9, 1, 4, 7, 3)

grd2 = (2, 7, 4, 9, 1, 3, 5, 6, 8,
        1, 5, 9, 6, 8, 7, 3, 4, 2,
        8, 3, 6, 2, 5, 4, 1, 9, 7,
        3, 2, 1, 4, 7, 9, 6, 8, 5,
        6, 4, 7, 5, 3, 8, 2, 1, 9,
        9, 8, 5, 1, 6, 2, 7, 3, 4,
        7, 1, 8, 3, 4, 5, 9, 2, 6,
        4, 9, 3, 7, 2, 6, 8, 5, 1,
        5, 6, 2, 8, 9, 1, 4, 7, 3)

grd3 = (3, 7, 8, 4, 2, 6, 5, 9, 1,
        2, 5, 4, 1, 8, 9, 6, 3, 7,
        1, 9, 6, 7, 3, 5, 4, 2, 8,
        5, 6, 9, 3, 7, 8, 2, 1, 4,
        7, 3, 2, 5, 1, 4, 8, 6, 9,
        4, 8, 1, 9, 6, 2, 7, 5, 3,
        9, 2, 5, 8, 4, 3, 1, 7, 6,
        8, 1, 3, 6, 5, 7, 9, 4, 2,
        6, 4, 7, 2, 9, 1, 3, 8, 5)

grd4 = (2, 6, 4, 3, 8, 9, 5, 1, 7,
        5, 1, 7, 6, 4, 2, 9, 8, 3,
        3, 8, 9, 7, 5, 1, 4, 6, 2,
        4, 2, 6, 5, 1, 7, 3, 9, 8,
        9, 3, 8, 2, 6, 4, 1, 7, 5,
        1, 7, 5, 8, 9, 3, 6, 2, 4,
        6, 4, 2, 1, 3, 8, 7, 5, 9,
        7, 5, 3, 9, 2, 6, 8, 4, 1,
        8, 9, 1, 4, 7, 5, 2, 3, 6)

grd5 =  (3, 1, 7, 2, 4, 6, 5, 9, 8, 
         5, 8, 6, 7, 3, 9, 2, 1, 4, 
         4, 9, 2, 1, 8, 5, 6, 7, 3, 
         9, 3, 4, 5, 2, 8, 1, 6, 7, 
         7, 2, 8, 6, 9, 1, 4, 3, 5, 
         6, 5, 1, 4, 7, 3, 9, 8, 2, 
         8, 6, 5, 3, 1, 4, 7, 2, 9, 
         1, 7, 3, 9, 5, 2, 8, 4, 6, 
         2, 4, 9, 8, 6, 7, 3, 5, 1) 






if __name__ == '__main__': 
    print("""\n\n\t\t====================================================
    \t\t*                                                  *
    \t\t*                    PSEUDO_Q                      *         
    \t\t*                                                  *
    \t\t*            The syntax of Sudoku Grids            *
    \t\t*                 v2.0 (enter NumPy)               *
    \t\t*                                                  *
    \t\t*              part 1a: Infrastructure             *
    \t\t*                   Cells, Grids,                  *
    \t\t*                 and permutations                 *
    \t\t*                  (and much more)                 *
    \t\t*                                                  *
    \t\t====================================================
    """)








