#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUDOKU GRIDs 

The Syntax of Sudoku Grids, Part B: Larger Infrastructure 
 
Created in Winter 2023, revised 2024/25

@author: alexanderpfaff

"""

from __future__ import annotations
from itertools import permutations #, combinations, product
from typing import Union, Tuple #, List,  Optional, TypeVar,  Set, Dict, Collection, Iterator
from random import sample #, randint, shuffle, choice 
# from math import factorial, sqrt
# from copy import deepcopy
# from dataclasses import dataclass
# from collections.abc import Callable
# import collections.abc.Collection
# import FunX  as fx
import numpy as np
from tqdm import tqdm
import sqlite3



class GridCollection:
    
    def __init__(self, collection: np.ndarray) -> None:
        
        #assert base_number**4 == collection[0].size
        base_number: int = 3
        
        try:
            self.__BASE_NUMBER:         int         = base_number
            self.__DIMENSION:           int         = base_number**2
            self.__SIZE_grid:           int         = base_number**4

            self.__collection:          np.ndarray  = collection.reshape(-1, self.SIZE_grid).astype(np.uint8)
            self.__SIZE_coll:           int         = collection.shape[0]
            self.__collIter:            iter        = iter(self.__collection)

            self.__activeSeries:        np.ndarray  = np.empty((0, self.SIZE_grid)).astype(np.uint8)
            self.__falseGrids:          np.ndarray  = np.empty((0, self.SIZE_grid)).astype(np.uint8) 
            self.__permutationStatus:   str         = "not activated"
        
        except Exception as e:
            if isinstance(e, AttributeError):
                raise AttributeError( f"Invalid datatype: expected ndarray, got instead {type(collection)}.")
            elif isinstance(e, ValueError):
                raise ValueError(f"Invalid input shape for reshaping to (n, {self.SIZE_grid}): shape={collection.shape}.")
            else:
                raise Exception(f"Unhandled exception: {e}")  
            


    """     basic stuff     """            

    @property 
    def BASE_NUMBER(self) -> int:
        return self.__BASE_NUMBER
    
    @property 
    def DIMENSION(self) -> int:
        return self.__DIMENSION

    @property 
    def SIZE_grid(self) -> int:
        return self.__SIZE_grid


    @property 
    def SIZE_coll(self) -> int:
        return self.__SIZE_coll
    
    @property 
    def SIZE_activeGrids(self) -> int:
        return self.__activeSeries.shape[0]
    
    @property 
    def SIZE_falseGrids(self) -> int:
        return self.__falseGrids.shape[0]

    @property 
    def permutationStatus(self) -> str:
        return self.__permutationStatus


    @property
    def shape(self):
        return self.__collection.shape



    @property 
    def activeGrid(self):
        return self.__activeSeries


    @property 
    def collection(self):
        return self.__collection


    @property 
    def falseGrids(self):
        return self.__falseGrids


    def __iter__(self) -> iter: 
        return iter(self.__collection)
    
    

    def __repr__(self) -> str:
        return("<class 'GridCollection'>")

    def __str__(self) -> str:
        return f"GridCollection[ shape: {self.shape}; " \
             + f"permutation series: {self.permutationStatus} " \
             + f"(=size: {self.SIZE_activeGrids}); false grids: {self.SIZE_falseGrids} ]" 


    def __eq__(self, other: GridCollection) -> bool:
        A_flat = self.reshape(-1, 81)
        B_flat = other.reshape(-1, 81)
        
        if A_flat.shape != B_flat.shape:
            return False  # early exit if shape mismatch
    
        # Lexicographically sort rows
        A_sorted = A_flat[np.lexsort(A_flat.T[::-1])]
        B_sorted = B_flat[np.lexsort(B_flat.T[::-1])]
    
        return np.array_equal(A_sorted, B_sorted)
    




    def reshape(self, *shape: int, inplace: bool =False) -> Union[None, np.ndarray]: 
        if inplace:
            self.__collection = self.__collection.reshape(*shape)
        else:
            return self.__collection.reshape(*shape)
        



    """     check if grids are valid    NB: time-consuming! """    
    

    def checkGridCollection(self, gridCollection: np.array = None) -> np.ndarray:
        """
        Validates a batch of 9x9 grids.
        
        Parameters:
        - gridCollection: np.ndarray of shape (n, 9, 9)
    
        Returns:
        - result: np.ndarray of shape (n,), with True for valid grids and False for invalid
        """        
        from PsQ_Grid import Grid
        
        if gridCollection is None:
            gridCollection = self.__collection

        result = np.full(self.SIZE_coll, True, dtype=bool)
            
        for idx in tqdm(range(self.SIZE_coll)):
            grid = gridCollection[idx]
    
            # Global check: count each value
            unique, counts = np.unique(grid, return_counts=True)
            if len(unique) != self.DIMENSION or not np.all(counts == self.DIMENSION):
                result[idx] = False
                continue
            
            # check single grid -- TODO: class method
            # if not Grid.checkGrid(grid):    
            if not Grid.checkGrid(grid):    
                result[idx] = False
        return result
    

    def check_allTrue(self, gridCollection: np.array = None) -> bool:
        if gridCollection is None:
            gridCollection = self.__collection
        return self.checkGridCollection(gridCollection=gridCollection).sum() == gridCollection.shape[0]

        
    def check_allFalse(self, gridCollection: np.array = None) -> bool:
        if gridCollection is None:
            gridCollection = self.__collection
        return self.checkGridCollection(gridCollection=gridCollection).sum() == 0



    def clear_falseGrids(self):
        self.__falseGrids: np.ndarray  = np.empty((0, self.SIZE_grid), dtype=np.uint8) 

    def clear_activeSeries(self):
        self.__activeSeries: np.ndarray  = np.empty((0, self.SIZE_grid), dtype=np.uint8) 




    """     activate permutation series & falseGrid collections    """

    def activate_HorizontalSeries(self, idx: int=0) -> None:
        """
        Generates all 9! (362,880) permutations of a Sudoku grid according to
        all possible re-coders that map [1,2,...,9] to a permutation of those digits.
    
        Parameters
        ----------
        grid : np.ndarray of shape (81,)
            The original grid with digits in the range 1-9 (no zeros).
    
        Returns
        -------
        np.ndarray of shape (362880, 81), dtype=np.uint8
            All re-coded permutations of the original grid.
        """
        if idx < 0 or idx >= self.SIZE_coll:
            raise ValueError(f"Index value idx must be 0 <= idx < {self.SIZE_coll}")
        
        grid = self.__collection[idx].copy()

        perms = np.array(list(permutations(range(1, 10))), dtype=np.uint8)  # shape: (362880, 9)
        
        # Create an array for all recoded grids
        recoded_grids = np.zeros((len(perms), 81), dtype=np.uint8)
        
        # Create a lookup index: grid maps values 1–9 to positions 0–8
        grid_index = grid - 1  # values 1–9 → 0–8
    
        # Use broadcasting: For each permutation, index into the perm array with grid_index
        recoded_grids = perms[:, grid_index]  # shape: (362880, 81)
    
        self.__activeSeries = recoded_grids.reshape(-1, 81)
        self.__permutationStatus = "horizontal"



    def activate_VerticalSeries(self) -> None:
        self.__activeSeries = self.__collection.reshape(-1, 81).copy()
        self.__permutationStatus = "vertical"



    def activate_RandomSeries(self, how_many: int = 362880, db_path="INT_Grid_rnd_2") -> None:

        self.__activeSeries =  GridCollection.fetch_GridCollection(how_many = how_many, db_path=db_path).reshape(-1, 81)
        self.__permutationStatus = "random"




####  fetch_GridCollection



    def makeFalseGrids_fromCurrent(self):
        
        assert self.__permutationStatus != "not activated"
        
        falseGrids = set()
        while len(falseGrids) < self.SIZE_activeGrids: 
            idx_coll = np.random.randint(0, self.SIZE_activeGrids)
            idx_grd = np.random.randint(0, 80)
            grid = list(self.activeGrid[idx_coll])
            if grid[idx_grd] != grid[idx_grd + 1]:  
                grid[idx_grd], grid[idx_grd + 1] = grid[idx_grd + 1], grid[idx_grd]
                falseGrids.add(tuple(grid))
        falseGrids = tuple(falseGrids)
        self.__falseGrids = np.array(falseGrids)
            
            

    def fetchFalseGrids(self, how_many: int = 100000, db_path="FalseGrids_rules") -> None:
        assert db_path in {"FalseGrids_rules",
                           "FalseGrids_v1off",
                           "FalseGrids_v2off",
                           "fromCurrent"
                           }, f"Invalid database name: {db_path}"
        if db_path == "fromCurrent":
            return self.makeFalseGrids_fromCurrent()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM int_grids")
        count = cursor.fetchone()[0]
        conn.close() 
        coll_size = min(count, how_many)
        ids = sample(range(1, count + 1), coll_size)
        ids.sort()
        
        _str_coll = GridCollection.fetch_stringsCollection(db_path, ids)
        _falseGrids = GridCollection.stringColl_toArray(_str_coll)
        self.__falseGrids = np.concatenate( (self.__falseGrids, _falseGrids), axis=0)

        

    def to_oneHot(self, grid_coll: np.ndarray = None) -> np.ndarray:
        """
        Convert an ndarray of int values to one-hot encoded form.
        
        Parameters:
            grid_coll (np.ndarray): Input array of shape (n, SIZE_grid) 
                                    containing ints from 1 to DIMENSION.
    
        Returns:
            np.ndarray: One-hot encoded array of shape (n , SIZE_grid, DIMENSION)
        """
        if grid_coll is None:
            grid_coll: np.ndarray = self.__activeSeries  
        gridCollection = grid_coll.astype(np.uint8).copy() - 1      # shift to zero-based indexing
        return np.eye(self.DIMENSION, dtype=np.uint8)[gridCollection]


        

    """     Class methods     """            

    @classmethod
    def from_sql(cls, db_name: str) -> GridCollection:
        """
        Reads Sudoku grids from a standardized SQLite database and returns them as a NumPy array.
    
        Parameters:
        - db_name: str - Path to the SQLite database file
    
        Returns:
        - grids: np.ndarray of shape (n, 9, 9), where each grid is parsed from the 'value' column
        """
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
    
        # Fetch all rows from the 'value' column of 'int_grids' table
        cursor.execute("SELECT value FROM int_grids")
        rows = cursor.fetchall()
        conn.close()
    
        # Convert each string to a 9x9 numpy array
        # grid_list = [
        #     np.fromiter((int(ch) for ch in row[0].strip()), dtype=np.uint8, count=81).reshape((9, 9))
        #     for row in tqdm(rows)
        # ]
        grid_list = [
            np.fromiter((int(ch) for ch in row[0].strip()), dtype=np.uint8, count=81).reshape((-1, 81))
            for row in tqdm(rows)
        ]
    
        return cls(np.stack(grid_list))  # CAREFUL HERE /todo!!


    @classmethod
    def from_randomDB(cls, db_path: str = "INT_Grid_rnd_1", how_many = 500000) -> GridCollection:
        gridcollection: np.array = GridCollection.fetch_GridCollection(how_many=how_many, db_path=db_path)
        return GridCollection(gridcollection)
        
        


    @classmethod
    def from_scratch(cls, double: bool = False) -> GridCollection:
        from PsQ_Grid import Grid
        grid = Grid()
        grid.generateGrid_deep()
        if double:
            return grid.fullPermutation()
        else: 
            return grid.permuteGrids()
        


    # TODO 
    @classmethod
    def from_file(cls, filepath: str):
        data = np.loadtxt(filepath, dtype=int, delimiter=",")
        return cls(data.reshape(-1))  # or r




    """     Static methods     """            



    @staticmethod
    def diaflect_Collection(gridCollection: np.ndarray) -> np.ndarray:

        gridReflections: list[np.ndarray] = []

        for grid_sequence in tqdm(gridCollection):
            grid = grid_sequence.reshape(9, 9)
            gridReflections.append(grid.T.copy())
        
        return np.array(gridReflections).reshape(-1, 81)



    @staticmethod
    def arrayColl_toStringCollection(gridCollection: np.ndarray) -> Tuple[str]:
        """
            Converts a (n, dim x dim) ndarray to a tuple of grid strings.    
            This method is intended as a pre-step for external storage (sql-db, .txt file etc.) 
            --> string is a simple & flexible dtype, easy to store and retrieve.

        Parameters
        ----------
        gridCollection : np.ndarray
            DESCRIPTION: int-grid collection; 

        Returns
        -------
        TYPE: Tuple[str]
            DESCRIPTION: collection of string representations of grids. 

        """
        try:
            base: int = int((gridCollection.size // gridCollection.shape[0])**(1.0/4))
            
            
            gridCollection.reshape(-1, base**4)
            coll_size: int = gridCollection.shape[0]
        except ValueError:
            print(f"Invalid input shape: {gridCollection.shape}.")
        
        return tuple(''.join(map(str, grid)) 
                     for grid in tqdm(gridCollection.reshape(coll_size, base**4)))


    @staticmethod
    def gridStringCollection_toSQL(gridCollection: Tuple[str], db_name: str):
             
        # Connect to SQLite database (it will create the database if it doesn't exist)
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
    
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS int_grids (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                value TEXT
            )
        ''')
    
        # Insert data into the table
        # The dataset should be a list or tuple of strings
        cursor.executemany('''
            INSERT INTO int_grids (value) VALUES (?)
        ''', [(item,) for item in tqdm(gridCollection, desc="Processing")])
    
        conn.commit()
        conn.close()


    @staticmethod
    def save_toSQL(gridCollection: np.ndarray, db_name: str) -> None:
        strColl: Tuple[str] = GridCollection.arrayColl_toStringCollection(gridCollection)
        GridCollection.gridStringCollection_toSQL(strColl, db_name)






    @staticmethod
    def stringColl_toArray(str_coll):
        """
        Converts a list of 81-digit strings to a NumPy array of shape (n, 81) with dtype uint8.
        
        Parameters:
            str_list (List[str]): List of digit strings, each exactly 81 characters long.
        
        Returns:
            np.ndarray: Array of shape (n, 81) with dtype uint8.
        """
        n = len(str_coll)
        flat_array = np.fromiter(
            (int(ch) for s in str_coll for ch in s), 
            dtype=np.uint8,
            count=n * 81
        )
        return flat_array.reshape((n, 81))
    



    @staticmethod
    def containsGrid(db_name: str, grid_str: str) -> bool:
        """
        Checks whether a given grid is already in the DB.
        
        Parameters:
            db_name (str):  Path to the SQLite database.
            grid_str (str): str representation of a grid.
        
        Returns:
            bool: True if db_name contains grid, False otherwise.
        """
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
    
        cursor.execute('SELECT 1 FROM int_grids WHERE value = ? LIMIT 1', (grid_str,))
        exists = cursor.fetchone() is not None
    
        conn.close()
        return exists




    @staticmethod
    def fetch_stringsCollection(db_path, id_list, batch_size=900):
        """
        Fetches strings from the specified column for a list of IDs, using batched queries for performance.
    
        Parameters:
            db_path (str): Path to the SQLite database.
            id_list (List[int]): List of integer IDs to fetch.
            table_name (str): Name of the table (default 'mytable').
            batch_size (int): Max number of IDs per batch (should be < 999).
        
        Returns:
            List[str]: Strings corresponding to the given IDs, in no guaranteed order.
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    
        results = []
        for i in range(0, len(id_list), batch_size):
            batch = id_list[i:i + batch_size]
            placeholders = ','.join(['?'] * len(batch))
            query = f'''
                SELECT value
                FROM int_grids
                WHERE id IN ({placeholders})
            '''
            cursor.execute(query, batch)
            results.extend(row[0] for row in cursor.fetchall())
    
        conn.close()
        return results
    
    
    
    
    
    @staticmethod    
    def fetch_GridCollection(how_many: int = 500000, db_path="INT_Grid_rnd_2"):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM int_grids")
        count = cursor.fetchone()[0]
        conn.close() 
        coll_size = min(count, how_many)
        ids = sample(range(1, count + 1), coll_size)
        ids.sort()
        
        _str_coll = GridCollection.fetch_stringsCollection(db_path, ids)
        return GridCollection.stringColl_toArray(_str_coll)
    
    
    
    
    @staticmethod
    def fetch_GridCollectionX(db_path="INT_Grid_rnd_1", id_list = None, how_many = 1500000, batch_size=900):
        
        if id_list is None:
            id_list = np.random.choice(range(11022000), how_many, replace=False)
        
        stringGridColl = GridCollection.fetch_stringsCollection(db_path, id_list, batch_size)
        
        return GridCollection.stringColl_toArray(stringGridColl)

    
    
    
    

    def train_test_gridSplit(self, 
        train_ratio:  float = 0.8,
        seed:         int   = None,
        label_dtype:  type  = np.uint8,
        to_oneHot:    bool  = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Shuffle, split, and mix two equally-sized grid collections of (shape (n,81) each) 
        for (binary) classification.
    
        Parameters
        ----------
        false_values : np.ndarray, shape (n,81)
            Grids labeled “False” (label 0).
        true_values : np.ndarray, shape (n,81)
            Grids labeled “True”  (label 1).
        train_ratio : float, default=0.8
            Fraction of each class to assign to the training set.
        seed : int or None
            Random seed for reproducibility.
        label_dtype : type, default=int
            Data type of the labels (e.g. int or bool).
    
        Returns
        -------
        X_train : np.ndarray, shape (~0.8·2n, 81)
        y_train : np.ndarray, shape (~0.8·2n,)
        X_test  : np.ndarray, shape (~0.2·2n, 81)
        y_test  : np.ndarray, shape (~0.2·2n,)
        
        """
        if to_oneHot:
            false_values: np.ndarray = self.to_oneHot(self.__falseGrids)
            true_values:  np.ndarray = self.to_oneHot(self.__activeSeries)
        else:
            false_values: np.ndarray = self.__falseGrids
            true_values:  np.ndarray = self.__activeSeries

        if seed is not None:
            np.random.seed(seed)
    
        # 1) Shuffle each class independently
        n_false = false_values.shape[0]
        n_true  = true_values.shape[0]
        idx_f = np.random.permutation(n_false)
        idx_t = np.random.permutation(n_true)
    
        false_shuffled = false_values[idx_f]
        true_shuffled  = true_values[idx_t]
    
        # 2) Split each into train/test
        n_f_train = int(np.floor(train_ratio * n_false))
        n_t_train = int(np.floor(train_ratio * n_true))
    
        f_train = false_shuffled[:n_f_train]
        f_test  = false_shuffled[n_f_train:]
        t_train = true_shuffled [:n_t_train]
        t_test  = true_shuffled [n_t_train:]
    
        # 3) Mix classes and build labels
        X_train = np.vstack((f_train, t_train))
        y_train = np.concatenate((
            np.zeros (f_train.shape[0], dtype=label_dtype),
            np.ones  (t_train.shape[0], dtype=label_dtype)
        ))
    
        X_test  = np.vstack((f_test,  t_test))
        y_test  = np.concatenate((
            np.zeros (f_test.shape[0],  dtype=label_dtype),
            np.ones  (t_test.shape[0],  dtype=label_dtype)
        ))
    
        # Final shuffle of each set to intermix classes
        def _shuffle(X, y):
            idx = np.random.permutation(X.shape[0])
            return X[idx], y[idx]
    
        X_train, y_train = _shuffle(X_train, y_train)
        X_test,  y_test  = _shuffle(X_test,  y_test)
    
        return X_train, y_train, X_test, y_test
    
        
        
        
    
    
    

if __name__ == '__main__': 
    print("""\n\n\t\t====================================================
    \t\t*                                                  *
    \t\t*                    PSEUDO_Q                      *         
    \t\t*                                                  *
    \t\t*            The syntax of Sudoku Grids            *
    \t\t*                 v2.0 (enter NumPy)               *
    \t\t*                                                  *
    \t\t*              part 1b: Infrastructure             *
    \t\t*                   GridCollections                *
    \t\t*                                                  *
    \t\t====================================================
    """)


    



