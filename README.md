# PsQ---Syntax-of-Sudoku-v2.0
 
See https://github.com/A-Lex-McLee/Sudoku_Grid_Syntax for a description of the base version.
### NB: This version is still under construction; its main purpose is for illustration!

Innovations in v2:
* use of Numpy instead of lists/tuples for grid operations
* class GridCollection to deal with collections of Sudoku grids, permutation series, false grids, one-hot encoding, and to produce train_test datasets e.g. for ML applications
* GUI, which is currently just for illustration 


## Main Components:
* class Grid (basic infrastructure)
* class GridCollection (extended infrastructure)
* class PseudoQ_Window (GUI visualization)
* auxiliary classes and functions
* Iupyter notebook "solver_full_CNN": to create a dataset, train a model, save data (= Sudoku game grids) and model to file; these can, in turn, be used in the GUI (check lines 49/50 that the filenames are correct)
