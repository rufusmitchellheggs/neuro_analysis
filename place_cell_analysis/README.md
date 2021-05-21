![alt text](https://github.com/rufusmitchellheggs/neuro_analysis/blob/master/place_cell_analysis/place_cell_table.png)


## place_cell_analysis: 
#### place_cell_identification_master.ipynb & place_cell_functions.py   
Place cell identification for Longitudinally registered cells - iterates through dataframe containing each session and respective stages, storing the following information:  

Place cell identification criteria:  

CONDITION 1: set n events for a cell to be considered

CONDITION 2: Set place cell spatial info percentile (spatial information can be calculated using Kraskow or Skaggs)

CONDITION 3: set number of traversals to be considered

CONDITION 4: % traversals that cell fires <-- accounts for rdm bursting

