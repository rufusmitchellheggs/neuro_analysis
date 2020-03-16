![alt text](https://github.com/rufusmitchellheggs/neuro_analysis/blob/master/place_cell_analysis/place_cell_table.png)


## place_cell_analysis: 
#### place_cell_identification_master.ipynb & place_cell_functions.py   
Place cell identification for Longitudinally registered cells - iterates through dataframe containing each session and respective stages, storing the following information:  

Place cell identification criteria:  
1. Cell has be a good cell both for df/f and for event traces  
2. Cell fire >3 times in any session  
3. Occupancy >5 (threshold can be adapted)  
4. Cell is in 95th mutual information percentile compared to 10000 SI shuffled distribution (threshold can be adapted)

(currently in the process of implementing CDM mutual information estimator for sparse sampling)
