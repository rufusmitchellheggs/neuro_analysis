## neuro_analysis
Neurodata preprocessing and secondary analysis

## Projects:
Social interaction (mPFC)
Memory and Spatial Navigation (CA1 Hippocampus)

## preprocessing: 
cnmfe.ipynb isx.wrapper \n
Applies Constrained-non negative matrix factorisation extended (CNMF-e), a method devloped by Caiman, to identify and seperate out neurons.  Consequent calcium traces are deconvolved using OASIS AR1 and event traces extracted.
Input: x2 spatially downsampled, spatially bandpassed and motion corrected .isxd file
Output: Calcium traces and event traces cell contours in .isxd format

preprocessing_ea_old.py \n
Aligns calcium traces/events with behavioural videos.  Features include LED light, sandwell and door opening detection.
Input: Calcium traces/events, behavioural videos and animal coordinates (generated using deeplabcut)
Ouput: Table with aligned calcium, x,y animal location, behavioural vector .csv file with extra features

## place_cell_analysis: 
place_cell_identification_master.ipynb & place_cell_functions.py
Place cell identification using criteria:
    1. Cell has be a good cell both for df/f and for event traces
    2. Cell fire >3 times in any session
    3. Cell is in 0.05 mutual information percentile (threshold can be adapted)
    4. Occupancy >5 (threshold can be adapted)

