## preprocessing: 
![alt text](https://raw.githubusercontent.com/rufusmitchellheggs/neuro_analysis/preprocessing/preprocessing_pipeline.png)


#### cnmfe.ipynb (isx wrapper)
Applies Constrained-non negative matrix factorisation extended (CNMF-e), a method devloped by Caiman, to identify and seperate out neurons.  Consequent calcium traces are deconvolved using OASIS AR1 and event traces extracted.  
- Input: x2 spatially downsampled, spatially bandpassed and motion corrected .isxd file  
- Output: Calcium traces and event traces cell contours in .isxd format

#### LR_alignment.ipynb & lr_alignment_functions.py & __init__.py
Alignment of DLC behaviour with raw calcium traces/event traces originating from the same neurons across multiple stages and multiple sessions.  Features include LED light, sandwell and door opening detection.  
- Input: Single csv containing all raw calcium traces or event traces, stage timestamps, behavioural videos and animal coordinates (generated using deeplabcut)  
- Ouput: .csv file containing table with aligned calcium, x,y animal location, behavioural vector .csv file with extra features for each individual session's stages and all sessions and stages concatanated

#### preprocessing_ea_old.py   
Aligns calcium traces/events with behavioural videos.  Features include LED light, sandwell and door opening detection.  
- Input: Calcium traces/events, behavioural videos and animal coordinates (generated using deeplabcut)  
- Ouput: Table with aligned calcium, x,y animal location, behavioural vector .csv file with extra features
