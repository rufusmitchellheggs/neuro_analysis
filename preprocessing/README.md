![alt text](https://github.com/rufusmitchellheggs/neuro_analysis/blob/master/preprocessing/preprocessing_pipeline.png)

## preprocessing: 
#### cnmfe.ipynb (isx wrapper)
Applies Constrained-non negative matrix factorisation extended (CNMF-e), a method devloped by Caiman, to identify and seperate out neurons.  Consequent calcium traces are deconvolved using OASIS AR1 and event traces extracted.  
- Input: x2 spatially downsampled, spatially bandpassed and motion corrected .isxd file  
- Output: Calcium traces and event traces cell contours in .isxd format

#### LR_alignment.ipynb & lr_alignment_functions.py & __init__.py
Alignment of DLC behaviour with raw calcium traces/event traces originating from the same neurons across multiple stages and multiple sessions.  Features include LED light, sandwell and door opening detection.  
  
Input - single csv containing:  
- All raw calcium traces or event traces
- Behavioural videos
- Animal tracking coordinates (generated using deeplabcut)  

Ouput - csv file containing aligned:  
- Calcium traces (raw or events) 
- x,y animal location (downsampling & interpolation OPTIONAL) 
- Behavioural vector 
- Speed 
- Movement status 
- Stage/Session 
- Sand well locations 
- Door opening time 
- Tone time 

#### preprocessing_ea_old.py   
Aligns calcium traces/events with behavioural videos.  Features include LED light, sandwell and door opening detection.  
- Input: Calcium traces/events, behavioural videos and animal coordinates (generated using deeplabcut)  
- Ouput: Table with aligned calcium, x,y animal location, behavioural vector .csv file with extra features
