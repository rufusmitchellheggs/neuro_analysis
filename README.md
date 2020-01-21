# neuro_analysis
Endoscopic single-photon calcium imaging enables across day neural imaging of freely moving animals.  This repository contains a modular toolkit that suggests approaches to preprocess and analyse large quantities of neural data.  The scripts described below have been developed from three projects, encompassing two brain regions.  

For any information, please don't hestitate to get in touch: rufusmitchellheggs@gmail.com

## Projects and brain regions:
#### - Social interaction (mPFC)  
#### - Memory and Spatial Navigation (CA1 Hippocampus)

## preprocessing: 
#### cnmfe.ipynb (isx wrapper)
Applies Constrained-non negative matrix factorisation extended (CNMF-e), a method devloped by Caiman, to identify and seperate out neurons.  Consequent calcium traces are deconvolved using OASIS AR1 and event traces extracted.  
- Input: x2 spatially downsampled, spatially bandpassed and motion corrected .isxd file  
- Output: Calcium traces and event traces cell contours in .isxd format

#### LR_alignment.ipynb & lr_alignment_functions.py
Alignment of DLC behaviour with raw calcium traces/event traces originating from the same neurons across multiple stages and multiple sessions.  Features include LED light, sandwell and door opening detection.  
- Input: Single csv containing all raw calcium traces or event traces, stage timestamps, behavioural videos and animal coordinates (generated using deeplabcut)  
- Ouput: .csv file containing table with aligned calcium, x,y animal location, behavioural vector .csv file with extra features for each individual session's stages and all sessions and stages concatanated

#### preprocessing_ea_old.py   
Aligns calcium traces/events with behavioural videos.  Features include LED light, sandwell and door opening detection.  
- Input: Calcium traces/events, behavioural videos and animal coordinates (generated using deeplabcut)  
- Ouput: Table with aligned calcium, x,y animal location, behavioural vector .csv file with extra features

## place_cell_analysis: 
#### place_cell_identification_master.ipynb & place_cell_functions.py   
Place cell identification for Longitudinally registered cells - iterates through dataframe containing each session and respective stages, storing the following information:  

|Animal|Session|Stage|Neuron|Place Cell Status|  
|------|-------|-----|------|-----------------|


|Place Cell Centre|Place Cell Centre Event count|Rewarded Well|
|-----------------|-----------------------------|-------------|


|Euclidean Distance from SW1|Euclidean Distance from SW2|Euclidean Distance from SW3|
|---------------------------|---------------------------|---------------------------|


|Mutual_Information|Percentile|Distribution|
|------------------|----------|------------|


Place cell identification criteria:  
1. Cell has be a good cell both for df/f and for event traces  
2. Cell fire >3 times in any session  
3. Occupancy >5 (threshold can be adapted)  
4. Cell is in 95th mutual information percentile compared to 10000 SI shuffled distribution (threshold can be adapted)  




