#All imports
import pandas as pd
import numpy as np
from numpy import *
import scipy.signal 
import cv2
import os
from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt

#Functions

def events_pivot_correction(events, traces):
    """Changes format of timestamped events file to time series format
    as with raw data traces
    INPUT:
    ------
    events = events timestamp csv location (table format below)
    ---------------------------
    |Time (s)|Cell Name|Value|
    ---------------------------
    |        |         |     |
    
    trace = raw traces csv location

    OUTPUT:
    ------
    save_as = corrected event traces over input events csv 
    and outputs the file location
    """
    #create new csv
    save_as = events

    #Read in all sessions for event traces 
    event = pd.read_csv(events)

    #Table must be less than 10 columns
    if event.shape[1] < 10:
        
        #Read in all sessions for df/f raw trace
        trace = pd.read_csv(traces)
        #Pivot event traces so that cell identities are column headers
        event = event.pivot(index='Time (s)', columns=' Cell Name', values=' Value')
        event.fillna(value=0, inplace=True, axis=1)
        event.index = event.index.map(float) 
        event = event.sort_index(axis=0) 

        #Prepare events frame for merging
        event['Time (s)'] = event.index
        del event.index.name
        event.astype(float)
        event['Time (s)']=event['Time (s)'].astype(float)

        #Isolate time column from traces 
        trace = trace[1:]
        trace.rename(columns={trace.columns[0]: "Time (s)" }, inplace = True)
        trace = trace.drop(trace.columns[1:], axis=1)
        trace['Time (s)']=trace['Time (s)'].astype(float)

        #Merge events with traces time column, any time gaps are filled with 0s
        event = pd.merge(trace,event, on="Time (s)", how="left")
        event.fillna(value=0, inplace=True, axis=1)
        event = event.astype(float)
        event['Time (s)'] = trace['Time (s)'].values

        #Overwrite csv
        event.to_csv(save_as, index=False)
        print('File', save_as[-25:], 'has corrected and been overwitten')
    return save_as

def lr_data_correction_SI(lr_traces_or_events):
    """Labels and corrects timings for longitudinally registered csv file of multiple sessions/stages
    
    INPUT
    -----
    lr_traces_or_events = .csv file location for longitudinally registered events or traces
    timestamps = .csv file for timestamps of manually identified stage start and endings
    
    timestamps table format:
    ---------------------------
    |session|pre  |sam  | cho |
    ---------------------------
    | N01   |12701|21496|30611|
    ---------------------------
    
    OUPUT:
    -----
    corrected_data = A datafame containing labelled sessions and stages with corrected timings
    """

    input_file = lr_traces_or_events[-7:-4]

    #Read in lr_trace file location and make minor corrections
    lr_traces_or_events = pd.read_csv(lr_traces_or_events)


    if input_file == 'TRA':
        lr_traces_or_events = lr_traces_or_events.drop(lr_traces_or_events.index[0])
        lr_traces_or_events = lr_traces_or_events.reset_index(drop=True)
        lr_traces_or_events = lr_traces_or_events.rename(columns={" ": "Time (s)"})

    # #Read in timestamp info
    # # timestamps = pd.read_csv(timestamps)
    # # sessions = list(timestamps['session'])

    #Identify start and end frames for all sessions
    all_data = list(lr_traces_or_events["Time (s)"].astype(float))
    session_starts = [0]
    session_ends = []
    for i in range(len(all_data)):
        if i + 1 < len(all_data):
            if abs(all_data[i+1] - all_data[i]) > 1 :
                session_starts.append(i+1)
                session_ends.append(i)
    session_ends.append(len(lr_traces_or_events))

    stages = ['EXP', 'PRE', 'NOV', 'EXP', 'PRE', 'NOV', 'EXP', 'GTP']
    sessions = [1,1,1,2,2,2,3,3]
    # Save each session and each stage as a list 
    indiv_sessions = []
    for sesh, stage, start, end in zip(sessions, stages, session_starts, session_ends):

        #Isolate individual sessions
        indiv_session = lr_traces_or_events[start:end]
        indiv_session = indiv_session.reset_index(drop=True)

        #Correct timings and add column showing stage
        stage_timings = [np.arange(0, len(indiv_session['Time (s)']), 1)*0.05006]
        stage_timings = [item for sublist in stage_timings for item in sublist]

        indiv_session.insert(loc=0, column='stage', value=stage)
        indiv_session["Time (s)"] = stage_timings

        # Insert column showing session
        indiv_session.insert(loc=0, column='Session', value=list((sesh,) * len(indiv_session)))
        indiv_sessions.append(indiv_session)

    # #Concatenate all sessions into single table
    corrected_data = pd.concat(indiv_sessions)

    return corrected_data

def led_time(behavioural_video):
    """Find frame that an LED is switched on
    INPUT:
    - behavioural_video = the video being analysed
    
    OUTPUT:
    - Frame of LED turing on"""
    
    #Read in video
    cap = cv2.VideoCapture(behavioural_video)
    
    #Start Frame number
    frame = 1
    while True:
        pixels = cap.read() # Read in Pixel values
        
        #Define approximate LED region and extract mean of highest 100 pixel values
        light_region = pixels[1][700:-50][:,300:600] # LED Region (y range = [400:-1], x range = 0:300)
        light_frame = max(np.array(light_region).flatten()) #Maximum pixel value for region

        #If max value exceeds 250 then LED is switched on
        if light_frame > 250: 
            start_frame_vid = frame
            start_time_vid = start_frame_vid*(0.050)
            break
        else:
            frame +=1
    try:
        start_frame_vid
    except NameError:
        start_frame_vid = 'unknown'
        start_time_vid = 'unknown'
        print('START Time =', start_time_vid)
    else:
        print('START Time =', start_time_vid, 's')
        
    return start_frame_vid


def trace_behav_sync(directory_name, output_directory, file_dictionary, lr_traces, lr_events, animal, session, stage, meta):
    """Synchronise raw calcium traces with rat XY coordinates (Deeplabcut).
    
    INPUT
    - directory_name containing all files for one animal
    - output_directory is the directory you want to save it to, default = directory with all files
    - file_dictionary is a dictionary containing all individual sessions
    - lr_traces = longitudinally registered traces table
    - animal = amimal ID
    - session = recording session
    - stage = recording context
    
    Required directory contents:
    - GPIO file (start and end time for calcium) .csv
    - Behavioural video .flv
    - Raw Calcium Trace .csv
    - Deep lab cut x,y coordinate .csv 
    
    OUTPUT
    Table as CSV file containing:
    - Time (s)
    - Session Sand Well Choice (SW1, SW2, SW3)
    - Position Vector (0=Other, 1=SW1, 2=SW2, 3=SW3, 8=Startbox, 9=Doorway)
    - Raw x,y coordinates
    - Raw calcium trace
    """    
    
    # Save DLC, Behavioural vids and GPIO files locations as local variables
    files = file_dictionary[animal][session][stage]
    for file in files:
        if file.endswith("DLC.csv"):
            input_dlc = os.path.join(directory_name, file)
        elif file.endswith("BEH.mp4"):
            input_behavioural_video = os.path.join(directory_name, file)
        elif file.endswith('BEH.csv'):
            input_beh = os.path.join(directory_name, file)
            
    #See if there is an events file to process also
    for file in file_dictionary[animal]['8']['ALL']:
        if file.endswith('EVE.csv'):
            input_events = os.path.join(directory_name, file)
            events_file = True

    # Define start/end time and duration
    gpio_start = 0
    gpio_end = 598
#--------------------------------------------------------------------------------------------------------------
#ADAPT LONGITUDINAL REGISTRATION CODE FOR EVENT TRACES ALSO
    try:
        input_events
    except NameError:
        events_file = False
        event_dlc = 0
    else:
        #Isolate individual sessions and stages to align with gpio file
        event = lr_events[lr_events['Session']==int(session)][lr_events[lr_events['Session']==int(session)]['stage']==stage]
        event = event.reset_index(drop=True)

        # Trim traces using GPIO start/end times and reset to 0 start
        event_trimmed = event[event['Time (s)'] >= gpio_start][event[event['Time (s)'] >= gpio_start]['Time (s)'] < gpio_end]
        event_trimmed['Time (s)'] = event['Time (s)']-event['Time (s)'].iloc[0]
        event_trimmed = event_trimmed.reset_index(drop=True)

    #Isolate individual sessions and stages to align with gpio file
    trace = lr_traces[lr_traces['Session']==int(session)][lr_traces[lr_traces['Session']==int(session)]['stage']==stage]
    trace = trace.reset_index(drop=True)

    # Trim traces using GPIO start/end times and reset to 0 start
    trace_trimmed = trace[trace['Time (s)'] >= gpio_start][trace[trace['Time (s)'] >= gpio_start]['Time (s)'] < gpio_end]
    trace_trimmed['Time (s)'] = trace_trimmed['Time (s)']-trace_trimmed['Time (s)'].iloc[0]
    trace_trimmed = trace_trimmed.reset_index(drop=True)

    #DLC input and alignment
    dlc = pd.read_csv(input_dlc)
    dlc = dlc.drop([0,1]).convert_objects(convert_numeric=True)
    dlc = dlc.rename(columns={'scorer': 'Time (s)', 
                              'DeepCut_resnet101_hippoarena_dlcJul3shuffle1_500000':'x', 
                              'DeepCut_resnet101_hippoarena_dlcJul3shuffle1_500000.1':'y',
                              'DeepCut_resnet101_hippoarena_dlcJul3shuffle1_500000.2':'likelihood'})
    dlc['Time (s)'] = dlc['Time (s)']*0.05
    
    # Additional columns
    behav = pd.read_csv(input_beh)
    behav.rename(columns={'Time': 'Time (s)'})
    dlc['raw_behaviour'] = behav["raw_behaviour"]

    # Read in behavioural video and call LED light identification function
    behavioural_video = input_behavioural_video
    led_start = led_time(behavioural_video)*0.05

    # Trim DLC file to size
    dlc = dlc[dlc['Time (s)'] >= led_start]
    dlc['Time (s)'] = dlc['Time (s)']-dlc['Time (s)'].iloc[0]
    dlc = dlc[dlc['Time (s)'] < gpio_end]
    dlc = dlc.reset_index(drop=True)    
    
    frame_added = int((0.05/0.00006)+1)
    dlc = dlc.drop(dlc.index[[i for i in np.arange(0,len(dlc),1) if i % frame_added == 0][1:]])
    dlc = dlc.reset_index(drop=True)    
    # dlc = scipy.signal.resample(dlc, len(trace_trimmed['Time (s)'])) (interpolation way)
    
    # Add in stage metadata (including stranger, familiar, Genotype)
    meta = pd.read_csv(meta, index_col=0)
    
    stages = {'EXP':[0,1],
              'PRE':[2,3],
              'NOV':[4,5],
              'GTP':[2,3]}

    stage_meta = meta[[col for col in meta if col.startswith(str(session))]].iloc[:,stages[stage]].loc[str(animal)]
    meta=[]
    for i in dlc.iloc[:,4]: # Raw behaviour column
        if i == 1:
            meta.append([stage_meta[0][:2], stage_meta[0][2:]])
        elif i == 2:
            meta.append([stage_meta[1][:2], stage_meta[1][2:]])
        else:
            meta.append([0,0])
    meta = np.array(meta).transpose()
    
    #Create output dataframe
    dlc = pd.DataFrame({'Time (s)':list(np.arange(0, (len(trace_trimmed['Time (s)'])-0.05006)*0.05006, 0.05006)),
                        'x':dlc.iloc[:,1],
                        'y':dlc.iloc[:,2],
                        'likelihood':dlc.iloc[:,3],
                        'raw_behaviour':dlc.iloc[:,4],
                        'interaction partner':meta[0], 
                        'genotype':meta[1]})
    
    trace_trimmed = trace_trimmed.drop(['Time (s)'], axis=1)
    trace_dlc = pd.merge(dlc,trace_trimmed,left_index=True, right_index=True)
    
    #Saves file to CSV
    trace_dlc.to_csv(output_directory+animal+'_S'+session+'_'+stage+'_trace_dlc.csv', index=False)
    print(animal+'_S'+session+'_'+stage+'_trace_dlc.csv', 'saved to:', output_directory)
    
    if events_file:
        event_trimmed = event_trimmed.drop(['Time (s)'], axis=1)
        event_dlc = pd.merge(dlc,event_trimmed,left_index=True, right_index=True)
        
        #Saves file to CSV
        event_dlc.to_csv(output_directory+animal+'_S'+session+'_'+stage+'_events_dlc.csv', index=False)
        print(animal+'_S'+session+'_'+stage+'_events_dlc.csv', 'saved to:', output_directory)
    return trace_dlc, event_dlc
