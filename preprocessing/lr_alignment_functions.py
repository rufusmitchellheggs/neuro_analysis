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

def lr_data_correction(lr_traces_or_events, timestamps):
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

    #Read in timestamp info
    timestamps = pd.read_csv(timestamps)
    sessions = list(timestamps['session'])
    
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
        
    # Save each session and each stage as a list 
    indiv_sessions = []
    for sesh, start, end in zip(sessions, session_starts, session_ends):
        
        #Isolate individual sessions
        indiv_session = lr_traces_or_events[start:end]
        indiv_session = indiv_session.reset_index(drop=True)

        #isolate indiviudal stages within a session
        pre, sam, cho = np.array(timestamps[timestamps['session'].str.contains(sesh)])[0][1:].astype(int)
        if sesh == 'P01' or sesh == 'P02':
            stages = list(('PRE',) * pre) + list(('SAM',) * (sam-pre)) + list(('PRO',) * (cho-sam))
        else:
            stages = list(('PRE',) * pre) + list(('SAM',) * (sam-pre)) + list(('CHO',) * (cho-sam))
        
        #Correct timings and add column showing stage
        stage_timings = [np.arange(0, pre, 1)*0.05006,np.arange(0, (sam-pre), 1)*0.05006,np.arange(0, (cho-sam), 1)*0.05006]
        stage_timings = [item for sublist in stage_timings for item in sublist]

        indiv_session.insert(loc=0, column='stage', value=stages)
        indiv_session["Time (s)"] = stage_timings
        
        # Insert column showing session
        indiv_session.insert(loc=0, column='Session', value=list((sesh,) * len(indiv_session)))
        
        indiv_sessions.append(indiv_session)
    
    #Concatenate all sessions into single table
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
        light_region = pixels[1][400:-50][:,50:300] # LED Region (y range = [400:-1], x range = 0:300)
        light_frame = max(np.array(light_region).flatten()) #Maximum pixel value for region

        #If max value exceeds 250 then LED is switched on
        if light_frame > 250: 
            start_frame_vid = frame
            start_time_vid = start_frame_vid*(0.04)
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

def door_time(behavioural_video, y_correction = 0):
    """Obtains the frame that the event arena door opens
    
    INPUT:
    - behavioural_video = video being analysed
    - y_correction = adapt for strange start areas
    
    OUTPUT:
    - Frame of door opening
    - Top edge of the startbox"""
    
    #Read in video
    cap = cv2.VideoCapture(behavioural_video)

    #Start Frame number
    frame = 1
    delta_door_frame = []
    while frame < 12000:
        pixels = cap.read() # Read in Pixel values
        if frame == 1:
            
            box_region = np.mean(pixels[1][350:520][:,300:500], axis=2) # Start box region
            
            #box region correction (if pixel shift required)
            box_region_correction = 0 
            if np.mean(box_region) > 120:
                box_region_correction = 50
                box_region = np.mean(pixels[1][350:520][:,300+box_region_correction:500+box_region_correction], axis=2)
            
            #Pixel coordinate inside box
            inside_box = np.argmin(box_region)
            y = int((inside_box/200))+y_correction
            x = int(round(200*((inside_box/200)-int(inside_box/200))))
            
            #right edge detection
            pix = 1
            pix_val = box_region[y][x]
            while pix_val < 25:
                pix_val = box_region[y][x+pix]
                right_edge = x+pix
                pix +=1
            
            #top edge detection
            pix = 1    
            pix_val = box_region[y][right_edge]
            while pix_val < 60:
                pix_val = box_region[y-pix][right_edge]
                top_edge = y-pix
                pix +=1
                
#         print('right edge =',right_edge)
#         print('top edge =', top_edge)
#         print('y', 350+top_edge-42,350+top_edge, 'x', 300+box_region_correction+right_edge-100,300+box_region_correction+right_edge-90)
        
        #Door region obtained from top right corner (consistent for all videos)
        door_region = pixels[1][350+top_edge-42:350+top_edge][:,300+box_region_correction+right_edge-100:300+box_region_correction+right_edge-90]

        #Monitor door opening by tracking slope of pixel values
        door_frame = mean(np.array(door_region).flatten()) #Maximum pixel value for region
        delta_door_frame.append(door_frame)
        if len(delta_door_frame) > 10:
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0,10),delta_door_frame[-10:])
            if abs(slope) > 3.2: 
                door_frame_vid = frame
                door_time_vid = door_frame_vid*(0.04)
                break
            else:
                frame+=1
        else:
            frame +=1
            
    #If no door opening found
    try:
        door_frame_vid
    except NameError:
        door_frame_vid = 'unknown'
        door_time_vid = 'unknown'
        print('Door Opening Time =','corrupted - trying again')
    else:
        print('Door Opening Time =', door_time_vid, 's')
        
    return door_frame_vid, 350+top_edge

def sandwell_loc(behavioural_video):
    """Indentify Sandwell locations and radii from first frame

    INPUT
    - Behavioural video
    
    OUTPUT
    - Sandwell locations for: sw1, sw2, sw3"""
    
    # Read in first video frame
    cap = cv2.VideoCapture(behavioural_video)
    correct_sandwell = 'n'
    frame=1
    while correct_sandwell != 'y':
        img = cap.read()[1] # Read in Pixel values
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        gray_blurred = cv2.blur(gray, (3, 3)) 

        # Circle detection algorithm
        sandwells = cv2.HoughCircles(gray_blurred,  
                           cv2.HOUGH_GRADIENT, 1, 100, param1 = 50, 
                       param2 = 30, minRadius = 10, maxRadius = 20)
        if sandwells is not None:
            # Convert the circle parameters a, b and r to integers. 
            sandwells = np.uint16(np.around(sandwells)) 
        
        # Manually check that sandwells are correct
        if frame == 1 or frame % 100 == 0:
            for pt in sandwells[0, :]: 
                x, y, r = pt[0], pt[1], pt[2]
                cv2.circle(img, (x, y), r+20, (0, 255, 0), 2)
            avg_dis = np.mean(distance.cdist(sandwells[0][:,:2], sandwells[0][:,:2], 'euclidean'))
            if len(sandwells[0])==3 and avg_dis < 130:
                print(len(sandwells[0]),'Wells detected')
                plt.imshow(img) 
                plt.show()
                correct_sandwell = input("Are all sandwells correct - y/n?")
                frame+=1
            else:
                correct_sandwell = 'n'
                frame+=1
        else:
            frame+=1
            
        # Classify which sandwell is sw1, sw2, sw3
        for pt in sandwells[0, :]: 
            x, y = pt[0], pt[1]
            if y == min(np.array(sandwells).transpose()[1]):
                sw1 = [x,y]
            elif x == max(np.array(sandwells).transpose()[0]):
                sw2 = [x,y]
            else:
                sw3 = [x,y]
    print(len(sandwells[0]),'Wells correctly detected')
    return sw1,sw2,sw3

def trace_behav_sync(directory_name, output_directory, file_dictionary, lr_traces, lr_events, animal, session, stage):
    """Synchronise raw calcium traces with rat XY coordinates (Deeplabcut).
    
    INPUT:
    ------
    - directory_name containing all files for one animal
    - output_directory is the directory you want to save it to, default = directory with all files
    - file_dictionary is a dictionary containing all individual sessions
    - lr_traces = longitudinally registered traces table
    - lr_events = longitudinally registered events table (corrected)
    - animal = amimal ID
    - session = recording session
    - stage = recording context
    
    Required directory contents:
    - GPIO file (start and end time for calcium) .csv
    - Behavioural video .flv
    - Raw Calcium Trace .csv
    - Events trace .csv
    - Deep lab cut x,y coordinate .csv 
    
    OUTPUT:
    ------
    trace_dlc = data frame and saved csv with calcium traces (as below)
    event_dlc = data frame and saved csv with event traces (as below)
    door_frame_vid = Frame of start box door opening
    sandwells = List of sandwell locations 
    -------------------------------------------------------------
    |Time (s)|well|position|x|y|likelihood|Session|stage|C000|Cn|
    -------------------------------------------------------------
    |        |    |        | | |          |       |     |       |
    """    
    
    # Save DLC, Behavioural vids and GPIO files locations as local variables
    files = file_dictionary[animal][session][stage]
    for file in files:
        if file.endswith("DLC.csv"):
            input_dlc = os.path.join(directory_name, file)
        elif file.endswith("BEH.flv"):
            input_behavioural_video = os.path.join(directory_name, file)
        elif file.endswith('LED.csv'):
            input_gpio = os.path.join(directory_name, file)
            
    #See if there is an events file to process also
    for file in file_dictionary[animal]['ALL']['ALL']:
        if file.endswith('EVE.csv'):
            input_events = os.path.join(directory_name, file)
            events_file = True
            
    # Read in GPIO pulse .csv file and extract LED section
    gpio = pd.read_csv(input_gpio)
    gpio = gpio[gpio[' Channel Name']==' GPIO-1'].convert_objects(convert_numeric=True)

    # Define start/end time and duration
    gpio_start = round(gpio[gpio['Time (s)'] < 100][gpio[gpio['Time (s)'] < 100][' Value'] > 1000].iloc[0][0] / 0.05) * 0.05
    gpio_end = round(gpio[gpio['Time (s)'] > 100][gpio[gpio['Time (s)'] > 100][' Value'] > 1000].iloc[0][0] / 0.05) * 0.05
    duration = round((gpio_end - gpio_start)/ 0.05) * 0.05    
#--------------------------------------------------------------------------------------------------------------
#ADAPT LONGITUDINAL REGISTRATION CODE FOR EVENT TRACES ALSO
    try:
        input_events
    except NameError:
        events_file = False
        event_dlc = 0
    else:
        #Isolate individual sessions and stages to align with gpio file
        event = lr_events[lr_events['Session']==session][lr_events[lr_events['Session']==session]['stage']==stage]
        event = event.reset_index(drop=True)
        
        # Trim traces using GPIO start/end times and reset to 0 start
        event_trimmed = event[event['Time (s)'] >= gpio_start][event[event['Time (s)'] >= gpio_start]['Time (s)'] <= gpio_end]
        event_trimmed['Time (s)'] = event['Time (s)']-event['Time (s)'].iloc[0]
        event_trimmed = event_trimmed.reset_index(drop=True)
    
    #Isolate individual sessions and stages to align with gpio file
    trace = lr_traces[lr_traces['Session']==session][lr_traces[lr_traces['Session']==session]['stage']==stage]
    trace = trace.reset_index(drop=True)
    
    # Trim traces using GPIO start/end times and reset to 0 start
    trace_trimmed = trace[trace['Time (s)'] >= gpio_start][trace[trace['Time (s)'] >= gpio_start]['Time (s)'] <= gpio_end]
    trace_trimmed['Time (s)'] = trace_trimmed['Time (s)']-trace_trimmed['Time (s)'].iloc[0]
    trace_trimmed = trace_trimmed.reset_index(drop=True)
    
    #DLC input and alignment
    dlc = pd.read_csv(input_dlc)
    dlc = dlc.drop([0,1]).convert_objects(convert_numeric=True)
    dlc = dlc.rename(columns={'scorer': 'Time (s)', 'DeepCut_resnet101_hippoarena_dlcJul3shuffle1_500000':'x', 
                              'DeepCut_resnet101_hippoarena_dlcJul3shuffle1_500000.1':'y',
                              'DeepCut_resnet101_hippoarena_dlcJul3shuffle1_500000.2':'likelihood'})
    dlc['Time (s)'] = dlc['Time (s)']*0.04

    # Read in behavioural video and call LED light identification function
    behavioural_video = input_behavioural_video
    led_start = led_time(behavioural_video)*0.04

    # Trim DLC file to size
    dlc = dlc[dlc['Time (s)'] >= led_start]
    dlc['Time (s)'] = dlc['Time (s)']-dlc['Time (s)'].iloc[0]
    dlc = dlc[dlc['Time (s)'] <= duration]
    dlc = dlc.reset_index(drop=True)

    # Calculates when extra frame is added and removes it
    frame_added = int((0.05/0.00006)+1)
    dlc = dlc.drop(dlc.index[[i for i in np.arange(0,len(dlc),1) if i % frame_added == 0][1:]])
    dlc = dlc.reset_index(drop=True)    
    # dlc = scipy.signal.resample(dlc, len(trace_trimmed['Time (s)'])) # Interpolation method
    
    # Additional columns
    well = [input_dlc[-15:-12]]*len(trace_trimmed) # Well that pellet is hidden in (SW1,2,3)

    door_frame_vid, start_box = door_time(behavioural_video)
    if door_frame_vid == 'unknown' or door_frame_vid < 1:
        door_frame_vid, start_box = door_time(behavioural_video, y_correction=10)
    tone_frame = door_frame_vid - (125)

    # Sandwell center coordinates and radius
    sw1,sw2,sw3 = sandwell_loc(behavioural_video)
    sandwells = [sw1,sw2,sw3]
    sw_radius = 32 #<-------------------------------------SAND WELL RADIUS
    position = []
    for i in range(len(trace_trimmed)):
        if i < door_frame_vid:
            position.append(12)
        elif dlc.transpose()[2][i] > start_box:
            position.append(12)
        elif distance.euclidean(sw1, [dlc.transpose()[1][i], dlc.transpose()[2][i]]) < sw_radius:
            position.append(1)
        elif distance.euclidean(sw2, [dlc.transpose()[1][i], dlc.transpose()[2][i]]) < sw_radius:
            position.append(2)
        elif distance.euclidean(sw3, [dlc.transpose()[1][i], dlc.transpose()[2][i]]) < sw_radius:
            position.append(3)
        else:
            position.append(0)
    
    #Creat output dataframe
    new_dlc_data_frame = {'Time (s)':list(np.arange(0, len(trace_trimmed['Time (s)'])*0.05006, 0.05006)),
                          'well':well,
                          'position':position,
                          'x':dlc.transpose()[1], 
                          'y':dlc.transpose()[2],
                          'likelihood':dlc.transpose()[3],
                          'door_frame':[door_frame_vid]*len(trace_trimmed),
                          'tone_frame':[tone_frame]*len(trace_trimmed), 
                          'SW_locs':[sandwells]*len(trace_trimmed)}

    dlc = pd.DataFrame(data=new_dlc_data_frame)

    trace_trimmed = trace_trimmed.drop(['Time (s)'], axis=1)
    trace_dlc = pd.merge(dlc,trace_trimmed,left_index=True, right_index=True)
    
    #Saves file to CSV
    trace_dlc.to_csv(output_directory+input_behavioural_video[-25:-7]+'trace_dlc.csv', index=False)
    print(input_behavioural_video[-25:-7]+'trace_dlc.csv', 'saved to:', output_directory)
    if events_file:
        event_trimmed = event_trimmed.drop(['Time (s)'], axis=1)
        event_dlc = pd.merge(dlc,event_trimmed,left_index=True, right_index=True)
        #Saves file to CSV
        event_dlc.to_csv(output_directory+input_behavioural_video[-25:-7]+'events_dlc.csv', index=False)
        print(input_behavioural_video[-25:-7]+'events_dlc.csv', 'saved to:', output_directory)
    return trace_dlc, event_dlc
