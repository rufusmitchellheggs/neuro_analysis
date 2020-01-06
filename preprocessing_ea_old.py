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
    - Sandwells: sw1, sw2, sw3"""
    
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
                           cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                       param2 = 30, minRadius = 10, maxRadius = 20)
        if sandwells is not None:
            # Convert the circle parameters a, b and r to integers. 
            sandwells = np.uint16(np.around(sandwells)) 
        
        # Manually check that sandwells are correct
        if frame == 1 or frame % 100 == 0:
            for pt in sandwells[0, :]: 
                x, y, r = pt[0], pt[1], pt[2]
                cv2.circle(img, (x, y), r+20, (0, 255, 0), 2)
            print(len(sandwells[0]),'Wells detected')
            plt.imshow(img) 
            plt.show()
            correct_sandwell = input("Are all sandwells correct - y/n?")
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

def trace_behav_sync(directory_name, output_directory, file_dictionary, animal, session, stage):
    """Synchronise raw calcium traces with rat XY coordinates (Deeplabcut).
    
    INPUT
    - directory_name containing all files for one animal
    - output_directory is the directory you want to save it to, default = directory with all files
    - file_dictionary is a dictionary containing all individual sessions
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
    
    # Save files in session as local variable
    files = file_dictionary[animal][session][stage]
    for file in files:
        if file.endswith("DLC.csv"):
            input_dlc = os.path.join(directory_name, file)
        elif file.endswith("BEH.flv"):
            input_behavioural_video = os.path.join(directory_name, file)
        elif file.endswith('TRA.csv'):
            input_trace = os.path.join(directory_name, file)
        elif file.endswith('EVE.csv'):
            input_events = os.path.join(directory_name, file)
            events_file = True
        elif file.endswith('LED.csv'):
            input_gpio = os.path.join(directory_name, file)
            
    # Read in GPIO pulse .csv file and extract LED section
    gpio = pd.read_csv(input_gpio)
    gpio = gpio[gpio[' Channel Name']==' GPIO-1'].convert_objects(convert_numeric=True)

    # Define start/end time and duration
    gpio_start = round(gpio[gpio['Time (s)'] < 100][gpio[gpio['Time (s)'] < 100][' Value'] > 1000].iloc[0][0] / 0.05) * 0.05
    gpio_end = round(gpio[gpio['Time (s)'] > 100][gpio[gpio['Time (s)'] > 100][' Value'] > 1000].iloc[0][0] / 0.05) * 0.05
    duration = round((gpio_end - gpio_start)/ 0.05) * 0.05
    
    try:
        input_events
    except NameError:
        events_file = False
    else:
        # Read in events and restructure table usig traces as template
        trace = pd.read_csv(input_trace)
        trace = trace[1:]
        trace.rename(columns={trace.columns[0]: "Time (s)" }, inplace = True)
        trace = trace.drop(trace.columns[1:], axis=1)
        trace['Time (s)']=trace['Time (s)'].astype(float)

        # Event trace read in fix naming
        event = pd.read_csv(input_events, sep=',', dtype=str, error_bad_lines=False, encoding="utf-8-sig")
        event = event.pivot(index='Time (s)', columns=' Cell Name', values=' Value')
        event.fillna(value=0, inplace=True, axis=1)
        event.index = event.index.map(float) 
        event = event.sort_index(axis=0) 
        event['Time (s)'] = event.index
        del event.index.name
        event.astype(float)
        event['Time (s)']=event['Time (s)'].astype(float)
        event = pd.merge(trace,event, on="Time (s)", how="left")
        event.fillna(value=0, inplace=True, axis=1)
        event = event.astype(float)
        event['Time (s)'] = trace['Time (s)'].values
        
        # Trim traces using GPIO start/end times and reset to 0 start
        event_trimmed = event[event['Time (s)'] >= gpio_start][event[event['Time (s)'] >= gpio_start]['Time (s)'] <= gpio_end]
        event_trimmed['Time (s)'] = event['Time (s)']-event['Time (s)'].iloc[0]
        event_trimmed = event_trimmed.reset_index(drop=True)
    
    # Read in trace and fix bad naming                                  
    trace = pd.read_csv(input_trace)
    trace = trace.drop([0]).convert_objects(convert_numeric=True)
    trace = trace.rename(columns={' ': 'Time (s)'})

    # Trim traces using GPIO start/end times and reset to 0 start
    trace_trimmed = trace[trace['Time (s)'] >= gpio_start][trace[trace['Time (s)'] >= gpio_start]['Time (s)'] <= gpio_end]
    trace_trimmed['Time (s)'] = trace_trimmed['Time (s)']-trace_trimmed['Time (s)'].iloc[0]
    trace_trimmed = trace_trimmed.reset_index(drop=True)
  

    
    #DLC
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
    dlc = scipy.signal.resample(dlc, len(trace_trimmed['Time (s)']))
    
    # Additional columns
    well = [input_trace[-15:-12]]*len(trace_trimmed) # Well that pellet is hidden in (SW1,2,3)
    session = [input_trace[-11:-8]]*len(trace_trimmed) # Session (PRE, SAM, CHO)

    door_frame_vid, start_box = door_time(behavioural_video)
    if door_frame_vid == 'unknown' or door_frame_vid < 1:
        door_frame_vid, start_box = door_time(behavioural_video, y_correction=10)
    tone_frame = door_frame_vid - (125)

    # Sandwell center coordinates and radius
    sw1,sw2,sw3 = sandwell_loc(behavioural_video)
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
                          'session':session,
                          'position':position,
                          'x':dlc.transpose()[1], 
                          'y':dlc.transpose()[2],
                          'likelihood':dlc.transpose()[3]}
    dlc = pd.DataFrame(data=new_dlc_data_frame)

    trace_trimmed = trace_trimmed.drop(['Time (s)'], axis=1)
    trace_dlc = pd.merge(dlc,trace_trimmed,left_index=True, right_index=True)
    
    #Saves file to CSV
    trace_dlc.to_csv(output_directory+input_behavioural_video[-25:-7]+'trace_dlc.csv')

    if events_file:
        event_trimmed = event_trimmed.drop(['Time (s)'], axis=1)
        event_dlc = pd.merge(dlc,event_trimmed,left_index=True, right_index=True)
        #Saves file to CSV
        event_dlc.to_csv(output_directory+input_behavioural_video[-25:-7]+'events_dlc.csv')
    return trace_dlc, door_frame_vid, event_dlc