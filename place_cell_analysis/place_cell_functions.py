import os
import time
import pandas as pd
import numpy as np
import math
from scipy import stats
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def trace_binning(trace_array, bin_size=15):
    """Bin trace time series in to select bin size (takes mean of bin)
    Input:
    ------
    - trace_array = N x R array, where N = Dimensions/number of neurons and R = Continuous trace
    - bin_size = The number of frames per bin
    
    Output:
    ------
    - binned_trace
    """ 
    bin_size = bin_size
    binned_trace = []
    for neuron in trace_array:
        per_neuron=[]
        i=0
        while i <= len(neuron):
            per_neuron.append(np.mean(neuron[i:i+bin_size]))
            i+=bin_size
        binned_trace.append(per_neuron)
    return binned_trace

def spike_binning(event_array, bin_size=15):
    """Bin event trace time series in to select bin size (takes sum of bin)
    Input:
    ------
    - event_array = N x R array, where N = Dimensions/number of neurons and R = event trace
    - bin_size = The number of frames per bin
    
    Output:
    ------
    - binned_events
    """ 
    bin_size = bin_size
    binned_events = []
    for neuron in event_array:
        per_neuron=[]
        i=0
        while i <= len(neuron):
            per_neuron.append(sum(neuron[i:i+bin_size]))
            i+=bin_size
        binned_events.append(per_neuron)
    return binned_events

def behav_vector_binning(behav_vector, bin_size=15):
    """Bin event trace time series in to select bin size (takes mode of bin)
    Input:
    ------
    - behav_vector = 1D behavioural vector
    - bin_size = The number of frames per bin
    
    Output:
    ------
    - binned_behav_vector
    """ 
    bin_size = bin_size
    binned_behav_vector = []
    i=0
    while i <= len(behav_vector):
        binned_behav_vector.append(stats.mode(behav_vector[i:i+bin_size])[0])
        i+=bin_size
    return binned_behav_vector


def occupancy_map(events_dlc, cells, plot_occupancy=None):
    """
    For N x neurons, generate occupancy maps of events in a spatial bin, 
    normalised by the number of times the animal is in the bin.  
    Also calculates the bin centre and counts the number of events in the bin/bins centre
    
    INPUT:
    ------
    events_dlc = event traces and dlc coordinates for N cells
    cells = Cell set to analyse (list of strings)
    plot_occupancy = None or True
    
    OUTPUT:
    ------
    max_events_all = maxmimum number of events in one place(or places) / cell 
    bin_centres_all = bin centre based on the maximum number of events
    bin_occupancy_all = number of visits to bin centre
    occupancy_map = subplot of all cell occupancy maps
    """
    
    place_field_dic = {} # <---- Create dictionary to store all place cells
    
    #Plot occupancy maps 
    if plot_occupancy == True:
        fig, axs = plt.subplots(int(len(cells)/5),5, figsize=(20, 100), facecolor='w', edgecolor='k', squeeze=False)
        fig.subplots_adjust(wspace=0.01, hspace=0.0001)
        axs = axs.ravel()
    
    #defining list to store max events in event occupancy map
    max_events_all = []
    bin_centres_all = []
    bin_occupancy_all = []
    for neuron in range(len(cells)):

        xedges = np.arange(0, 700, 720/33)
        yedges = np.arange(0, 600, 720/33)

        # Read in x,y coordinates
        x = events_dlc['x']
        y = events_dlc['y']

        # Read in correspoding events at each x,y coordinates
        weights = events_dlc[str(cells[neuron])]

        # Create 2D histogram for occupancy and for events
        spatial_occupancy, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        spatial_occupancy = spatial_occupancy.T  

        event_occupancy, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges), weights = weights)
        event_occupancy = event_occupancy.T

        # # Normalize events by occupancy and add gaussian filter
        # place_field = np.nan_to_num(event_occupancy/spatial_occupancy)
        # place_field = gaussian_filter(place_field, sigma=1)
        # place_field_dic[cells[neuron]] = place_field
        
        # plot place fields (Optional)   
        if plot_occupancy == True:
            try:
                axs[neuron].imshow(place_field, interpolation='nearest', origin='low',
                           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='coolwarm')
                axs[neuron].set_title(cells[neuron])
                axs[neuron].axis('off')
                axs[neuron].invert_yaxis()
            except IndexError:
                None

        #Returns the max number of events in event_occupancy array
        max_events = np.amax(event_occupancy, axis=None)
        max_events_all.append(max_events)

        #Returns the bin centre for the max number of events
        bin_edge_index = np.argwhere(event_occupancy == max_events)
        
        #If there are two bin centres, this will save the output
        if len(bin_edge_index) > 1:
            bin_centre = []
            bin_occupancy = [] 
            for i in range(len(bin_edge_index)):
                y_bin_edge_index, x_bin_edge_index = bin_edge_index[i]
                bin_centre.append([((xedges[x_bin_edge_index]+xedges[x_bin_edge_index+1])/2),
                                   ((yedges[y_bin_edge_index]+yedges[y_bin_edge_index+1])/2)])
                bin_occupancy.append(spatial_occupancy[bin_edge_index[i][0]][bin_edge_index[i][1]])
        else:
            y_bin_edge_index, x_bin_edge_index = bin_edge_index[0]
            bin_centre = [((xedges[x_bin_edge_index]+xedges[x_bin_edge_index+1])/2),
                          ((yedges[y_bin_edge_index]+yedges[y_bin_edge_index+1])/2)]
            bin_occupancy = [spatial_occupancy[bin_edge_index[0][0]][bin_edge_index[0][1]]]

        bin_centres_all.append(np.array(bin_centre))
        bin_occupancy_all.append(np.array(bin_occupancy))

    return max_events_all, bin_centres_all, bin_occupancy_all
    
def mi_place_response(neuron_events, occupancy_map):
    """Calculate the Spatial mutual information (bits) for neuronal response and occupancy
    Input:
    ------
    neuron_events = N x R array, where N = Dimensions/number of neurons and R = event trace (binned)
    occupancy_map = 1D Vector of the number of times an animal occupies a spatial bin
    
    Output:
    ------
    mutual_info = N x Spatial information (N = Neuron)
    """

    # Count the number of ones and zeros 
    count_individ_dic = {}
    probability_response_dic = {}
    for sample in neuron_events:
        if sample not in count_individ_dic:
            count_individ_dic[sample] = 1
        else:
            count_individ_dic[sample] += 1 

    # Count the number of times in each place/bin 
    count_dic = {}
    for place in occupancy_map:
        if place not in count_dic:
            count_dic[place] = 1
        else:
            count_dic[place] +=1  

    # Probability of being in each place/bin   
    probability_place_dic = {}
    for key in count_dic:
        probability_place_dic[key] = count_dic[key]/sum(list(count_dic.values())) 

    # Entropy for place vector
    entropyp = 0
    for key in probability_place_dic:
        entropyp += -(probability_place_dic[key]*math.log2(probability_place_dic[key]))

    # Probability of 1 and 0
    for key in count_individ_dic:
        probability_response_dic[key] = count_individ_dic[key]/sum(list(count_dic.values()))  

    # Entropy for neuron response
    entropyr = 0
    for key in probability_response_dic:
        entropyr += -(probability_response_dic[key]*math.log2(probability_response_dic[key]))
        
    # count the number of spikes per place
    count_spike_place_dic = {}
    for key in occupancy_map:
        count_spike_place_dic['0'+str(key)]=0
        count_spike_place_dic['1'+str(key)]=0   
    for sample, place in zip(neuron_events, occupancy_map):
        count_entry = str(int(sample))+str(int(place))
        if int(count_entry[0]) > 0:
            count_entry = str(1)+str(int(place))
            count_spike_place_dic[count_entry] += sample*1
        else:
            count_spike_place_dic[count_entry] += 1

    # Probability of event in each place/bin   
    probability_spike_place_dic = {}         
    for key in count_spike_place_dic:
        probability_spike_place_dic[key] = int(count_spike_place_dic[key])/sum(list(count_dic.values()))  

    # Joint entropy for place and response
    entropy_spike_place_dic = {}         
    for key in probability_spike_place_dic:
        if probability_spike_place_dic[key] > 0:
            entropy_spike_place_dic[key] = probability_spike_place_dic[key]*math.log2(1/probability_spike_place_dic[key])
        else:
            entropy_spike_place_dic[key] = 0
    j_entropy = sum(list(entropy_spike_place_dic.values()))

    #mutual information
    mutual_info = entropyr + entropyp - j_entropy
    return mutual_info

def mi_place_response_2(neuron_events, occupancy_map):
    """Calculate the Spatial mutual information (bits) for neuronal response and occupancy
    Input:
    ------
    neuron_events = N x R array, where N = Dimensions/number of neurons and R = event trace (binned)
    occupancy_map = 1D Vector of the number of times an animal occupies a spatial bin
    
    Output:
    ------
    mutual_info = N x Spatial information (N = Neuron)
    """
    
    # Count the number of ones and zeros 
    count_individ_dic = {}
    probability_response_dic = {}
    for sample in neuron_events:
        if sample not in count_individ_dic:
            count_individ_dic[sample] = 1
        else:
            count_individ_dic[sample] += 1 
            
    # Probability of 1 and 0 ----> p(k)
    for key in count_individ_dic:
        probability_response_dic[key] = count_individ_dic[key]/sum(list(count_individ_dic.values()))  

    # Count the number of times in each place/bin 
    count_dic = {}
    for place in occupancy_map:
        if place not in count_dic:
            count_dic[place] = 1
        else:
            count_dic[place] +=1  

    # Probability of being in each place/bin ----> p(x) 
    probability_place_dic = {}
    for key in count_dic:
        probability_place_dic[key] = count_dic[key]/sum(list(count_dic.values())) 

    # Probability of being in each place/bin ----> p(k,x) 
    count_spike_place_dic = {}
    for key in occupancy_map:
        count_spike_place_dic['0'+str(key)]=0
        count_spike_place_dic['1'+str(key)]=0   

    for sample, place in zip(neuron_events, occupancy_map):
        count_entry = str(int(sample))+str(int(place))
        if sample > 0:
            count_entry = str(1)+str(int(place))
            count_spike_place_dic[count_entry] += sample
        else:
            count_spike_place_dic[count_entry] += 1
                           
    # Probability of event in each place/bin   
    probability_spike_place_dic = {}
    for key in count_spike_place_dic:
        event_entry = int(key[0])
        place_entry = int(key[1:])
        probability_spike_place_dic[key] = (count_spike_place_dic[key]/count_dic[place_entry])*probability_place_dic[place_entry]
    
    # Sum of all spatial info 
    spatial_info=0
    for key in probability_spike_place_dic:
        event_entry = int(key[0])
        place_entry = int(key[1:])
        if probability_spike_place_dic[key] > 0:
            spatial_info += probability_spike_place_dic[key]*(math.log2(probability_spike_place_dic[key]/(probability_place_dic[place_entry]*(probability_response_dic[event_entry]))))
            
    return spatial_info

def shuffled_mi_place_response(neuron_events, occupancy_map, shuffles = 5000):
    """ 
    Creates n shuffles of the behavioral vector and calculates the new similarity 
    score between behavioral vectors and calcium trace vectors
    INPUT:
    -------
    >> behavioral_vector -  use behav_vectors func
    >> calcium_traces - (raw calcium trace for each file)
    >> shuffles - the number of times to shuffle the data (default = 5000)
    
    OUTPUT:
    -------
    >> similarity_shuffled_all - Similarity score for calcium trace vectors and shuffled behavioral vector
    >> similarity_calcium_traces - Shuffled behavioral vectors (as a list)
    """
    mi_shuffled_all = []
    neuron_events = neuron_events
    for i in range(shuffles):
        np.random.shuffle(neuron_events)
        mi_shuffled_all.append(mi_place_response_2(neuron_events, occupancy_map))

#         if i%1000:
#             print('iteration=',i)
    return mi_shuffled_all

def percentile(mi_shuffled_all, mi):
    """ 
    Calculates the percentile of the calcium trace vector similarity score
    INPUT:
    -------
    >> similarity_shuffled_all -  use shuffled_vector_scores function
    >> similarity_calcium_traces - use similarity_calc
    
    OUTPUT:
    -------
    >> percentile - percentile out of shuffled dsimilarity score distribution
    >> similarity_distribution_all - similarity score distribution for all behav_vectors
    """
    percentile = []
    similarity_distribution = mi_shuffled_all
    percentile.append(stats.percentileofscore(similarity_distribution, mi))
    return percentile, similarity_distribution

def mi_perc_dis(neuron_events, occupancy_map, shuffles=10):
    """Calculates spatial mutual for neuron and occupancy and compares
        it to spatial information distribution of shuffled neuron event traces and occupancy
        
        Input:
        -----
        neuron_events = N x R array, where N = Dimensions/number of neurons and R = event trace (binned)
        occupancy_map = 1D Vector of the number of times an animal occupies a spatial bin
        shuffles = number of shuffles (default = 10000)
        
        Ouput:
        -----
        mi = N x Spatial information (N = Neuron)
        perc = percentile of neuron spatial information compared to shuffled distribution
        dist = shuffled distribution"""
    #How much mutual information does the neuron response have for the positional vector
    mi = np.array(mi_place_response_2(neuron_events, occupancy_map))
    
    # Create a shuffled distribition and determine the percentile of information
    perc, dist = percentile(shuffled_mi_place_response(neuron_events, occupancy_map, shuffles=shuffles), mi)
    return mi, perc, dist
