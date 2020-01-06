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

# # occupancy_map
# cols = events_dlc.columns[7:] # <--- Get all cell names from their columns
    
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
