from minisom import MiniSom

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from scipy.io import arff
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#data = arff.loadarff('../data/flags.arff')
data = arff.loadarff('../data/emotions.arff')
#data = arff.loadarff('../data/CAL500.arff')

labels = 6
features = 72

df = pd.DataFrame(data[0])


'''
# flags 7, 19
df1 = df[['landmass','zone', 'area', 'population', 'language', 'religion', 'bars', 'stripes', 'colours', 'circles', 'crosses', 'saltires', 'quarters', 'sunstars', 'crescent', 'triangle', 'icon', 'animate', 'text']]
df2 = df[['red', 'green', 'blue', 'yellow', 'white', 'black', 'orange']]

str_df = df1.select_dtypes([np.object])
str_df = str_df.stack().str.decode('utf-8').unstack()
for col in str_df:
    df1[col] = str_df[col]
X = df1.values.astype('float')
'''





# emotions 6,72

df1 = df[['Mean_Acc1298_Mean_Mem40_Centroid',  'Mean_Acc1298_Mean_Mem40_Rolloff',  'Mean_Acc1298_Mean_Mem40_Flux',  'Mean_Acc1298_Mean_Mem40_MFCC_0', 'Mean_Acc1298_Mean_Mem40_MFCC_1',  'Mean_Acc1298_Mean_Mem40_MFCC_2',  'Mean_Acc1298_Mean_Mem40_MFCC_3',  'Mean_Acc1298_Mean_Mem40_MFCC_4',  'Mean_Acc1298_Mean_Mem40_MFCC_5', 'Mean_Acc1298_Mean_Mem40_MFCC_6', 'Mean_Acc1298_Mean_Mem40_MFCC_7', 'Mean_Acc1298_Mean_Mem40_MFCC_8', 'Mean_Acc1298_Mean_Mem40_MFCC_9', 'Mean_Acc1298_Mean_Mem40_MFCC_10', 'Mean_Acc1298_Mean_Mem40_MFCC_11', 'Mean_Acc1298_Mean_Mem40_MFCC_12', 'Mean_Acc1298_Std_Mem40_Centroid', 'Mean_Acc1298_Std_Mem40_Rolloff', 'Mean_Acc1298_Std_Mem40_Flux', 'Mean_Acc1298_Std_Mem40_MFCC_0', 'Mean_Acc1298_Std_Mem40_MFCC_1', 'Mean_Acc1298_Std_Mem40_MFCC_2', 'Mean_Acc1298_Std_Mem40_MFCC_3', 'Mean_Acc1298_Std_Mem40_MFCC_4', 'Mean_Acc1298_Std_Mem40_MFCC_5', 'Mean_Acc1298_Std_Mem40_MFCC_6', 'Mean_Acc1298_Std_Mem40_MFCC_7', 'Mean_Acc1298_Std_Mem40_MFCC_8', 'Mean_Acc1298_Std_Mem40_MFCC_9', 'Mean_Acc1298_Std_Mem40_MFCC_10', 'Mean_Acc1298_Std_Mem40_MFCC_11', 'Mean_Acc1298_Std_Mem40_MFCC_12', 'Std_Acc1298_Mean_Mem40_Centroid', 'Std_Acc1298_Mean_Mem40_Rolloff', 'Std_Acc1298_Mean_Mem40_Flux', 'Std_Acc1298_Mean_Mem40_MFCC_0', 'Std_Acc1298_Mean_Mem40_MFCC_1', 'Std_Acc1298_Mean_Mem40_MFCC_2', 'Std_Acc1298_Mean_Mem40_MFCC_3', 'Std_Acc1298_Mean_Mem40_MFCC_4', 'Std_Acc1298_Mean_Mem40_MFCC_5', 'Std_Acc1298_Mean_Mem40_MFCC_6', 'Std_Acc1298_Mean_Mem40_MFCC_7', 'Std_Acc1298_Mean_Mem40_MFCC_8', 'Std_Acc1298_Mean_Mem40_MFCC_9', 'Std_Acc1298_Mean_Mem40_MFCC_10', 'Std_Acc1298_Mean_Mem40_MFCC_11', 'Std_Acc1298_Mean_Mem40_MFCC_12', 'Std_Acc1298_Std_Mem40_Centroid', 'Std_Acc1298_Std_Mem40_Rolloff', 'Std_Acc1298_Std_Mem40_Flux', 'Std_Acc1298_Std_Mem40_MFCC_0', 'Std_Acc1298_Std_Mem40_MFCC_1', 'Std_Acc1298_Std_Mem40_MFCC_2', 'Std_Acc1298_Std_Mem40_MFCC_3', 'Std_Acc1298_Std_Mem40_MFCC_4', 'Std_Acc1298_Std_Mem40_MFCC_5', 'Std_Acc1298_Std_Mem40_MFCC_6', 'Std_Acc1298_Std_Mem40_MFCC_7', 'Std_Acc1298_Std_Mem40_MFCC_8', 'Std_Acc1298_Std_Mem40_MFCC_9', 'Std_Acc1298_Std_Mem40_MFCC_10', 'Std_Acc1298_Std_Mem40_MFCC_11', 'Std_Acc1298_Std_Mem40_MFCC_12', 'BH_LowPeakAmp', 'BH_LowPeakBPM', 'BH_HighPeakAmp', 'BH_HighPeakBPM', 'BH_HighLowRatio', 'BHSUM1', 'BHSUM2', 'BHSUM3']]

df2 = df[['amazed-suprised', 'happy-pleased', 'relaxing-calm', 'quiet-still', 'sad-lonely', 'angry-aggresive']]

X = df1.values



'''
# cal500 174, 68

df1 = df[['Mean_Acc1000_Mean_Mem40_ZeroCrossings_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Mean_Mem40_Centroid_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Mean_Mem40_Rolloff_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Mean_Mem40_Flux_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Mean_Mem40_MFCC0_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Mean_Mem40_MFCC1_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0','Mean_Acc1000_Mean_Mem40_MFCC2_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0','Mean_Acc1000_Mean_Mem40_MFCC3_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0','Mean_Acc1000_Mean_Mem40_MFCC4_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Mean_Mem40_MFCC5_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Mean_Mem40_MFCC6_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Mean_Mem40_MFCC7_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Mean_Mem40_MFCC8_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Mean_Mem40_MFCC9_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Mean_Mem40_MFCC10_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Mean_Mem40_MFCC11_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0','Mean_Acc1000_Mean_Mem40_MFCC12_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Std_Mem40_ZeroCrossings_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Std_Mem40_Centroid_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Std_Mem40_Rolloff_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Std_Mem40_Flux_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Std_Mem40_MFCC0_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Std_Mem40_MFCC1_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Std_Mem40_MFCC2_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Std_Mem40_MFCC3_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Std_Mem40_MFCC4_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Std_Mem40_MFCC5_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Std_Mem40_MFCC6_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Std_Mem40_MFCC7_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Std_Mem40_MFCC8_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Std_Mem40_MFCC9_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Std_Mem40_MFCC10_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Std_Mem40_MFCC11_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Mean_Acc1000_Std_Mem40_MFCC12_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Mean_Mem40_ZeroCrossings_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Mean_Mem40_Centroid_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Mean_Mem40_Rolloff_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Mean_Mem40_Flux_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0','Std_Acc1000_Mean_Mem40_MFCC0_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Mean_Mem40_MFCC1_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Mean_Mem40_MFCC2_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Mean_Mem40_MFCC3_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Mean_Mem40_MFCC4_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Mean_Mem40_MFCC5_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Mean_Mem40_MFCC6_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Mean_Mem40_MFCC7_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Mean_Mem40_MFCC8_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Mean_Mem40_MFCC9_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Mean_Mem40_MFCC10_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Mean_Mem40_MFCC11_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Mean_Mem40_MFCC12_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Std_Mem40_ZeroCrossings_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Std_Mem40_Centroid_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Std_Mem40_Rolloff_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Std_Mem40_Flux_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Std_Mem40_MFCC0_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Std_Mem40_MFCC1_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Std_Mem40_MFCC2_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Std_Mem40_MFCC3_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Std_Mem40_MFCC4_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Std_Mem40_MFCC5_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Std_Mem40_MFCC6_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Std_Mem40_MFCC7_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Std_Mem40_MFCC8_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Std_Mem40_MFCC9_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Std_Mem40_MFCC10_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Std_Mem40_MFCC11_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0', 'Std_Acc1000_Std_Mem40_MFCC12_Power_powerFFT_WinHamming_HopSize512_WinSize512_AudioCh0']]

df2 = df[['Angry-Agressive', 'NOT-Emotion-Angry-Agressive', 'Emotion-Arousing-Awakening', 'NOT-Emotion-Arousing-Awakening', 'Emotion-Bizarre-Weird', 'NOT-Emotion-Bizarre-Weird', 'Emotion-Calming-Soothing', 'NOT-Emotion-Calming-Soothing', 'Emotion-Carefree-Lighthearted', 'NOT-Emotion-Carefree-Lighthearted', 'Emotion-Cheerful-Festive', 'NOT-Emotion-Cheerful-Festive', 'Emotion-Emotional-Passionate', 'NOT-Emotion-Emotional-Passionate', 'Emotion-Exciting-Thrilling', 'NOT-Emotion-Exciting-Thrilling', 'Emotion-Happy', 'NOT-Emotion-Happy', 'Emotion-Laid-back-Mellow', 'NOT-Emotion-Laid-back-Mellow', 'Emotion-Light-Playful', 'NOT-Emotion-Light-Playful', 'Emotion-Loving-Romantic', 'NOT-Emotion-Loving-Romantic', 'Emotion-Pleasant-Comfortable', 'NOT-Emotion-Pleasant-Comfortable', 'Emotion-Positive-Optimistic', 'NOT-Emotion-Positive-Optimistic', 'Emotion-Powerful-Strong', 'NOT-Emotion-Powerful-Strong', 'Emotion-Sad', 'NOT-Emotion-Sad', 'Emotion-Tender-Soft', 'NOT-Emotion-Tender-Soft', 'Emotion-Touching-Loving', 'NOT-Emotion-Touching-Loving', 'Genre--_Alternative', 'Genre--_Alternative_Folk', 'Genre--_Bebop', 'Genre--_Brit_Pop', 'Genre--_Classic_Rock', 'Genre--_Contemporary_Blues', 'Genre--_Contemporary_RandB', 'Genre--_Cool_Jazz', 'Genre--_Country_Blues', 'Genre--_Dance_Pop', 'Genre--_Electric_Blues','Genre--_Funk', 'Genre--_Gospel', 'Genre--_Metal-Hard_Rock', 'Genre--_Punk', 'Genre--_Roots_Rock', 'Genre--_Singer-Songwriter', 'Genre--_Soft_Rock', 'Genre--_Soul', 'Genre--_Swing', 'Genre-Bluegrass', 'Genre-Blues', 'Genre-Country', 'Genre-Electronica', 'Genre-Folk', 'Genre-Hip_Hop-Rap', 'Genre-Jazz', 'Genre-Pop', 'Genre-RandB', 'Genre-Rock', 'Genre-World', 'Instrument_-_Acoustic_Guitar', 'Instrument_-_Ambient_Sounds', 'Instrument_-_Backing_vocals', 'Instrument_-_Bass', 'Instrument_-_Drum_Machine', 'Instrument_-_Drum_Set', 'Instrument_-_Electric_Guitar_(clean)', 'Instrument_-_Electric_Guitar_(distorted)', 'Instrument_-_Female_Lead_Vocals', 'Instrument_-_Hand_Drums',  'Instrument_-_Harmonica', 'Instrument_-_Horn_Section', 'Instrument_-_Male_Lead_Vocals','Instrument_-_Organ', 'Instrument_-_Piano', 'Instrument_-_Samples', 'Instrument_-_Saxophone', 'Instrument_-_Sequencer', 'Instrument_-_String_Ensemble', 'Instrument_-_Synthesizer', 'Instrument_-_Tambourine', 'Instrument_-_Trombone', 'Instrument_-_Trumpet', 'Instrument_-_Violin-Fiddle', 'Song-Catchy-Memorable', 'NOT-Song-Catchy-Memorable', 'Song-Changing_Energy_Level', 'NOT-Song-Changing_Energy_Level', 'Song-Fast_Tempo', 'NOT-Song-Fast_Tempo', 'Song-Heavy_Beat', 'NOT-Song-Heavy_Beat', 'Song-High_Energy', 'NOT-Song-High_Energy', 'Song-Like', 'NOT-Song-Like', 'Song-Positive_Feelings', 'NOT-Song-Positive_Feelings', 'Song-Quality', 'NOT-Song-Quality', 'Song-Recommend', 'NOT-Song-Recommend', 'Song-Recorded', 'NOT-Song-Recorded', 'Song-Texture_Acoustic', 'Song-Texture_Electric', 'Song-Texture_Synthesized', 'Song-Tonality', 'NOT-Song-Tonality', 'Song-Very_Danceable', 'NOT-Song-Very_Danceable', 'Usage-At_a_party', 'Usage-At_work', 'Usage-Cleaning_the_house', 'Usage-Driving', 'Usage-Exercising', 'Usage-Getting_ready_to_go_out', 'Usage-Going_to_sleep', 'Usage-Hanging_with_friends', 'Usage-Intensely_Listening', 'Usage-Reading', 'Usage-Romancing', 'Usage-Sleeping', 'Usage-Studying', 'Usage-Waking_up', 'Usage-With_the_family', 'Vocals-Aggressive', 'Vocals-Altered_with_Effects', 'Vocals-Breathy', 'Vocals-Call_and_Response', 'Vocals-Duet', 'Vocals-Emotional', 'Vocals-Falsetto', 'Vocals-Gravelly', 'Vocals-High-pitched', 'Vocals-Low-pitched', 'Vocals-Monotone', 'Vocals-Rapping', 'Vocals-Screaming', 'Vocals-Spoken', 'Vocals-Strong', 'Vocals-Vocal_Harmonies', 'Genre-Best--_Alternative', 'Genre-Best--_Classic_Rock', 'Genre-Best--_Metal-Hard_Rock', 'Genre-Best--_Punk', 'Genre-Best--_Soft_Rock', 'Genre-Best--_Soul', 'Genre-Best-Blues', 'Genre-Best-Country', 'Genre-Best-Electronica', 'Genre-Best-Folk', 'Genre-Best-Hip_Hop-Rap', 'Genre-Best-Jazz', 'Genre-Best-Pop', 'Genre-Best-RandB', 'Genre-Best-Rock', 'Genre-Best-World', 'Instrument_-_Acoustic_Guitar-Solo', 'Instrument_-_Electric_Guitar_(clean)-Solo', 'Instrument_-_Electric_Guitar_(distorted)-Solo', 'Instrument_-_Female_Lead_Vocals-Solo', 'Instrument_-_Harmonica-Solo', 'Instrument_-_Male_Lead_Vocals-Solo', 'Instrument_-_Piano-Solo', 'Instrument_-_Saxophone-Solo', 'Instrument_-_Trumpet-Solo']]

X = df1.values

'''

str_df = df2.select_dtypes([np.object])
str_df = str_df.stack().str.decode('utf-8').unstack()
for col in str_df:
    df2[col] = str_df[col]
y = df2.values.astype('float')


#X, y = make_multilabel_classification(n_samples=100, n_classes=5, n_labels=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#print(y_train.shape)

# Initialization and training
som = MiniSom(5, 5, 100, labels, features, sigma=1.0, learning_rate=0.05)
#print(X.shape)
som.random_weights_init(X_train)
print("Training...")
#print('data : ' + str(X_train))
som.trainsom(X_train, y_train, 100)  # random training
print("\n...ready!")
y_pred = som.classify(X_train, X_test, 100)


#print(y_pred.shape)
#print(y_test.shape)

for i in range(10):
    print('hello')
'''
# Plotting the response for each pattern in the iris dataset
plt.bone()
plt.pcolor(som.distance_map().T)  # plotting the distance map as background
plt.colorbar()

print(y.shape)
target = y_test
t = np.zeros(len(y_test), dtype=int)
#print(len(t))

#target = np.genfromtxt('iris.csv', delimiter=',', usecols=(4), dtype=str)
#print(len(target))

#print(t)
# use different colors and markers for each label
markers = ['o', 's', 'D', 'T']
colors = ['r', 'g', 'b', 'y']
#som = som.quantization(data)
#print('\n' + str(X))
for cnt, xx in enumerate(X_test):
    w = som.winner(xx)  # getting the winner
    #print(' w : ' + str(w) + '\n')
    # palce a marker on the winning position for the sample xx
    plt.plot(w[0]+.5, w[1]+.5, markers[t[cnt]], markerfacecolor='None',
             markeredgecolor=colors[t[cnt]], markersize=12, markeredgewidth=2)
plt.axis([0, 7, 0, 7])
plt.show()
'''