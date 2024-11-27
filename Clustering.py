import mne
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from scipy.linalg import eigh
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from scipy.signal import welch
from sklearn.metrics import accuracy_score
from mne.preprocessing import ICA

def psd_generator(data, fs=250, nperseg=125, freq_range=(6, 35)):
    f, psd = welch(data, fs=fs, nperseg=nperseg, scaling='density')
    idx = np.where((f >= freq_range[0]) & (f <= freq_range[1]))
    return psd[idx]

def initialization(sub):
    file_path = rf"dir.gdf"
    raw = mne.io.read_raw_gdf(file_path, preload=True)
    events, _ = mne.events_from_annotations(raw)
    raw.filter(6, 35, fir_design='firwin')
    raw.set_channel_types({ch_name: 'eeg' for ch_name in raw.ch_names[:22]})
    raw.set_channel_types({ch_name: 'eog' for ch_name in raw.ch_names[-3:]})

    ica = ICA(n_components=15, max_iter='auto', random_state=97)
    ica.fit(raw, picks=["eeg", "eog"])
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    for i in range(3):
        ica.exclude.append(np.argmax(np.abs(eog_scores[i])))
    ica.apply(raw)

    M_picks = ["EEG-C3", "EEG-C4", "EEG-Cz", "EEG-9", "EEG-10", "EEG-11", "EEG-12", "EEG-13"]
    event_id_MI = {'769': 7, '770': 8}
    event_id_rest = {'276': 3}

    epochs_MI = mne.Epochs(raw, events, event_id=event_id_MI, tmin=1.0, tmax=4.0, proj=True, picks=M_picks, baseline=None, preload=True)
    epochs_rest = mne.Epochs(raw, events, event_id=event_id_rest, tmin=1.0, tmax=4.0, proj=True, picks=M_picks, baseline=None, preload=True)

    data_MI = epochs_MI.get_data()
    labels_MI = epochs_MI.events[:, -1]
    data_rest = epochs_rest.get_data()[0]

    return data_MI, labels_MI, data_rest

def generate_training_rest_features(data_rest):
    channels = [0, 1]
    psds = []
    for ch in channels:
        ch_data = data_rest[ch]
        psds_per_ch = psd_generator(ch_data)
        psds.append(psds_per_ch)
    return np.concatenate(psds)

def rest_clustering():
    rest_features = {sub: generate_training_rest_features(initialization(sub)[2]) for sub in all_subs}
    psd_values = np.array([rest_features[sub] for sub in sorted(rest_features.keys())])
    
    psd_values_c3 = psd_values[:, :15]
    psd_values_c3_6_11_Hz = psd_values_c3[:, :5]
    psd_values_c4 = psd_values[:, -15:]
    psd_values_c4_6_11_Hz = psd_values_c4[:, :5]

    psd_c3_peak = np.max(psd_values_c3_6_11_Hz, axis=1)
    psd_c4_peak = np.max(psd_values_c4_6_11_Hz, axis=1)
    psd_peak_diff = np.abs(psd_c3_peak - psd_c4_peak)

    cluster1 = (np.argsort(psd_peak_diff)[:3] + 1).tolist()
    cluster2 = (np.argsort(psd_peak_diff)[3:6] + 1).tolist()
    cluster3 = (np.argsort(psd_peak_diff)[6:] + 1).tolist()

    return cluster1, cluster2, cluster3

all_subs = np.arange(1, 10)
cluster1, cluster2, cluster3 = rest_clustering()
print("cluster 1 is:",cluster1)
print("cluster 2 is:",cluster2)
print("cluster 3 is:",cluster3)