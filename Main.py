import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import CSP
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from scipy.linalg import eigh
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from scipy.signal import welch
from sklearn.metrics import accuracy_score
from mne.preprocessing import ICA


all_subs = np.arange(1, 10)
n_components_CSP=2 # or 3

from Clustering import cluster1, cluster2, cluster3
clusters=[cluster1,cluster2,cluster3]


def plot_testing(data,y,classifier_constructed,classifier_original,testing_sub):
    x0=data[:,0];x1=data[:,1]
    plt.figure()
    plt.scatter(x0, x1, c=y, cmap='viridis', edgecolors='k')

    # Create grid to evaluate model
    xx, yy = np.meshgrid(np.linspace(-7, 7, 100),
                         np.linspace(-7, 7, 100))
    zz1 = classifier_original.predict(np.c_[xx.ravel(), yy.ravel()])
    zz2 = classifier_constructed.predict(np.c_[xx.ravel(), yy.ravel()])
    zz1 = zz1.reshape(xx.shape)   
    zz2 = zz2.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, zz1, alpha=0.5, cmap='viridis')
    #plt.contourf(xx, yy, zz2, alpha=0.5, cmap='viridis')

    plt.xlabel('CSP 1')
    plt.ylabel('CSP 2')
    #plt.savefig(rf"decision_boundary_{testing_sub}.png")
    plt.clf()
    #plt.show()
    return()
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
def multiple_regression(training_models_arr,rest_features):
    num_models = training_models_arr.shape[1]
    my_independent_variables=rest_features[:, np.r_[0:5, 15:20]]
    models = []
    for i in range(num_models):
        my_dependent_variables = training_models_arr[:, i]
        model = LinearRegression()
        model.fit(my_independent_variables, my_dependent_variables)
        models.append(model)
    return tuple(models)
def generate_training_MI_models(data_MI,labels_MI):
    csp = CSP(n_components=n_components_CSP, reg=None, log=False)
    lda = LDA()
    X_csp = csp.fit_transform(data_MI, labels_MI)                     
    lda.fit(X_csp, labels_MI)
    coeffs= lda.coef_
    intercept= lda.intercept_.reshape(1,-1)
    model_parameters = np.hstack((coeffs, intercept))
    return (model_parameters)
def training_function():
    for sub in training_subs:
        data_MI,labels_MI,data_rest=initialization(sub); labels_MI=labels_MI-7

        model_parameters=generate_training_MI_models(data_MI,labels_MI)
        training_models.append(model_parameters)

        rest_features_set=generate_training_rest_features(data_rest)
        all_training_rest_features.append(rest_features_set)

    all_training_rest_features_arr = np.vstack(all_training_rest_features)
    training_models_arr = np.vstack(training_models)
    #
    models=multiple_regression(training_models_arr,all_training_rest_features_arr)
    return(models)
def testing_function(testing_sub):
    data_MI_testing,labels_MI_testing,data_rest_testing=initialization(testing_sub); labels_MI_testing=labels_MI_testing-7
    rest_features_testing=generate_training_rest_features(data_rest_testing)
    return(rest_features_testing,data_MI_testing, labels_MI_testing) 
def construct_LDA(rest_features_testing):
    testing_model_all_coeff=[]
    for model in models[:-1]:
        testing_model_coeff=model.predict(rest_features_testing[np.r_[0:5, 15:20]].reshape(1, -1))
        testing_model_all_coeff.append(testing_model_coeff)
    testing_model_intercept=models[-1].predict(rest_features_testing[np.r_[0:5, 15:20]].reshape(1, -1))
    lda_constructed = LDA()
    lda_constructed.coef_ = np.array([testing_model_all_coeff])
    lda_constructed.intercept_ = np.array(testing_model_intercept)
    lda_constructed.classes_ =  np.array([0,1])    
    return(lda_constructed)
def generate_training_rest_features(data_rest):
    channels = [0, 1]
    psds = []
    for ch in channels:
        ch_data = data_rest[ch]
        psds_per_ch = psd_generator(ch_data)
        psds.append(psds_per_ch)
    return np.concatenate(psds)
def psd_generator(data, fs=250, nperseg=125, freq_range=(6, 35)):
    f, psd = welch(data, fs=fs, nperseg=nperseg, scaling='density')
    idx = np.where((f >= freq_range[0]) & (f <= freq_range[1]))
    return psd[idx]

# Main
for cluster in clusters: 
    for testing_sub in cluster:
        training_subs = cluster.copy(); training_subs.remove(testing_sub)

        # Training
        training_models=[]; all_training_rest_features=[]
        models=training_function()

        # Testing
        rest_features_testing,data_MI_testing, labels_MI_testing=testing_function(testing_sub)

        # Generate constructed LDA
        lda_constructed=construct_LDA(rest_features_testing)

        # CSP         
        csp = CSP(n_components=n_components_CSP, reg=None, log=False)
        X_csp = csp.fit_transform(data_MI_testing, labels_MI_testing)
        
        # Performance-constructed
        predicted_classes = lda_constructed.predict(X_csp)
        accuracy_using_constructed = accuracy_score(labels_MI_testing, predicted_classes)
        
        # Performance-original
        lda_orig=LDA()                          
        lda_orig.fit(X_csp, labels_MI_testing)    
        predicted_classes = lda_orig.predict(X_csp)
        accuracy_using_orig = accuracy_score(labels_MI_testing, predicted_classes)
        if n_components_CSP==2:
            plot_testing(X_csp,labels_MI_testing,lda_constructed,lda_orig,testing_sub)
        print("sub:",testing_sub,"original acc:",accuracy_using_orig,"const acc:",accuracy_using_constructed)


