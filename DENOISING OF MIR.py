import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr

pd.options.display.mpl_style = 'default'
plt.rcParams['figure.figsize'] = (15,5)

training = pd.read_csv('training.csv', index_col='PIDN')
targets = ['Ca', 'P', 'pH', 'SOC', 'Sand']
spectra = [m for m in list(training.columns) if m[0]=='m']

co2_cols = ['m2379.76', 'm2377.83', 'm2375.9', 'm2373.97',
'm2372.04', 'm2370.11', 'm2368.18', 'm2366.26',
'm2364.33', 'm2362.4', 'm2360.47', 'm2358.54',
'm2356.61', 'm2354.68', 'm2352.76']

corr_spectra = pd.DataFrame(data=None, index=spectra)

for t in targets:

 corr_spectra[t] = 0.
 for m in spectra:
  corr_spectra[t].ix[m] = pearsonr(training[m], training[t])[0]

 # blank out the CO2 bands
 corr_spectra[corr_spectra.index.isin(co2_cols)] = np.nan
 corr_spectra[t].plot()
 plt.savefig('{}.png'.format(t))
 plt.show()

# code in python for derivative + denoising filter.

targets = ['Ca','P','pH','SOC','Sand']
train_cols_to_remove = ['PIDN']+targets

df_train = pd.read_csv(training_file,tupleize_cols =True)
df_test = pd.read_csv(test_file)

x_train=df_train.drop(train_cols_to_remove,axis=1)
y_train=df_train[targets]
train_feature_list = list(x_train.columns)
spectra_features = train_feature_list
non_spectra_feats=['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI','Depth'] 
for feats in non_spectra_feats:
     spectra_features.remove(feats)

fltSpectra=flt.gaussian_filter1d(np.array(x_train[spectra_features]),sigma=20,order=1)

x_train["Depth"] = x_train["Depth"].apply(lambda depth:0 if depth =="Subsoil" else 1)

x_train[spectra_features]=fltSpectra


