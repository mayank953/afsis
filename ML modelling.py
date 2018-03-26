
# =============================================================================
# MODEL & MACHINE LEARNING WITH RESULT
# =============================================================================
xtrain, xtest = np.array(train)[:,:3578], np.array(test)[:,:3578]


sup_vec = svm.SVR(C=10000.0, verbose = 2)
rf = RandomForestRegressor(n_estimators=20, n_jobs=-1, random_state = 512)

preds = np.zeros((xtest.shape[0], 5))
for i in range(5):
    sup_vec.fit(xtrain, labels[:,i])
    preds[:,i] = sup_vec.predict(xtest).astype(float)
    
    
preds = np.zeros((xtest.shape[0], 5))
for i in range(5):
    rf.fit(xtrain, labels[:,i])
    preds[:,i] = sup_vec.predict(xtest).astype(float)
    
sample = pd.read_csv('sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]

sample.to_csv('sub3_svd.csv', index = False)

