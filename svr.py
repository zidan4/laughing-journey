from sklearn.svm import LinearSVR

X, y = [...] # a linear dataset
svm_reg = make_pipeline(StandardScaler(),
LinearSVR(epsilon=0.5, random_state=42))
svm_reg.fit(X, y)
