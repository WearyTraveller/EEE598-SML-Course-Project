import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from keras import initializers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("processed.cleveland.data", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13].astype(float)
Y = dataset[:,-1]

print(X.shape)

#We only care about the presence or absence of heart disease for now. Not a classification of the specific type of angiographic tension
for i in range(Y.shape[0]):
	if Y[i]>0:
		Y[i]=1
	
print(Y)

#Code to perform PCA
#pca = PCA(n_components=13)
#X_transform = pca.fit_transform(X)

#plt.plot(numpy.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('Number of components')
#plt.ylabel('Cumulative explained variance')
#plt.show()

# Create Neural Network model
def create_model(optimizer='adam', init='normal', nodes=13, loss='binary_crossentropy', activation='sigmoid', rate=0.1):
	model = Sequential()
	#First hidden layer
	model.add(Dense(11, input_dim=13, kernel_initializer=init, activation=activation))# Add regularization using kernel_regularizer=regularizers.l2(rate)))
	#Second Hidden layer
	model.add(Dense(nodes, input_dim=11, kernel_initializer=init, activation=activation))
	#Dropout layer
	model.add(Dropout(rate))
	#Output layer
	model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
	# Compile model
	model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
	return model

#Evaluate model with standardized dataset
#estimator = KerasClassifier(build_fn=create_baseline, epochs=200, batch_size=5, verbose=0)
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(estimator, X, Y, cv=kfold)
#print("Results with Normal Initializer, Sigmoid activation, 13 hidden nodes, Adam Optimizer, Binary Cross Entropy Loss, 200 epochs: %.2f%% (%.2f%%)\n" % (results.mean()*100, results.std()*100))


#grid search
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# grid search epochs, batch size and optimizer
dropout_rate = [0]
activation = ['sigmoid']
optimizers = ['adam']
loss = ['binary_crossentropy']
init = ['normal', 'uniform']
epochs = [100]
batches = [5]
nodes = [10]
param_grid = dict(nodes=nodes, optimizer=optimizers, epochs=epochs, batch_size=batches, init=init, loss=loss, activation=activation, rate=dropout_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y) #Activate PCA code above and change this to X_transform

# Best performance
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
f = open('results.txt', 'a')
for mean, stdev, param in zip(means, stds, params):
	f.write("%.2f%% (%.2f%%) with: %r" % (mean*100, stdev*100, param))






