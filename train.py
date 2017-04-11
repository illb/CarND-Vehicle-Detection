import numpy as np
import util
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

import data, feature
car_data, not_car_data = data.get_car_data()

params = data.ModelParams()

print("--------------------------------------")
timer = util.Timer("extract features")
car_features = feature.extract_features(car_data, params=params)
notcar_features = feature.extract_features(not_car_data, params=params)
timer.finish()

feature_list = [car_features, notcar_features]
# Create an array stack, NOTE: StandardScaler() expects np.float64
X = np.vstack(feature_list).astype(np.float64)

from sklearn.preprocessing import StandardScaler
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define a labels vector based on features lists
y = np.hstack((np.ones(len(car_features)),
              np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print("--------------------------------------")
print("train count={}, test count={}".format(len(X_train), len(X_test)))
print('Feature vector length:', len(X_train[0]))

print("--------------------------------------")
# Use a linear SVC (support vector classifier)
svc = LinearSVC()
# Check the training time for the SVC
timer = util.Timer("train SVC")
svc.fit(X_train, y_train)
timer.finish()


print("--------------------------------------")
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
predict_num = 10
timer = util.Timer("predict with SVC (predict_num = {})".format(predict_num))
print('predicts: ', svc.predict(X_test[0:predict_num]))
print('actual labels: ', y_test[0:predict_num])
timer.finish()

data.save_model(svc, X_scaler, params)
