from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model
import pandas as pd
import numpy as np 
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

def readData():
    iris = datasets.load_iris()
    print(type(iris.data), type(iris.target))
    X = iris.data
    Y = iris.target
    df = pd.DataFrame(X, columns=iris.feature_names)
    print(df.head())
    
    # Logging Addition 1 by @cmh02
    # Print the shapes of the input data after reading
    print(f"[readData] Data Shape of X: {X.shape}")
    
    # Logging Addition 2 by @cmh02
    # Print the shapes of the target after reading
    print(f"[readData] Data Shape of Y: {Y.shape}")

    return df 

def makePrediction():
    iris = datasets.load_iris()
    
    # Logging Addition 3 by @cmh02
    # Print the type of the iris data
    print(f"[makePrediction] Iris Data Type: {type(iris.data)}")
    
    # Logging Addition 4 by @cmh02
    # Print the type of the iris target
    print(f"[makePrediction] Iris Target Type: {type(iris.target)}")

    # Logging Addition 5 by @cmh02
    # Print the first 5 entries of the iris data
    print(f"[makePrediction] Iris Data Sample: {iris.data[:5]}")

    # Logging Addition 6 by @cmh02
    # Print the shape of the iris data
    print(f"[makePrediction] Iris Data Shape: {iris.data.shape}")

    # Logging Addition 7 by @cmh02
    # Print the shape of the iris target
    print(f"[makePrediction] Iris Target Shape: {iris.target.shape}")

    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(iris['data'], iris['target'])
    X = [
        [5.9, 1.0, 5.1, 1.8],
        [3.4, 2.0, 1.1, 4.8],
    ]

    # Logging Addition 8 by @cmh02
    # Print the input data for prediction
    print(f"[makePrediction] Prediction Input Data: {X}")

    # Logging Addition 9 by @cmh02
    # Print the shape of the input data for prediction
    print(f"[makePrediction] Prediction Input Data Shape: {len(X)} x {len(X[0])}")

    prediction = knn.predict(X)
    print(prediction)   

    # Logging Addition 10 by @cmh02
    # Print the type of the prediction
    print(f"[makePrediction] Prediction Type: {type(prediction)}")

    # Logging Addition 11 by @cmh02
    # Print the shape of the prediction result
    print(f"[makePrediction] Prediction Shape: {prediction.shape}")

def doRegression():
    diabetes = datasets.load_diabetes()
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Logging Addition 12 by @cmh02
    # Print the type of the diabetes X data
    print(f"[doRegression] Diabetes X Data Type: {type(diabetes_X)}")

    # Logging Addition 13 by @cmh02
    # Print the shape of the diabetes X data
    print(f"[doRegression] Diabetes X Data Shape: {diabetes_X.shape}")

    diabetes_X_train = diabetes_X[:-20]

    # Logging Addition 14 by @cmh02
    # Print the type of the training X data
    print(f"[doRegression] Diabetes X Train Data Type: {type(diabetes_X_train)}")

    # Logging Addition 15 by @cmh02
    # Print the shape of the training X data
    print(f"[doRegression] Diabetes X Train Data Shape: {diabetes_X_train.shape}")

    diabetes_X_test = diabetes_X[-20:]

    # Logging Addition 16 by @cmh02
    # Print the type of the testing X data
    print(f"[doRegression] Diabetes X Test Data Type: {type(diabetes_X_test)}")

    # Logging Addition 17 by @cmh02
    # Print the shape of the testing X data
    print(f"[doRegression] Diabetes X Test Data Shape: {diabetes_X_test.shape}")

    diabetes_y_train = diabetes.target[:-20]

    # Logging Addition 18 by @cmh02
    # Print the type of the training y data
    print(f"[doRegression] Diabetes y Train Data Type: {type(diabetes_y_train)}")

    # Logging Addition 19 by @cmh02
    # Print the shape of the training y data
    print(f"[doRegression] Diabetes y Train Data Shape: {diabetes_y_train.shape}")

    diabetes_y_test = diabetes.target[-20:]

    # Logging Addition 20 by @cmh02
    # Print the type of the testing y data
    print(f"[doRegression] Diabetes y Test Data Type: {type(diabetes_y_test)}")

    # Logging Addition 21 by @cmh02
    # Print the shape of the testing y data
    print(f"[doRegression] Diabetes y Test Data Shape: {diabetes_y_test.shape}")

    regr = linear_model.LinearRegression()
    regr.fit(diabetes_X_train, diabetes_y_train)
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # Logging Addition 22 by @cmh02
    # Print the type of the predicted values
    print(f"[doRegression] Predicted Values Type: {type(diabetes_y_pred)}")

    # Logging Addition 23 by @cmh02
    # Print the shape of the predicted values
    print(f"[doRegression] Predicted Values Shape: {diabetes_y_pred.shape}")

    # Logging Addition 24 by @cmh02
    # Print the first 5 predicted values
    print(f"[doRegression] Predicted Values: {diabetes_y_pred[:5]}")

    # Logging Addition 25 by @cmh02
    # Get the r2 score of the regression model and print it to display incorrect prediction performance
    r2Score = regr.score(diabetes_X_test, diabetes_y_test)
    print(f"[doRegression] R2 Score of Regression Model: {r2Score}")

def doDeepLearning():
    train_images = mnist.train_images()

    # Logging Addition 26 by @cmh02
    # Print the type of the training images
    print(f"[doDeepLearning] Train Images Type: {type(train_images)}")

    # Logging Addition 27 by @cmh02
    # Print the shape of the training images
    print(f"[doDeepLearning] Train Images Shape: {train_images.shape}")

    train_labels = mnist.train_labels()

    # Logging Addition 28 by @cmh02
    # Print the type of the training labels
    print(f"[doDeepLearning] Train Labels Type: {type(train_labels)}")

    # Logging Addition 29 by @cmh02
    # Print the shape of the training labels
    print(f"[doDeepLearning] Train Labels Shape: {train_labels.shape}")

    test_images = mnist.test_images()

    # Logging Addition 30 by @cmh02
    # Print the type of the testing images
    print(f"[doDeepLearning] Test Images Type: {type(test_images)}")

    # Logging Addition 31 by @cmh02
    # Print the shape of the testing images
    print(f"[doDeepLearning] Test Images Shape: {test_images.shape}")

    test_labels = mnist.test_labels()

    # Logging Addition 32 by @cmh02
    # Print the type of the testing labels
    print(f"[doDeepLearning] Test Labels Type: {type(test_labels)}")

    # Logging Addition 33 by @cmh02
    # Print the shape of the testing labels
    print(f"[doDeepLearning] Test Labels Shape: {test_labels.shape}")

    train_images = (train_images / 255) - 0.5

    # Logging Addition 34 by @cmh02
    # Print the type of the trained images after modification
    print(f"[doDeepLearning] Modified Train Images Type: {type(train_images)}")

    # Logging Addition 35 by @cmh02
    # Print the shape of the trained images after modification
    print(f"[doDeepLearning] Modified Train Images Shape: {train_images.shape}")

    test_images = (test_images / 255) - 0.5

    # Logging Addition 36 by @cmh02
    # Print the type of the test images after modification
    print(f"[doDeepLearning] Modified Test Images Type: {type(test_images)}")

    # Logging Addition 37 by @cmh02
    # Print the shape of the test images after modification
    print(f"[doDeepLearning] Modified Test Images Shape: {test_images.shape}")

    train_images = np.expand_dims(train_images, axis=3)

    # Logging Addition 38 by @cmh02
    # Print the type of the train images after expanding dimensions with numpy
    print(f"[doDeepLearning] Expanded Train Images Type: {type(train_images)}")

    # Logging Addition 39 by @cmh02
    # Print the shape of the train images after expanding dimensions with numpy
    print(f"[doDeepLearning] Expanded Train Images Shape: {train_images.shape}")

    test_images = np.expand_dims(test_images, axis=3)

    # Logging Addition 40 by @cmh02
    # Print the type of the test images after expanding dimensions with numpy
    print(f"[doDeepLearning] Expanded Test Images Type: {type(test_images)}")

    # Logging Addition 41 by @cmh02
    # Print the shape of the test images after expanding dimensions with numpy
    print(f"[doDeepLearning] Expanded Test Images Shape: {test_images.shape}")

    num_filters = 8
    filter_size = 3
    pool_size = 2

    model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(10, activation='softmax'),
    ])

    # Compile the model.
    model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )

    # Train the model.
    model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=3,
    validation_data=(test_images, to_categorical(test_labels)),
    )

    model.save_weights('cnn.h5')

    predictions = model.predict(test_images[:5])

    # Logging Addition 42 by @cmh02
    # Print the type of the predictions
    print(f"[doDeepLearning] Predictions Type: {type(predictions)}")

    # Logging Addition 43 by @cmh02
    # Print the shape of the predictions
    print(f"[doDeepLearning] Predictions Shape: {predictions.shape}")

    # Logging Addition 44 by @cmh02
    # Get the accuracy and print it to display incorrect prediction performance
    results = model.evaluate(test_images, to_categorical(test_labels))
    print(f"[doDeepLearning] Model Loss and Accuracy: {results}")

    print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

    print(test_labels[:5]) # [7, 2, 1, 0, 4]

if __name__=='__main__': 
    data_frame = readData()
    makePrediction() 
    doRegression() 
    doDeepLearning() 