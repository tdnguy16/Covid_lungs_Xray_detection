import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import statsmodels.api as sm      # Scipy or sklearn or statmodels.api
import statsmodels.formula.api as smf # A way to do forward and backward selections
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import chisquare
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from concurrent.futures import ThreadPoolExecutor

#==============================================================================================
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None

    def _normalize(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X - self.mean) / (self.std + 1e-10)  # added epsilon to avoid division by zero

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, train_generator, test_generator, regularization=None, lambda_=0.1):
        # Calculate steps per epoch
        steps_per_epoch = train_generator.n // train_generator.batch_size

        # Process test data
        x_test, y_test = next(test_generator)
        x_test = self._normalize(x_test)  # Normalize the batch
        x_test = x_test.reshape((x_test.shape[0], -1)).astype(
            np.float32) / 255.  # Preprocess the data: flatten the images and scale to [0, 1]
        x_test = self.PCA_transform(x_test)

        # Loop through all training epochs
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}")

            # Iterate over batches
            for step in range(steps_per_epoch):
                x_batch, y_batch = next(train_generator)

                # Preprocess the data: flatten the images and scale to [0, 1]
                X_train = x_batch.reshape((x_batch.shape[0], -1)).astype(np.float32) / 255.
                Y_train = y_batch  # Assuming y_batch is already in the correct form

                # PCA
                X_train = self.PCA_transform(X_train)

                # Get number of samples (rows) and number of features (columns) from matrix X
                num_samples, num_features = X_train.shape

                # Initialize weights and bias if it's the first batch
                if epoch == 0:
                    self.weights = np.zeros(num_features)
                    self.bias = 0
                    self.mean = np.zeros(num_features)
                    self.std = np.zeros(num_features)

                # Forward pass
                linear_model = np.dot(X_train, self.weights) + self.bias
                predictions = self.sigmoid(linear_model)

                # Compute gradients
                dw = (1 / num_samples) * np.dot(X_train.T, (predictions - Y_train))
                db = (1 / num_samples) * np.sum(predictions - Y_train)

                # Regularization
                if regularization == 'l2':
                    dw += lambda_ * self.weights  # L2 regularization term
                    db += lambda_ * self.bias  # L2 regularization term
                elif regularization == 'l1':
                    dw += lambda_ * np.sign(self.weights)  # L1 regularization term
                    db += lambda_ * np.sign(self.bias)  # L1 regularization term

                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

                # Evaluate on this batch (optional, could also evaluate on a separate validation set)
                self.evaluate(X_train, Y_train)

        ## ROC
        self.plot_roc_curve_train(X_train, Y_train)
        self.plot_roc_curve_test(x_test, y_test)
        plt.show()

    def fit_adam(self, train_generator, test_generator, regularization=None, lambda_=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Calculate steps per epoch
        steps_per_epoch = train_generator.n // train_generator.batch_size

        # Process test data
        x_test, y_test = next(test_generator)
        x_test = self._normalize(x_test) # Normalize the batch
        x_test = x_test.reshape((x_test.shape[0], -1)).astype(np.float32) / 255. # Preprocess the data: flatten the images and scale to [0, 1]

        m_w, v_w = None, None
        m_b, v_b = 0, 0
        t = 0

        # Adam optimization
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}")

            # Iterate over batches
            for step in range(steps_per_epoch):
                x_batch, y_batch = next(train_generator)

                # Normalize the batch
                x_batch = self._normalize(x_batch)

                # Preprocess the data: flatten the images and scale to [0, 1]
                xi = x_batch.reshape((x_batch.shape[0], -1)).astype(np.float32) / 255.
                yi = y_batch


                # Initialize weights and bias if it's the first batch
                if epoch == 0 and m_w is None:
                    num_features = xi.shape[1]
                    self.weights = np.zeros(num_features)
                    m_w, v_w = np.zeros_like(self.weights), np.zeros_like(self.weights)
                    self.bias = 0
                    self.mean = np.zeros(num_features)
                    self.std = np.zeros(num_features)

                # Forward pass
                linear_model = np.dot(xi, self.weights) + self.bias
                predictions = self.sigmoid(linear_model)

                # Compute gradients
                dw = np.dot(xi.T, (predictions - yi)) / xi.shape[0]
                db = np.sum(predictions - yi) / xi.shape[0]

                # Regularization
                if regularization == 'l2':
                    dw += lambda_ * self.weights  # L2 regularization term
                    db += lambda_ * self.bias  # L2 regularization term
                elif regularization == 'l1':
                    dw += lambda_ * np.sign(self.weights)  # L1 regularization term
                    db += lambda_ * np.sign(self.bias)  # L1 regularization term
                elif regularization == 'both':
                    dw += lambda_ * np.sign(self.weights) + lambda_ * self.weights  # L1 and L2 regularization terms
                    db += lambda_ * np.sign(self.bias) + lambda_ * self.bias  # L1 and L2 regularization terms

                # Update biased first and second moment estimates for weights
                m_w = beta1 * m_w + (1 - beta1) * dw
                v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)

                # Update biased first and second moment estimates for bias
                m_b = beta1 * m_b + (1 - beta1) * db
                v_b = beta2 * v_b + (1 - beta2) * (db ** 2)

                # Compute bias-corrected moment estimates
                m_w_hat = m_w / (1 - beta1 ** (t + 1))
                v_w_hat = v_w / (1 - beta2 ** (t + 1))

                m_b_hat = m_b / (1 - beta1 ** (t + 1))
                v_b_hat = v_b / (1 - beta2 ** (t + 1))

                # Update parameters
                self.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
                self.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

                t += 1

                self.evaluate(xi, yi)

        ## ROC
        self.plot_roc_curve_train(xi, yi)
        self.plot_roc_curve_test(x_test, y_test)
        plt.show()

        # # Return the last batch's predictions for now
        # return predictions

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)

        class_predictions = [1 if i > 0.5 else 0 for i in y_pred]
        return class_predictions

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return y_pred

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)  # Labels
        # Calculate accuracy by commparing predicted y and tested y, which creates a boolean of 1 and 0
        # The mean is the percentage of True
        accuracy = np.mean(y_pred == y_test)
        print(f'Model Accuracy: {accuracy * 100:.2f}%')

        return y_pred

    def plot_roc_curve_train(self, X, y_true):
        X = self._normalize(X)

        # Assuming predict_proba is a method in your class that predicts the probability
        # of the positive class
        y_score = self.predict_proba(X)

        thresholds = np.linspace(1, 0, 100)
        tpr = []  # True Positive Rate
        fpr = []  # False Positive Rate

        for thresh in thresholds:
            y_pred = (y_score > thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))

            tpr.append(tp / (tp + fn))
            fpr.append(fp / (fp + tn))

        # Calculating AUC using the trapezoidal rule
        auc = np.trapz(tpr, x=fpr)

        # Plotting the ROC curve
        plt.plot(fpr, tpr, label=f'ROC Curve for TRAIN set (AUC = {auc:.5f})')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(True)

    def plot_roc_curve_test(self, X, y_true):

        X = self._normalize(X)

        # Assuming predict_proba is a method in your class that predicts the probability
        # of the positive class
        y_score = self.predict_proba(X)

        thresholds = np.linspace(1, 0, 100)
        tpr = []  # True Positive Rate
        fpr = []  # False Positive Rate

        for thresh in thresholds:
            y_pred = (y_score > thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))

            tpr.append(tp / (tp + fn))
            fpr.append(fp / (fp + tn))

        # Calculating AUC using the trapezoidal rule
        auc = np.trapz(tpr, x=fpr)

        # Plotting the ROC curve
        plt.plot(fpr, tpr, label=f'ROC Curve for TEST set (AUC = {auc:.5f})')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(True)

    def confusion_matrix(self, y_predict, y_test):
        print("Confusion Matrix:")

        y_predict = [1 if i == True else 0 for i in y_predict]
        y_test = [1 if i == True else 0 for i in y_test]

        print(len(y_predict))
        print(len(y_test))

    def PCA_transform(self, X_train):
        # Standardize the data
        scaler = StandardScaler()
        data_std = scaler.fit_transform(X_train)

        # Perform PCA
        pca = PCA(n_components=62)  # Reduce to 2 components for visualization
        principalComponents = pca.fit_transform(data_std)

        # Create a DataFrame with the principal components
        principalDf = pd.DataFrame(data=principalComponents,
                                   columns=[f'PC{i + 1}' for i in range(principalComponents.shape[1])])
        # Only take the top 6 parameters
        principalDf = principalDf[
            ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14',
             'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20', 'PC21', 'PC22', 'PC23', 'PC24', 'PC25', 'PC26', 'PC27',
             'PC28', 'PC29', 'PC30']]

        X_train = np.asarray(principalDf)

        # Explained variance ratio
        explained_variance = pca.explained_variance_ratio_

        # Eigenvalues (explained variance)
        eigenvalues = pca.explained_variance_

        # Create a DataFrame with the eigenvalues
        eigenvalues_df = pd.DataFrame(eigenvalues, index=['PC' + str(i + 1) for i in range(len(eigenvalues))],
                                      columns=['Eigenvalue'])

        # Sort the DataFrame by eigenvalues in descending order
        eigenvalues_df.sort_values(by='Eigenvalue', ascending=False, inplace=True)

        # Print the eigenvalues and their ranking
        # print(eigenvalues_df)

        return X_train

#==================================================================================================================
### Data processing
# Create an instance of ImageDataGenerator for training and one for testing
# Here we are just rescaling the images by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# images are in a directory named 'data/train' for training
# and 'data/test' for testing
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(28, 28),  # Resize images to the desired size
    color_mode='grayscale',  # Convert images to grayscale if needed
    batch_size=200,
    class_mode='binary')  # Use 'binary' for binary labels

num_batches = np.ceil(train_generator.n / train_generator.batch_size).astype(int)
print(f"Number of train batches: {num_batches}")

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(28, 28),
    color_mode='grayscale',
    batch_size=200,
    class_mode='binary')

num_batches = np.ceil(test_generator.n / test_generator.batch_size).astype(int)
print(f"Number of test batches: {num_batches}")

#==================================================================================================================
### Modelling
epoch = 10
#   Training models
model_0 = LogisticRegression(epochs=epoch)
model_1 = LogisticRegression(epochs=epoch)

print("This is fit model")
model_0.fit(train_generator, test_generator, regularization='l1', lambda_=0.01)

print("This is adam model")
model_1.fit_adam(train_generator, test_generator, regularization='l2', lambda_=0.01)
