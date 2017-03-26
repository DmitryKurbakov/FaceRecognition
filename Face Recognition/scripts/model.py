from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.svm import SVC


class FaceRecognizer:
    def __init__(self):
        self._pca = PCA(n_components=300, whiten=True)
        self._pipeline = (
            SVC(kernel='rbf', class_weight='balanced', C=5, gamma=0.001, probability=True, tol=0.0000001),
            SVC(kernel='rbf', class_weight='balanced', C=10, gamma=0.0001, probability=True, tol=0.00000001),
        )

        self._classes = dict()
        self._classes_reverse = dict()

    def fit(self, X_train, y_train):
        self._pca.fit(X_train)

        for y in y_train:
            if not self._classes.get(y):
                self._classes[y] = len(self._classes) + 1
                self._classes_reverse[len(self._classes)] = y

        y_train_transform = []
        for y in y_train:
            y_train_transform.append(self._classes[y])

        X_train = self._pca.transform(X_train)

        for pipe in self._pipeline:
            pipe.fit(X_train, y_train_transform)

    def reverse_y(self, Y):
        return [self._classes[y] for y in Y]

    def get_target_names(self):
        return [self._classes_reverse[i] for i in range(1, len(self._classes) + 1)]

    def predict(self, X_test):
        pipe_predict = self.predict_proba(X_test)

        y_predict = []
        for y_pred in pipe_predict:
            y_predict.append(y_pred.index(max(y_pred)) + 1)

        return y_predict

    def predict_proba(self, X_test):
        X_test = self._pca.transform(X_test)
        pipe_predict = [[0] * len(self._classes) for _ in range(len(X_test))]

        for pipe in self._pipeline:
            prediction = pipe.predict_proba(X_test).tolist()

            for i in range(len(pipe_predict)):
                for j in range(len(pipe_predict[i])):
                    pipe_predict[i][j] = max(pipe_predict[i][j], prediction[i][j])

        return pipe_predict

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
