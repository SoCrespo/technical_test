#encoding=utf8
import functools
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import utils


class FedasClassifier:
    """
    Class to fit model and predict FEDAS codes from article characteristics.
    """

    def __init__(self, n_estimators=100, random_state=42):
        self.columns = ['brand', 'article_main_category', 'article_type', 'article_detail', 
                        'comment', 'size', 'accurate_gender']
        self.columns_with_digits = ['brand', 'size']
        self.fedas_columns = ['fedas_1', 'fedas_2', 'fedas_3', 'fedas_4']
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=n_estimators, 
                                                                       random_state=random_state))
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)


    def _normalize_features(self, features: pd.DataFrame) -> pd.Series:
        """
        Return a pd.Series with normalized features as a single string for each row.

        E.g. of normalization of 1 row : 
        brand	    article_main_category	article_type	article_detail	comment	 size	accurate_gender
	    brand_293	Training	            Homme	        Shoes           2-low    38.5	 ho

        becomes: ['brand 293 training homme shoes low 38 5 ho'].
        """
        print("Normalizing features...")
        output = features.copy(deep=True)[self.columns]
        for col in self.columns:
            if col in self.columns_with_digits:
                output[col] = output[col].apply(lambda text: utils.normalize(text, keep_digits=True))
            else:
                output[col] = output[col].apply(lambda text: utils.normalize(text))
        return output.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)


    def _normalize_target(self, target: pd.Series) -> pd.DataFrame:
        """
        Return a pd.DataFrame with normalized fedas codes and self.fedas_columns.
        E.g. : [[123456], [456789]] --> [[1, 23, 45, 6], [4, 56, 78, 9]].
        """
        print("Normalizing target...")
        output = pd.DataFrame(target.apply(lambda x: utils.split_fedas_code(x)).tolist(), 
                columns=self.fedas_columns)
        return output


    def fit(self, X:pd.DataFrame, y:pd.Series)-> None:
        """
        Fit the model on the given data. X is a pd.DataFrame with features
        and y is a pd.Series with target fedas codes.
        """
        X = self._normalize_features(X)
        X = self.vectorizer.fit_transform(X)
        y = self._normalize_target(y)
        print("Fitting model...")
        self.classifier.fit(X, y)
        print('Done.')

    
    def predict_from_vector(self, vector: np.ndarray) -> tuple:
        """
        Return tuple (fedas code (str), fedas code confidence (float))
        for given vector.
        """
        codes = self.classifier.predict(vector)
        codes = codes[0].astype(str).tolist()
        fedas_code = ""
        for i, code in enumerate(codes):
            if i in [1, 2]:
                code = code.rjust(2, '0') # fedas_2 and fedas_3 have 2 digits
            fedas_code += code

        probas = self.classifier.predict_proba(vector)
        max_probas = [max(proba[0]) for proba in probas]
        confidence = functools.reduce(lambda x, y: x*y, max_probas)

        return (fedas_code, confidence)


    def predict(self, X):
        """
        Return a pd.DataFrame with predicted fedas codes
        and their confidence scores (from 0 to 1).
        """
        output = pd.DataFrame()
        features = self._normalize_features(X)
        features = self.vectorizer.transform(features)
        print("Predicting fedas codes...")
        for i in range(features.shape[0]):
            fedas, confidence = self.predict_from_vector(features[i])
            output.loc[i, 'fedas'] = fedas
            output.loc[i, 'confidence'] = confidence
        return output
        

