import os
import pandas as pd 
import pickle  
import seaborn as sns     # Statistical visualization library based on Matplotlib
import matplotlib.pyplot as plt   # MATLAB-like plotting, useful for interactive viz
import json
#import matplotlib.pyplot as plt              # MATLAB-like plotting, useful for interactive viz


from sklearn.pipeline import Pipeline
from sklearn.datasets.base import Bunch
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

trainFile='adult.data'
testFile='adult.test'
names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
]

data=pd.read_csv('/home/lycui/Adult_Project/adult.data',sep="\s*",names=names)
#print data.head()
#sns.countplot(y='occupation', hue='income', data=data,)
#sns.plt.show()
#g=sns.FacetGrid(data, col='race', size=4, aspect=.5)
#g = g.map(sns.boxplot, 'income', 'education-num')
#sns.plt.show()
#sns.violinplot(x='sex', y='education-num', hue='income', data=data, split=True, scale='count')
#sns.plt.show()

meta={
    'target_names': list(data.income.unique()),
    'feature_names': list(data.columns),
    'categorical_features': {
    column: list(data[column].unique())
    for column in data.columns
    if data[column].dtype == 'object'
 },
}

with open('/home/lycui/Adult_Project/meta.json','w') as f:
    json.dump(meta,f,indent=2)


def load_data(root='/home/lycui/Adult_Project'):
    # Load the meta data from the file
    with open(os.path.join(root, 'meta.json'), 'r') as f:
        meta = json.load(f)

    names = meta['feature_names']   #The feature_names are simply the names of all the columns.
    
        # Load the training and test data, skipping the bad row in the test data
    train = pd.read_csv(os.path.join(root, 'adult.data'), names=names)
    test  = pd.read_csv(os.path.join(root, 'adult.test'), names=names, skiprows=1)

    # Remove the target from the categorical features
    meta['categorical_features'].pop('income')

    return Bunch(
    data = train[names[:-1]],
    target = train[names[-1]],
    data_test = test[names[:-1]],
    target_test = test[names[-1]],
    target_names = meta['target_names'],
    feature_names = meta['feature_names'],
    categorical_features = meta['categorical_features'],
    )
dataset = load_data()  


class EncodeCategorical(BaseEstimator, TransformerMixin):
    """
    Encodes a specified list of columns or all columns if None.
    """

    def __init__(self, columns=None):
        self.columns  = columns
        self.encoders = None

    def fit(self, data, target=None):
        """
        Expects a data frame with named columns to encode.
        """
        # Encode all columns if columns is None
        if self.columns is None:
            self.columns = data.columns

        # Fit a label encoder for each column in the data frame
        self.encoders = {
            column: LabelEncoder().fit(data[column])
            for column in self.columns
        }
        return self

    def transform(self, data):
        """
        Uses the encoders to transform a data frame.
        """
        output = data.copy()
        for column, encoder in self.encoders.items():
            output[column] = encoder.transform(data[column])

        return output

encoder = EncodeCategorical(dataset.categorical_features.keys())
data = encoder.fit_transform(dataset.data)
print data

if 0:
    class ImputeCategorical(BaseEstimator, TransformerMixin):
        """
        Encodes a specified list of columns or all columns if None.
        """

        def __init__(self, columns=None):
            self.columns = columns
            self.imputer = None

        def fit(self, data, target=None):
            """
            Expects a data frame with named columns to impute.
            """
            # Encode all columns if columns is None
            if self.columns is None:
                self.columns = data.columns

            # Fit an imputer for each column in the data frame
            self.imputer = Imputer(missing_values=0, strategy='most_frequent')
            self.imputer.fit(data[self.columns])

            return self

        def transform(self, data):
            """
            Uses the encoders to transform a data frame.
            """
            output = data.copy()
            output[self.columns] = self.imputer.transform(output[self.columns])

            return output


    imputer = ImputeCategorical(['workclass', 'native-country', 'occupation'])
    data = imputer.fit_transform(data)

    # Ee need to encode our target data as well
    yencode = LabelEncoder().fit(dataset.target)

    # Construct the pipeline
    census = Pipeline([
            ('encoder',  EncodeCategorical(dataset.categorical_features.keys())),
            ('imputer', ImputeCategorical(['workclass', 'native-country', 'occupation'])),
            ('classifier', LogisticRegression())
        ])

    # Fit the pipeline
    census.fit(dataset.data, yencode.transform(dataset.target))

    # encode test targets
    y_true = yencode.transform([y for y in dataset.target_test])

    # use the model to get the predicted value
    y_pred = census.predict(dataset.data_test)

    def dump_model(model, path='/home/lycui/Adult_Project', name='classifier.pickle'):
        with open(os.path.join(path, name), 'wb') as f:
            pickle.dump(model, f)

    dump_model(census)

    def load_model(path='/home/lycui/Adult_Project/classifier.pickle'):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def predict(model, meta=meta):
        data = {} # Store the input from the user

        for column in meta['feature_names'][:-1]:
            # We cheat and use the mean value for the weighting category, figuring
            # that most users won't know what theirs is.
            if column == 'fnlwgt':
                data[column] = 189778
            else:
                # Get the valid responses
                valid = meta['categorical_features'].get(column)

                # Prompt the user for an answer until good
                while True:
                    val = " " + raw_input("enter {} >".format(column))
                    if valid and val not in valid:
                        print "Not valid, choose one of {}".format(valid)
                    else:
                        data[column] = val
                        break

        # Create prediction and label
        yhat = model.predict(pd.DataFrame([data]))
        print "We predict that you make %s" % yencode.inverse_transform(yhat)[0]


    # Execute the interface
    model = load_model()
    predict(model)