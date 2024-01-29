import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

df = sns.load_dataset("penguins")
df.dropna(inplace=True)

# define hyperparameters
predictors = ["flipper_length_mm", "bill_depth_mm", "sex"]
target = "species"
training_size = 0.8

# define the data
X = df[predictors]
y = df[target]

Xtrain, Xval, ytrain, yval = train_test_split(
    X, y, train_size=training_size, random_state=42
)

# define the model pipeline
coltrans = ColumnTransformer(
    [("one_hot_encode", OneHotEncoder(), ["sex"])], remainder="passthrough"
)

model = LogisticRegression(C=0.5)

pipeline = make_pipeline(
    coltrans,
    MinMaxScaler(),
    model,
)

# train
pipeline.fit(Xtrain, ytrain)

# evaluate the model
ypred = pipeline.predict(Xtrain)
train_acc = accuracy_score(y_pred=ypred, y_true=ytrain)
print("training accuracy  : ", train_acc)

ypred_val = pipeline.predict(Xval)
train_acc = accuracy_score(y_pred=ypred_val, y_true=yval)
print("validation accuracy: ", train_acc)
