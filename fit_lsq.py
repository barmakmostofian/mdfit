import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score


# Set up command-line arguments
parser = argparse.ArgumentParser(description='Read data filefrom the command line.')
parser.add_argument('--feat_data',      type=str,   help='Path to the feature data file')
parser.add_argument('--response_data',  type=str,   help='Path to the response data file')
args = parser.parse_args()


# Load data
df_feat     = pd.read_csv(args.feat_data,     sep=',', header=0, index_col=0)
df_response = pd.read_csv(args.response_data, sep=',', header=0, index_col=0)


# Merge data to clearly define feature matrix (X) and response variable (y) for all data instanes
df_feat.index.name     = 'mol_name'
df_response.index.name = 'mol_name'

df_data = pd.merge(df_feat, df_response, on='mol_name', how='left')

df_data.set_index('mol_name')
df_data = df_data.dropna()

X = np.array(df_data.drop('potency', axis=1))
y = np.array(df_data['potency'])


# Initialize model and LOO strategy
model = LinearRegression()
loo = LeaveOneOut()


# Train and predict using LOO-CV
y_pred = np.zeros(df_data.shape[1])

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    y_pred[test_index] = model.predict(X_test)


# Evaluate overall performance
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score:           {r2:.4f}")
