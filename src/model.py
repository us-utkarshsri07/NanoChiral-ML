from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class NanotubePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def train(self, df):
        X = df[['chiral_n', 'chiral_m', 'u_coord', 'v_coord', 'initial_w']]
        y = df['target_w']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        return X_test, y_test
    
    def predict(self, input_data):
        return self.model.predict(input_data)