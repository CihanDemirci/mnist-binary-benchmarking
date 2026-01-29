from sklearn.linear_model import LogisticRegression

def train_log_reg(X_train, y_train, X_test, y_test):
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return acc