import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)

selector = RFE(model, n_features_to_select=2)  # Выбираем 2 наиболее значимых признака
selector.fit(X_train, y_train)

selected_features = X_train.columns[selector.support_]
print("Выбранные признаки:", selected_features)

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)

accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели с отобранными признаками: {accuracy:.2f}")
