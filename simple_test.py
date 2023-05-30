import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

rng = np.random.RandomState(0)
X, y = make_regression(n_samples=20, n_features=1, random_state=0,
                      noise=4.0, bias=100.0)

print(f'x: {type(X)} shape: {X.shape}')
print(f'y: {type(y)} shape: {y.shape}')

model = LinearRegression(fit_intercept=True, 
                         copy_X=True,
                         n_jobs=1,
                        positive=False)


model.fit(X, y)

y_train_pred = model.predict(X)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(X, model.coef_ * X + model.intercept_, label=f'model (R2={model.score(X,y)})')
ax.plot(X, y, 'k.', label='data')
ax.plot(X, y_train_pred, 'ro', label='prediction')
ax.grid(True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()

plt.savefig('simple01.png')
plt.show()
