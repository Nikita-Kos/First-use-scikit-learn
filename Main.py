#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
sns.pairplot(iris, hue='species', height=1.5)


# In[74]:


# простая линейная регрессия
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model


# In[76]:


rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y)


# In[78]:


X = x[:, np.newaxis]
model.fit(X, y)


# In[80]:


xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)
plt.scatter(x, y)
plt.plot(xfit, yfit)


# In[14]:


iris = sns.load_dataset('Iris')
x_iris = iris.drop('species', axis=1)
y_iris = iris.species
x_iris


# In[21]:


# классификация набора данных Iris
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_iris, y_iris, random_state=1)


# In[22]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()  
model.fit(Xtrain, Ytrain)
y_model = model.predict(Xtest)
y_model


# In[23]:


from sklearn.metrics import accuracy_score
accuracy_score(Ytest, y_model)


# In[24]:


# понижение размерности набора данных Iris
from sklearn.decomposition import PCA
model = PCA(n_components=2)
model.fit(x_iris)
X_2D = model.transform(x_iris)
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot(x="PCA1", y="PCA2", hue='species', data=iris, fit_reg=False)


# In[25]:


# кластеризация набора данных Iris
from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components=3, covariance_type='full')
model.fit(x_iris)
y_gmm = model.predict(x_iris)
iris['cluster'] = y_gmm
sns.lmplot(x="PCA1", y="PCA2", data=iris, hue='species', col='cluster', fit_reg=False)


# In[27]:


# анализ рукописных цифр
from sklearn.datasets import load_digits
digits = load_digits()
digits.images.shape


# In[35]:


fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                        subplot_kw={'xticks':[], 'yticks':[]},
                        gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    
ax.text(0.05, 0.05, str(digits.target[i]), 
        transform=ax.transAxes, color='green')


# In[54]:


# понижение размерности
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
data_projected.shape


# In[56]:


plt.scatter(data_projected[:, 0], data_projected[:, 1],
                    c=digits.target, edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('CMRmap', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)


# In[57]:


# классификация цифр
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)


# In[66]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)


# In[59]:


from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)


# In[72]:


from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, y_model)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value') 
plt.ylabel('true value')


# In[73]:


fig, axes = plt.subplots(10, 10, figsize=(8, 8),
    subplot_kw={'xticks':[], 'yticks':[]},
    gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(y_model[i]), transform=ax.transAxes,
    color='green' if (ytest[i] == y_model[i]) else 'red')

