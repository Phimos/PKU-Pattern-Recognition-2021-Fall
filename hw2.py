import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat

from linear import FisherLinearDiscriminant, LogisticRegression

if __name__ == "__main__":
    # Load data
    data = loadmat('hw2.mat')['data']
    X, y = data[:, :-1], data[:, -1]
    y[y == 2] = 0
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    # Fit Fisher Linear Discriminant
    model = FisherLinearDiscriminant()
    model.fit(X, y)
    print("Fisher Linear Discriminant Accuracy:", model.score(X, y))

    # Fit Logistic Regression
    model = LogisticRegression()
    model.fit(X, y)
    print("Logistic Regression Accuracy:", model.score(X, y))

    plt.plot([item['loss'] for item in model.history])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.savefig('loss.png', dpi=300)
    plt.close()

    plt.plot([item['accuracy'] for item in model.history])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.savefig('accuracy.png', dpi=300)
    plt.close()
