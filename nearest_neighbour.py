import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def learnknn(k: int, x_train: np.array, y_train: np.array):
    def knn_predict(x_test: np.array):
        preds = []
        for i in range(len(x_test)):
            distances = np.linalg.norm(x_train - x_test[i], axis=1)
            k_indices = np.argsort(distances)[:k]
            k_nearest_labels = y_train[k_indices]
            values, counts = np.unique(k_nearest_labels, return_counts=True)
            predicted_label = values[np.argmax(counts)]
            preds.append(predicted_label)
        return np.array(preds).reshape(-1, 1)
    return knn_predict
def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """

    preds = classifier(x_test)
    return np.array(preds).reshape(-1, 1)

def test_knn():
    # Create a simple training set
    x_train = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y_train = np.array([0, 0, 1, 1])

    # Create test samples
    x_test = np.array([[0.9, 0.9], [0.1, 0.2], [0.2, 0.8]])
    
    # Train and predict
    classifier = learnknn(1, x_train, y_train)
    preds = predictknn(classifier, x_test)
    
    print("Predictions:", preds.flatten())
    # Expected: [0, 1, 1] or similar, depending on k and tie-breaking


def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")

def knn_experiment(sample_sizes, repeats=10):
    avg_errors = []
    min_errors = []
    max_errors = []

    data = np.load('mnist_all.npz')
    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train_full, y_train_full = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)
    for m in sample_sizes:
        errors = []
        for _ in range(repeats):
            x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], m)
            classifier = learnknn(1, x_train, y_train)
            y_pred=predictknn(classifier, x_test).flatten()
            error = np.mean(y_test != y_pred)
            errors.append(error)
        avg_errors.append(np.mean(errors))
        min_errors.append(np.min(errors))
        max_errors.append(np.max(errors))

    plt.errorbar(sample_sizes, avg_errors, yerr=[np.array(avg_errors)-np.array(min_errors), np.array(max_errors)-np.array(avg_errors)], fmt='-o')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Average Test Error')
    plt.title('k-NN (k=1) Test Error vs Training Size')
    plt.ylim(0, 1)
    plt.show()

def knn_experiment_30_runs(sample_sizes, repeats=30):
    data = np.load('mnist_all.npz')
    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']
    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']
    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 200)

    for m in sample_sizes:
        avg_errors = []
        for k in range(1, 16):
            errors = []
            for _ in range(repeats):
                x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], m)
                classifier = learnknn(k, x_train, y_train)
                y_pred = predictknn(classifier, x_test).flatten()
                error = np.mean(y_test != y_pred)
                errors.append(error)
            avg_errors.append(np.mean(errors))
        plt.figure()
        plt.plot(range(1, 16), avg_errors, marker='o')
        plt.xlabel('k')
        plt.ylabel('Average Test Error')
        plt.title(f'k-NN Test Error vs k (Training Size={m})')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.show()
        optimal_k = 1 + avg_errors.index(min(avg_errors))
        print(f"Optimal k for m={m}: {optimal_k} (error={min(avg_errors):.3f})")


def knn_experiment_30_runs_corrupted_labels(sample_sizes, repeats=30):
    data = np.load('mnist_all.npz')
    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']
    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']
    x_test_orig, y_test_orig = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 200)

    def corrupt_labels(y, num_classes=4, corruption_ratio=0.3):
        y_corrupt = y.copy()
        n = len(y_corrupt)
        num_corrupt = int(corruption_ratio * n)
        corrupt_indices = np.random.choice(n, num_corrupt, replace=False)
        for idx in corrupt_indices:
            current_label = y_corrupt[idx]
            other_labels = [l for l in range(num_classes) if l != current_label]
            y_corrupt[idx] = np.random.choice(other_labels)
        return y_corrupt

    for m in sample_sizes:
        avg_errors = []
        for k in range(1, 16):
            errors = []
            for _ in range(repeats):
                x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], m)
                y_train_corrupt = corrupt_labels(y_train)
                y_test_corrupt = corrupt_labels(y_test_orig)
                classifier = learnknn(k, x_train, y_train_corrupt)
                y_pred = predictknn(classifier, x_test_orig).flatten()
                error = np.mean(y_test_corrupt != y_pred)
                errors.append(error)
            avg_errors.append(np.mean(errors))
        plt.figure()
        plt.plot(range(1, 16), avg_errors, marker='o')
        plt.xlabel('k')
        plt.ylabel('Average Test Error')
        plt.title(f'k-NN Test Error vs k (Training Size={m}, 30% labels corrupted)')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.show()
        optimal_k = 1 + avg_errors.index(min(avg_errors))
        print(f"Optimal k for m={m} (corrupted labels): {optimal_k} (error={min(avg_errors):.3f})")
if __name__ == '__main__':

    # before submitting, make sure that the function simple_test runs without errors
    #simple_test()
    #test_knn()

    
    #knn_experiment(sample_sizes=[10, 20, 35, 50, 75, 100])
    #knn_experiment_30_runs(sample_sizes=[50, 150, 500], repeats=30)
    knn_experiment_30_runs_corrupted_labels(sample_sizes=[50, 150, 500], repeats=30)


