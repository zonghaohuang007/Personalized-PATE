import numpy as np
import pickle
from scipy.stats import truncnorm
import matplotlib.pyplot as plt


def uniform_privacy(a, b, n):
    return np.random.uniform(a, b, n)


def skewed_privacy(a, b, loc, scale, n):
    return truncnorm.rvs(a, b, loc=loc, scale=scale, size=n)


def constant_privacy(a, n):
    return np.ones(n)*a


if __name__ == '__main__':
    print('==> generate privacy preference')
    p1 = uniform_privacy(1.0, 10.0, 60000)
    mean = 5.0
    scale = 2.5
    a = (1.0 - mean) / scale
    b = (10.0 - mean) / scale
    p2 = skewed_privacy(a, b, loc=mean, scale=scale, n=60000)
    mean = 3.0
    scale = 2.5
    a = (1.0 - mean) / scale
    b = (10.0 - mean) / scale
    p3 = skewed_privacy(a, b, loc=mean, scale=scale, n=60000)
    mean = 8.0
    scale = 2.5
    a = (1.0 - mean) / scale
    b = (10.0 - mean) / scale
    p4 = skewed_privacy(a, b, loc=mean, scale=scale, n=60000)

    p5 = constant_privacy(a=5, n=60000)

    fig, ax = plt.subplots(2, 2, sharex='all')
    fig.suptitle('Privacy Preference Distribution')
    # fig.subxlabel('Privacy Budget')
    # fig.subylabel('Number')
    ax[0][0].hist(p1)
    ax[0][0].set_title(f"Uniform (p1, [1.0, 10.0])")
    ax[0][1].hist(p2)
    ax[0][1].set_title(f"Skewed Normal (p2, mean=5.0)")
    ax[1][0].hist(p3)
    ax[1][0].set_title(f"Skewed Normal (p3, mean=2.0)")
    ax[1][1].hist(p4)
    ax[1][1].set_title(f"Skewed Normal (p4, mean=8.0)")
    fig.show()
    fig.savefig('./data/privacy.png')

    with open('./data/p1.pickle', 'wb') as f:
        pickle.dump(p1, f)
    with open('./data/p2.pickle', 'wb') as f:
        pickle.dump(p2, f)
    with open('./data/p3.pickle', 'wb') as f:
        pickle.dump(p3, f)
    with open('./data/p4.pickle', 'wb') as f:
        pickle.dump(p4, f)
    with open('./data/p5.pickle', 'wb') as f:
        pickle.dump(p5, f)
    print('==> finished.')
