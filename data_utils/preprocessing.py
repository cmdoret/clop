import numpy as np
from matplotlib import pyplot as plt



def fcgr(seq: str, k=8):
    letter_to_num = {'A': 0,
                    'C': 1,
                    'G': 2,
                    'T': 3}

    letter_to_x = {'A': 0,
                    'C': 1,
                    'G': 0,
                    'T': 1}

    letter_to_y = {'A': 0,
                    'C': 0,
                    'G': 1,
                    'T': 1}

    IMGSIZE = 2**k
    img = np.zeros((IMGSIZE, IMGSIZE))

    substrs = [seq[i:i+k] for i in range(len(seq)-k+1)]

    for substr in substrs:
        x = 0
        y = 0
        for i, s in enumerate(substr):
            x = x + letter_to_x[s] * IMGSIZE/(2 ** (i+1))
            y = y + letter_to_y[s] * IMGSIZE / (2 ** (i+1))
        img[int(y), int(x)] += 1

    return img

if __name__ == '__main__':
    img = fcgr('ACG', k=3)
    print(img,)
    #plt.imshow(img)
    #plt.show()