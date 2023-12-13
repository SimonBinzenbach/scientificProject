# Description: This file contains the functions that are used to create the feature maps
def verticalEdgeMap(image):
    # create a matrix of zeros with the size of the image
    matrix = np.zeros((image.shape[0], image.shape[1]))
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1] - 1):
            matrix[x][y] = np.max(image[x, y]) - np.max(image[x, y + 1])
    return matrix

