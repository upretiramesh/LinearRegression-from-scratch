import numpy as np


class PolynomialFeature:
    def __init__(self, degree=2, interaction_only=False):
        self.degree = degree
        self.interaction = interaction_only

    def transform(self, data=None):
        if data is not None:

            poly_x = data[:]

            if not self.interaction:
                for deg in range(2, self.degree + 1):
                    new_val = data ** deg
                    poly_x = np.hstack((poly_x, new_val))
                    poly_x = np.unique(poly_x, axis=1)

            for deg in range(2, self.degree + 1):
                if deg == 2:
                    new_val = np.hstack(
                        [np.expand_dims(data[:, i], axis=1) * data[:, i + 1:] for i in range(data.shape[1] - 1)])
                    poly_x = np.hstack((poly_x, new_val))
                    poly_x = np.unique(poly_x, axis=1)
                    # print('degree:', deg, 'shape:', poly_x.shape)
                else:
                    new_val = np.unique(
                        np.hstack(
                            [np.expand_dims(data[:, i], axis=1) * new_val[:, i + 1:] for i in range(data.shape[1])]),
                        axis=1)
                    poly_x = np.hstack((poly_x, new_val))
                    poly_x = np.unique(poly_x, axis=1)
                    # print('degree:', deg, 'shape:', poly_x.shape)
            return poly_x
        else:
            print('Argument missing: pass the data to transform into polynomial form')
