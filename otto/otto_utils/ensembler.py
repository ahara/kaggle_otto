import csv
import numpy as np
import os

import blender
import consts
import utils


if __name__ == '__main__':
    weights = blender.get_weights()
    prediction_files = utils.get_prediction_files()

    with open(os.path.join(consts.OUTPUT_PATH, 'ensembler_weighted_models.csv'), 'wb') as f_out:
        writer = csv.writer(f_out)
        readers = []
        f_ins = []
        for fpred in prediction_files:
            f_in = open(os.path.join(consts.ENSEMBLE_PATH, fpred), 'rb')
            f_ins.append(f_in)
            readers.append(csv.reader(f_in))
        # Copy header
        writer.writerow(readers[0].next())
        for r in readers[1:]:
            r.next()
        # Merge content
        for line in readers[0]:
            file_name = line[0]
            preds = weights[0] * np.array(map(float, line[1:]))
            for i, r in enumerate(readers[1:]):
                preds += weights[i+1] * np.array(map(float, r.next()[1:]))
            preds /= np.sum(weights)
            writer.writerow([file_name] + list(preds))
        # Close files
        for f_in in f_ins:
            f_in.close()
