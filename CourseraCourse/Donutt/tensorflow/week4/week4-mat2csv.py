import scipy.io
import pandas as pd
import numpy as np

basic_dir = 'files/'
filename = 'ex8_movieParams'
MAT_FILE_FORMAT = '.mat'
CSV_FILE_FORMAT = '.csv'

mat = scipy.io.loadmat(basic_dir + filename + MAT_FILE_FORMAT)
mat = {k:v for k, v in mat.items() if k[0] != '_'}

for k, v in mat.iteritems():
    _v = np.array(v)

    print 'export ==> ' + k + " ..."
    print _v.shape
    save_filename = basic_dir + filename + "_" + k + CSV_FILE_FORMAT

    np.savetxt(save_filename, _v, delimiter=',')
    print 'done'
