import numpy as np
import glob
import util


class CarData:
    def __init__(self, list, is_car):
        self.len = len(list)
        self.list = list
        self.is_car = is_car
        first = list[0]
        img = util.imread(first)
        self.image_shape = img.shape
        self.image_data_type = img.dtype

    def sample(self):
        index = np.random.randint(0, self.len)
        return self.list[index]

    def sample_image(self):
        return util.imread(self.sample())


def get_car_data():
    cars = glob.glob('train_images/vehicles/*/*.png')
    notcars = glob.glob('train_images/non-vehicles/*/*.png')

    car_data = CarData(cars, True)
    not_car_data = CarData(notcars, False)

    return car_data, not_car_data


def get_test_paths():
    res = []
    for i in range(6):
        res.append('./test_images/test{}.jpg'.format(i + 1))

    return res


def get_video_paths():
    res = ["test_video.mp4", "project_video.mp4"]
    return res


class ModelParams:
    def __init__(self):
        self.color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

        self.spatial_feat = True  # Spatial features on or off
        self.spatial_size = (64, 64)  # Spatial binning dimensions

        self.hist_feat = True  # Histogram features on or off
        self.hist_bins = 32  # Number of histogram bins
        self.hist_range = (0, 256)

        self.hog_feat = True  # HOG features on or off
        self.hog_orient = 9  # HOG orientations
        self.hog_pix_per_cell = 8  # HOG pixels per cell
        self.hog_cell_per_block = 2  # HOG cells per block
        self.hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

import pickle
_MODEL_SVC_PICKLE_PATH = "model_svc.p"


def save_model(svc, scaler, params: ModelParams):
    d = { "svc": svc, "scaler": scaler, "params": params }
    with open(_MODEL_SVC_PICKLE_PATH, "wb") as f:
        pickle.dump(d, f)


def load_model():
    d = {}
    with open(_MODEL_SVC_PICKLE_PATH, "rb") as f:
        d = pickle.load(f)
    svc = d["svc"]
    scaler = d["scaler"]
    params = d["params"]
    return svc, scaler, params
