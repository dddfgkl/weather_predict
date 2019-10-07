import os
import h5py
import numpy as np
import read_ctl

# bin_file_path = "/home/datanfs/macong_data/IAP41_Hindcast_SEasian_daily_Tmax_ens_mean_87x54x180x32.bin"

def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)


def extract_data_from_h5(cpc_data, cpc_longitude, cpc_latitude):
    # cpc_data and bin_data should be trim
    # the core of the extraction, find the axis
    """
    :param cpc_data: 720x160x365x32(longitude x latitude x day x year)
    # :param bin_data: 87x54x180x32(longitude x latitude x day x year)
    :param cpc_longitude: longitude 720
    :param cpc_latitude: latitude 87
    :return:
    """
    cpc_x = 0
    cpc_y = 0

    longitude = 59.0625 - 1.40625
    latitude = -14.88189 - 1.417
    extract_data = []
    for i in range(87):
        longitude += 1.40625
        latitude_array = []
        for j in range(54):
            latitude += 1.417
            while cpc_x < len(cpc_longitude) and cpc_longitude[cpc_x] < longitude:
                cpc_x += 1
            while cpc_y < len(cpc_latitude) and cpc_latitude[cpc_y] < latitude:
                cpc_y += 1
            d_array = []
            for d in range(180):
                y_array = []
                for y in range(32):
                    points = ((cpc_longitude[cpc_x-1], cpc_latitude[cpc_y-1], cpc_data[cpc_x-1][cpc_y-1][d][y]),
                              (cpc_longitude[cpc_x-1], cpc_latitude[cpc_y], cpc_data[cpc_x-1][cpc_y][d][y]),
                              (cpc_longitude[cpc_x], cpc_latitude[cpc_y-1], cpc_data[cpc_x][cpc_y-1][d][y]),
                              (cpc_longitude[cpc_x], cpc_latitude[cpc_y], cpc_data[cpc_x][cpc_y][d][y]))
                    inter_value = bilinear_interpolation(longitude, latitude, points)
                    y_array.append(inter_value)
                d_array.append(y_array)
            latitude_array.append(d_array)
        extract_data.append(latitude_array)
    extract_data = np.array(extract_data)
    extract_data = extract_data.transpose(3, 2, 1, 0)
    return extract_data


def read_data_from_file(h5_file_path, store_file_path):
    # cpc row h5 file
    if os.path.exists(h5_file_path) == False:
        raise FileExistsError
    f = h5py.File(h5_file_path, 'r')
    f_store = h5py.File(store_file_path, 'w')
    print(f.keys())
    for key in f.keys():
        print(key, f[key].shape)
    cpc_data = f["tmax"]
    cpc_longitude = f["lon"]
    cpc_latitude = f["lat"]
    extract_data = extract_data_from_h5(cpc_data, cpc_longitude, cpc_latitude)
    f_store["cpc"] = extract_data
    f_store.close()
    print("desc over all keys")

    # bin label data
    """
    bin_data = read_ctl.read_bin_to_numpy()
    print("data shape, ", bin_data.shape)
    """

if __name__ == '__main__':
    extract_data_from_h5()