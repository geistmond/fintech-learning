from sklearn.cluster import KMeans

import numpy as np

from api_keys import av_key
av_api_key = av_key

start = datetime.datetime(2010,1,1)
end = datetime.datetime(2020,1,1)

data = web.DataReader(company, "av-daily", start, end, api_key=av_api_key)

initial_centroids = np.zeroes((len()))

def get_clusters(df):
    df['cluster'] = KMeans(n_clusters=4, random_state=0, init=initial_centroids)