"""RecSys Models"""

import lzma
import pickle

import nmslib
from rectools.tools.ann import UserToItemAnnRecommender

from service.settings import get_config


class Range:
    """Class for predict Range model"""

    def __init__(self, N_recs: int = 10):
        self.N_recs = N_recs

    def predict(self):
        return list(range(self.N_recs))


class Popular:
    """Class for predict Popular model"""

    def __init__(self, N_recs: int = 10):
        self.N_recs = N_recs
        self.answer = None
        self.load_flag = False

    def load(self):
        if not self.load_flag:
            with open("./service/recmodels_folder/popular_answer.pkl", "rb") as file:
                self.answer = pickle.load(file)
            self.load_flag = True

    def predict(self) -> list:
        return self.answer


class userKNN:
    """Class for predict userKNN model"""

    def __init__(self, N_recs: int = 10):
        self.user_knn = None
        self.N_recs = N_recs
        self.load_flag = False
        self.popular_model = Popular(self.N_recs)

    def load(self):
        if not self.load_flag:
            with lzma.open("./service/recmodels_folder/user_knn.xz", "rb") as file:
                self.user_knn = pickle.load(file)
            self.popular_model.load()
            self.load_flag = True

    def predict(self, user_id: int) -> list:
        if user_id in self.user_knn.users_mapping:
            reco = self.user_knn.eval(user_id, N_recs=self.N_recs).item_id.to_list()
            if len(reco) < self.N_recs:
                reco_popular = self.popular_model.predict()
                reco += [item for item in reco_popular if item not in reco][: self.N_recs - len(reco)]
        else:
            reco = self.popular_model.predict()
        return reco


class ANN_ALS:
    """Class for predict ALS model"""

    def __init__(self, N_recs: int = 10):
        self.N_recs = N_recs
        self.als_wrapper = None
        self.ann = None
        self.item_id_map = None
        self.user_id_map = None
        self.item_vectors = None
        self.user_vectors = None
        self.load_flag = False
        self.popular_model = Popular(self.N_recs)

    def load(self):
        if not self.load_flag:
            with open("./service/recmodels_folder/als_wrapper.pkl", "rb") as file:
                self.als_wrapper = pickle.load(file)
            self.popular_model.load()

            self.user_vectors, self.item_vectors = self.als_wrapper.get_vectors()
            with open("./service/recmodels_folder/user_id_map.pkl", "rb") as file:
                self.user_id_map = pickle.load(file)
            with open("./service/recmodels_folder/item_id_map.pkl", "rb") as file:
                self.item_id_map = pickle.load(file)

            index_init_params = {"method": "hnsw", "space": "negdotprod", "data_type": nmslib.DataType.DENSE_VECTOR}
            self.ann = UserToItemAnnRecommender(
                user_vectors=self.user_vectors,
                item_vectors=self.item_vectors,
                user_id_map=self.user_id_map,
                item_id_map=self.item_id_map,
                index_init_params=index_init_params,
            )
            self.ann.index.loadIndex("./service/recmodels_folder/ann_index.pkl")
            self.load_flag = True

    def predict(self, user_id: int) -> list:
        if user_id in self.user_id_map.external_ids:
            reco = self.ann.get_item_list_for_user(user_id, top_n=self.N_recs).tolist()
        else:
            reco = self.popular_model.predict()
        return reco


app_config = get_config()

simple_range = Range(N_recs=app_config.k_recs)

popular = Popular(N_recs=app_config.k_recs)

user_knn = userKNN(N_recs=app_config.k_recs)

ann_als = ANN_ALS(N_recs=app_config.k_recs)
