"""RecSys Models"""

import lzma
import pickle

from service.settings import get_config


class Range:
    """Class for predict Range model"""

    def __init__(self, N_recs: int = 10):
        self.N_recs = N_recs

    def predict(self):
        return list(range(self.N_recs))


class Popular:
    """Class for predict Range model"""

    def __init__(self, N_recs: int = 10):
        self.N_recs = N_recs

        with open("./service/recmodels_folder/popular_answer.pkl", "rb") as file:
            self.answer = pickle.load(file)

    def predict(self) -> list:
        return self.answer


class userKNN:
    """Class for predict Range model"""

    def __init__(self, N_recs: int = 10):
        self.N_recs = N_recs

        with lzma.open("./service/recmodels_folder/user_knn.xz", "rb") as file:
            self.user_knn = pickle.load(file)

        self.popular_model = Popular(self.N_recs)

    def predict(self, user_id: int) -> list:
        if user_id in self.user_knn.users_mapping:
            reco = self.user_knn.eval(user_id, N_recs=self.N_recs).item_id.to_list()
            if len(reco) < self.N_recs:
                reco_popular = self.popular_model.predict()
                reco += [item for item in reco_popular if item not in reco][: self.N_recs - len(reco)]
        else:
            reco = self.popular_model.predict()
        return reco


app_config = get_config()

simple_range = Range(N_recs=app_config.k_recs)
popular = Popular(N_recs=app_config.k_recs)
user_knn = userKNN(N_recs=app_config.k_recs)
