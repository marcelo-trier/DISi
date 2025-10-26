
import numpy as np
from deslib.des import KNORAU

class DISi(KNORAU):
    # valores estaticos..
    propUL = 0.27
    ulsize = int(propUL * 100)
    n_neighbors = 7
    alphaK = 2 * n_neighbors


    def get_competence_region(self, query, n_neighbors=None):
        n_neighbors = n_neighbors or DISi.n_neighbors

        # captura vizinhos mais próximos ( 2 * K ) ==> 14
        dists_orig, rocks_orig = super().get_competence_region(query, DISi.alphaK)
        dists_orig = np.atleast_2d(dists_orig)
        rocks_orig = np.atleast_2d(rocks_orig)

        # define estrutura de resultados..
        di_rocks = []
        di_dists=[]

        # perform discriminant-index... one by one..
        for idx, (dist, irock_orig) in enumerate(zip(dists_orig, rocks_orig)):
            divalues = self.DIVALUES(irock_orig)

            #irock = self.irock_bestDI(divalues, n_neighbors)
            isort = np.argsort(-divalues)  # descendent order..
            irock = isort[ :n_neighbors]

            di_rocks.append(irock_orig[irock])
            di_dists.append(dist[irock])

        di_dists = np.array(di_dists)
        di_rocks = np.array(di_rocks)
        return (di_dists, di_rocks)


    # rocklen -> numero de instancias coletadas para estimar a competencia 
    #           dos classificadores, para depois dividi-los em 2 grupos: U-L
    def DIVALUES(self, irock):
        ulsize = DISi.ulsize
        rocklen = DISi.n_neighbors

        ytrue = self.DSEL_target_[irock]
        ypred = self.DSEL_processed_[ irock, : ]

        ytrue = ytrue.reshape((1,-1))

        HITS = ypred.T == ytrue

        # TODO: calculo do score knn=7, mas poderia ser knn=14
        # TODO: calculo do score eh acuracia (numero acertos), mas poderia ser F1
        scoore = np.count_nonzero(HITS[:, :rocklen], axis=1)
        isort = np.argsort(scoore)

        # getting n_best and n_worst
        iupper = isort[::-1][:ulsize]
        ilower = isort[:ulsize]

        # acc for upper and lower group
        upper = HITS[iupper]
        lower = HITS[ilower]

        # estimating DIScrimate index
        upper = np.count_nonzero(upper, axis=0)
        lower = np.count_nonzero(lower, axis=0)
        DI = upper-lower

        return DI
