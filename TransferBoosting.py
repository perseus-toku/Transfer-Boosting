import sklearn
import numpy as np
from sklearn import linear_model
from copy import  deepcopy



class TransferBoost():
    def __init__(self):
        """
        initialize an object of TransferBoost
        """
        self.MODEL =[]
        self.BETA =[]

    def Learner(self,X,y,W):
        model = linear_model.LogisticRegression(C=sum(W))
        model.fit(X,y)
        return model


    def fit(self, sources, sources_y, target, target_y, M=10):
        """
        sources is a list of list of training samples
        :return:
        """
        num_sources = len(sources)
        lengths =[]
        for source in sources:
            lengths.append(len(source))
        lengths.append(len(target))
        sources = np.asarray([ x for s in deepcopy(sources) for x in s])
        sources_y = np.asarray([ x for s in deepcopy(sources_y) for x in s])
        target = np.asarray(deepcopy(target))
        target_y = np.asarray(deepcopy(target_y))
        #X is the combiend training set including target , y is the combined y including target
        X = np.concatenate((sources,target), axis=0)
        y = np.concatenate((sources_y,target_y),axis= 0)
        d =  X.shape[0]
        W = np.ones(d)/d
        y[y==0] = -1
        target_y[target_y==-0] = -1

        print(lengths)
        print("num sources",num_sources)
        for t in range(M):
            print("\n-------------iteration {}---------------".format(t+1))
            ALPHA =[]
            model = self.Learner(X,y,W)
            preds = model.predict(X)
            t_W = W[-lengths[-1] :]
            model_target = self.Learner(target,target_y, t_W)
            target_preds = model_target.predict(target)
            target_error = np.dot(t_W, [int(x) for x in target_preds==target_y] )/ np.sum(np.abs(t_W))
            print(np.sum( [int(x) for x in target_preds==target_y]))
            print(target_error)
            print(np.sum(np.abs(t_W)))
            print(np.dot(t_W, [int(x) for x in target_preds==target_y] ))

            for i in range(num_sources):
                alpha = self.return_alpha(X, y,i, target, target_y, W, lengths, target_error )
                ALPHA.append(alpha)
            beta = 0.5 * np.log((1+ np.sum(np.multiply( W, np.multiply(preds,y))) ) / (1-np.sum(np.multiply( W, np.multiply(preds,y)))))
            final_alpha =np.asarray([])
            for i in range(num_sources):
                final_alpha = np.concatenate( (final_alpha,ALPHA[i] * np.ones(lengths[i])), axis=0)
            final_alpha = np.concatenate( (final_alpha, np.zeros(len(target))), axis=0)
            print(len(final_alpha))
            print(final_alpha)
            print("beta", beta)
            print("round preds",preds[-10:])
            print("true labels,",y[-10:])
            print("round weights", W[-10:])
            Z = np.dot( W, np.exp(np.add(-beta * np.multiply(preds,y),final_alpha)))
            W = np.multiply(W, np.exp(np.add(-beta * np.multiply(preds,y),final_alpha)))/Z
            print("weights updated", W[-10:])

            self.MODEL.append(model)
            self.BETA.append(beta)

    def return_alpha(self, X,y, ind, T,ty, W, lengths, t_error):
        """"""
        r = self.ind_to_range(ind,lengths)
        print(r)
        s_X = X[r[0]:r[1]]
        sy = y[r[0]:r[1]]
        s_W = W[r[0]:r[1]]
        t_W = W[-lengths[-1]:]
        st_X = np.concatenate((s_X , T),axis=0)
        st_y = np.concatenate((sy,ty),axis=0)
        st_W = np.concatenate((s_W , t_W),axis=0)
        model = self.Learner(st_X,st_y,st_W)
        preds = model.predict(T)
        error = np.dot(t_W, [int(x) for x in preds==ty] )/ np.sum(np.abs(t_W))
        alpha = t_error - error
        print("t_error value ", t_error)
        print("alpha value ", alpha)
        return alpha



    def ind_to_range(self,ind,lengths):
        """
        return the range of the particular source based on the index
        sources are indexed 0, 1, 2 ... to target
        :param ind:
        :param lengths:
        :return:
        """
        if ind >= len(lengths):
            raise IndexError("index out of range")
        elif ind ==0:
            return (0,lengths[0])
        elif ind == len(lengths)-1:
            return (sum(lengths[:-1]),sum(lengths))
        else:
            return (sum(lengths[:ind]),sum(lengths[:ind]) +lengths[ind] )



    def predict(self,test):
        """"""
        if self.MODEL ==[] or self.BETA == []:
            raise ValueError("first fit a model ")
        test = np.asarray(test)
        n = test.shape[0]
        preds = np.zeros(n)
        for i in range (len(self.MODEL)):
            beta = self.BETA[i]
            model = self.MODEL[i]
            cur_preds =  beta* model.predict(test)
            preds = np.add(preds, cur_preds)
        preds = np.sign(preds)
        return preds


    def predict_and_evaluate(self,test,y):
        preds = self.predict(test)
        label = deepcopy(y)
        label = np.asarray(label)
        label[label == 0] = -1
        c = np.sum([x for x in preds == label])
        c = c / len(preds)
        error = 1 - c
        c *= 100
        error *= 100
        c = round(c, 3)
        error = round(error, 3)
        print(
            "\n-----TEST INFORMATION-----: \nnumber of test sample :{} \ncorrect rate is :{}%  \nerror is: {}% \n----------------------".format(
                len(preds), c, error))
        preds = np.asarray(preds)
        preds[preds==-1] = 0
        return preds


    def basic_model_evaluate(self,sources, sources_y, target, target_y, test,test_y):
        """the result achieved on basic model
            Default basic model: Logistic Regression
        """
        sources = np.asarray([ x for s in deepcopy(sources) for x in s])
        sources_y = np.asarray([ x for s in deepcopy(sources_y) for x in s])
        target = np.asarray(deepcopy(target))
        target_y = np.asarray(deepcopy(target_y))
        X = np.concatenate((sources,target), axis=0)
        y = np.concatenate((sources_y,target_y),axis= 0)

        model = linear_model.LogisticRegression()
        model.fit(X,y)
        preds = model.predict(test)
        c = np.sum([x for x in preds == test_y])
        c = round(c/len(preds),6)
        c *= 100
        print("\n-----base model correct rate :{}-----".format(c))









