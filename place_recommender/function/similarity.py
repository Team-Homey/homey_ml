import numpy as np
import pickle
import torch

class ComputeSimilarity():
    def __init__(self):
        self.data = pickle.load(open('models/places365_standard.pickle', 'rb'))
        self.data_val = list()
        self.data_key = list()
        self.similarity_scores = list()
        self.top_indexes = [0, 1, 2]
        
        for key in self.data.keys():
            for val in self.data[key]:
                self.data_val.append(val.reshape(-1, 2048))
                self.data_key.append(key)
    
    def find_top3_similar(self, input_imgs):
        tensors = [self.compute(x, self.data_val, metric='cosine') for x in input_imgs]
        concat_scores = torch.cat(tensors, dim=0)
        mean_scores = torch.mean(concat_scores, dim=0)
        top_values = sorted(mean_scores[:3], reverse=True)
        
        for i in range(3, len(mean_scores)):
            if mean_scores[i] > top_values[-1]:
                top_values[-1] = mean_scores[i]
                self.top_indexes[-1] = i
                for j in range(2, -1, -1):
                    if top_values[j] > top_values[j-1]:
                        top_values[j], top_values[j-1] = top_values[j-1], top_values[j]
                        self.top_indexes[j], self.top_indexes[j-1] = self.top_indexes[j-1], self.top_indexes[j]
                    else:
                        break

        return [self.data_val[i] for i in self.top_indexes]
            
    def compute(self, input_feature, features, metric='cosine'):
        if metric == 'cosine':
            input_feature = input_feature.cpu().numpy()
            features = torch.cat(features, dim=0).cpu().numpy()
            similarity_scores = np.dot(input_feature, features.T)
            # most_similar_index = np.argmax(similarity_scores)
            # most_similar_feature = features[most_similar_index]
        
        elif metric == 'euclidean':
            input_feature = input_feature.cpu().numpy()
            features = torch.cat(features, dim=0).cpu().numpy()
            similarity_scores = np.sum((features - input_feature) ** 2, axis=1)
            # most_similar_index = np.argmin(similarity_scores)
            # most_similar_feature = features[most_similar_index]
        
        return torch.Tensor(similarity_scores)
    