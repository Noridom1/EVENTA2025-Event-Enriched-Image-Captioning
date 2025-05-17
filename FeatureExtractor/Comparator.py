import numpy as np
import torch.nn.functional as F
import torch

class Comparator:
    def __init__(self):
        """
        Initialize with a FeatureExtractor object.
        """
        pass

    def cosine_similarity(self, feature1, feature2):
        """
        Compute cosine similarity between two image paths.
        """

        # Convert to torch tensors for cosine similarity
        vec1 = torch.tensor(feature1)
        vec2 = torch.tensor(feature2)

        # Compute cosine similarity
        similarity = F.cosine_similarity(vec1, vec2).item()
        return similarity

