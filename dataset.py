import torch

class RatingDataset:
  def __init__(self, User_ID, Movie_ID, Rating):
    self.User_ID = User_ID
    self.Movie_ID = Movie_ID
    self.Rating = Rating 

  def __len__(self):
    return len(self.User_ID)

  def __getitem__(self, item):
    features = torch.tensor([self.User_ID[item],self.Movie_ID[item]], dtype= torch.long)
    label = torch.tensor(self.Rating[item], dtype=torch.long)
    return features,label 
  