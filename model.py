import torch
import torch.nn as nn

class RecSys(nn.Module):
  def __init__(self, num_users, num_movies):
    super(RecSys, self).__init__()
    self.user_embed = nn.Embedding(num_users, 32)
    self.movie_embed = nn.Embedding(num_movies, 32)
    self.l1 = nn.Linear(64, 128)
    self.l2 = nn.Linear(128,128)
    self.out = nn.Linear(128,5)
    self.relu = nn.ReLU()

  def forward(self, User_ID, Movie_ID, Rating):
    user_embeds = self.user_embed(User_ID)
    movie_embeds = self.movie_embed(Movie_ID)
    input_layer = torch.cat([user_embeds,movie_embeds], dim=1)
    middle_1 = self.relu(self.l1(input_layer))
    middle_2 = self.relu(self.l2(middle_1))
    output = self.out(middle_2)
    return output