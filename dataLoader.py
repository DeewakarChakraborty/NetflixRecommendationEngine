import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from dataset import RatingDataset
from train import trainRecSys

df = pd.read_csv('/data/Netflix_Dataset_Rating.csv')

Label_User_ID = preprocessing.LabelEncoder()
Label_Movie_ID = preprocessing.LabelEncoder()

df.User_ID = Label_User_ID.fit_transform(df.User_ID.values)
df.Movie_ID = Label_Movie_ID.fit_transform(df.Movie_ID.values)
  
df_train, df_valid = train_test_split(df, 
                                      test_size=0.2, 
                                      random_state=42, 
                                      stratify=df.Rating.values)
  
train_dataset = RatingDataset(User_ID = df_train.User_ID.values,
                              Movie_ID = df_train.Movie_ID.values,
                              Rating = (df_train.Rating.values-1))
  
valid_dataset = RatingDataset(User_ID = df_valid.User_ID.values,
                              Movie_ID = df_valid.Movie_ID.values,
                              Rating = (df_valid.Rating.values-1))
  
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=1024, 
                                            shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, 
                                            batch_size=1024, 
                                            shuffle=False)

trainRecSys(
    Label_User_ID=Label_User_ID,
    Label_Movie_ID=Label_Movie_ID,
    train_loader=train_loader,
    valid_loader=valid_loader,    
)