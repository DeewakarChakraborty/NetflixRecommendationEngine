from model import RecSys
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def trainRecSys(Label_User_ID,Label_Movie_ID,train_loader,valid_loader):
  model = RecSys(num_users = len(Label_User_ID.classes_), num_movies = len(Label_Movie_ID.classes_)).to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  n_total_steps = len(train_loader)

  for epoch in range(5):
    for i, (features, labels) in enumerate(train_loader):
      features = features.to(device)
      labels = labels.to(device)
      output = model(features[:,0],features[:,1],labels[:])
      loss = criterion(output, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if (i+1) % 1000 == 0:
            print (f'Epoch [{epoch+1}/{5}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    
    with torch.no_grad():
      n_correct = 0
      n_samples = 0
      for features, labels in valid_loader:
        features = features.to(device)
        labels = labels.to(device)
        output = model(features[:,0],features[:,1],labels[:])
        _, predicted = torch.max(output.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
      acc = 100.0 * n_correct / n_samples
      print(f'Accuracy: {acc} %')