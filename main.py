import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch import nn
import torch.optim as optim

from textDetector import TextDetector
from waterDetector import WaterDetector
from backgroundDetect import BackDetector
from gan_detect_iris.main import EyeDetector


device = "cpu"
print(f"Using {device} device")

detector_num = 4

class MainDecider(nn.Module):
    def __init__(self):
        super().__init__()
        self.result = nn.Linear(detector_num, 1)
        self.act_result = nn.Sigmoid()

    def forward(self, inputs):
        x = self.act_result(self.result(inputs))
        return x

model = MainDecider()
detectors = [TextDetector(), WaterDetector(), BackDetector(), EyeDetector()]

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(folder_path, tf=False, accuracy=0.9):
    
    res = [detector.testFolder(folder_path) for detector in detectors]

    X = [[result[det][1] for det in range(detector_num)] for result in res]

    n_epochs = 100
    batch_size = 10
    
    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = [tf for i in range(batch_size)]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')

    with torch.no_grad():
        y_pred = model(X)

    accuracy = np.mean([float(int(y >= accuracy) == tf) for y in y_pred])
    print(f"{"True Images" if not tf else "False Images"} Accuracy: {accuracy}")

def test(folder_path, accuracy):
    res = [detector.testFolder(folder_path) for detector in detectors]

    X = [[result[det][1] for det in range(detector_num)] for result in res]

    with torch.no_grad():
        y_pred = model(X)
    
    accuracy = np.mean([float(int(y >= accuracy) == tf) for y in y_pred])
    print(f"Test Accuracy: {accuracy}")

train("fakeimages", True)
train("trueimages", False)
test("testimages")
