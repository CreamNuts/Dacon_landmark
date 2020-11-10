import torch
import torch.nn as nn
import torch.nn.functional as F

'''
main.py 에서 모델 정의하고
모델 넘겨받고 앙상블만 하는 파일.
'''

class Ensemble(nn.Module):
    def __init__(self, model_A, model_B, model_C, NUM_CLASSES):
        super(Ensemble, self).__init__()
        self.model_A = model_A
        self.model_B = model_B
        self.model_C = model_C

        #Remove Last Linear Layer
        self.model_A.fc = nn.Identity()
        self.model_B.fc = nn.Identity()
        self.model_C.fc = nn.Identity()

        #Create new classifier
        self.classifier = nn.Linear(2048, NUM_CLASSES)

    def forward(self, x):
        x1 = self.model_A(x.clone())
        x1 = x1.view(x1.size(0), -1)
        x2 = self.model_B(x.clone())
        x2 = x2.view(x2.size(0), -1)
        x3 = self.model_C(x)
        x3 = x3.view(x3.size(0), -1)
        x = torch.cat((x1, x2, x3), dim=1)

        output = self.classifier(F.relu(x))
        return output


