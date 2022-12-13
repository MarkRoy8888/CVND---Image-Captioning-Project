import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()

#         resnet18 = models.resnet18(pretrained=True)
#         alexnet = models.alexnet(pretrained=True)
#         squeezenet = models.squeezenet1_0(pretrained=True)
#         vgg16 = models.vgg16(pretrained=True)
#         densenet = models.densenet161(pretrained=True)
#         inception = models.inception_v3(pretrained=True)
#         googlenet = models.googlenet(pretrained=True)
#         shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
#         mobilenet_v2 = models.mobilenet_v2(pretrained=True)
#         mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
#         mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
#         resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
#         wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
#         mnasnet = models.mnasnet1_0(pretrained=True)
        # TODO  select model 
        use_model = models.googlenet(pretrained=True)
        for param in use_model.parameters():
            param.requires_grad_(False) # True  False 
        
        modules = list(use_model.children())[:-1]
        self.use_model = nn.Sequential(*modules)
        self.embed = nn.Linear(use_model.fc.in_features, embed_size)

    def forward(self, images):
        features = self.use_model(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#         pass
    
#     def forward(self, features, captions):
#         pass
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.word_embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM( input_size = embed_size, 
                             hidden_size = hidden_size, 
                             num_layers = num_layers, 
                             dropout = 0.2, 
                             batch_first=True
                           )
        
        self.linear_fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        captions = captions[:, :-1] 
        captions = self.word_embedding_layer(captions)
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)
        outputs, _ = self.lstm(inputs)
        outputs = self.linear_fc(outputs)
        
        return outputs
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass
    
