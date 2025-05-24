#导入需要的库
import torch
import torchvision.models as models

#保存权重和模型
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

#加载模型和权重
model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()



#保存和加载训练好的模型
torch.save(model, 'model.pth')
model = torch.load('model.pth', weights_only=False)




