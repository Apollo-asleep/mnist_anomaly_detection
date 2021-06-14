from torchvision import datasets, transforms, models
model = models.resnet152(pretrained=True)
print(model)
my_model = nn.Sequential(*list(model.modules())[:-1]) 