import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms
from matplotlib import pyplot as plt
from torch import nn
from PIL import Image
import os
import torch.optim as optim
import random

# 用GPU训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def data_ready(Path: str = "datasets"):
    # 将图像转换为28*28, 张量并进行归一化
    transform = transforms.Compose(
        [transforms.Resize((28, 28)),
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )

    # 下载MNIST训练集
    mnist_train_dataset = torchvision.datasets.MNIST(
        root=Path, train=True, download=True, transform=transform)
    # 下载MNIST测试集
    mnist_test_dataset = torchvision.datasets.MNIST(
        root=Path, train=False, download=True, transform=transform)
    # 设置随机种子
    random.seed(0)
    # 生成随机索引并获取与额外训练集相近大小的测试集
    indices = random.sample(range(len(mnist_train_dataset)), 19000)
    mnist_train_dataset = torch.utils.data.Subset(mnist_train_dataset, indices)
    #额外的训练集和测试集
    extra_dataset_path = r'datasets\EXTRA2'
    extra_dataset = extradata(root_dir=extra_dataset_path,transform=transform)
    #划分测试集和训练集
    train_size = int(0.9*len(extra_dataset))
    test_size = len(extra_dataset)-train_size
    extra_train_dataset, extra_test_dataset = random_split(extra_dataset,[train_size,test_size])
    #合并
    combined_train_dataset = ConcatDataset([mnist_train_dataset, extra_train_dataset])
    combined_test_dataset = ConcatDataset([mnist_test_dataset, extra_test_dataset])
    #测试
    #print(len(combined_train_dataset))
    #print(len(combined_test_dataset))
    # 加载训练集
    train_loader = DataLoader(dataset=combined_train_dataset, batch_size=64, shuffle=True)
    # 加载测试集
    test_loader = DataLoader(dataset=combined_test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

class extradata(Dataset): #从kaggle上额外寻找的数据集，约有21k图片
    #初始化
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
    #添加数据
        for label in range(10):
            folder_path = os.path.join(root_dir,str(label))
            for filename in os.listdir(folder_path):
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    self.images.append(os.path.join(folder_path,filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)
    #定义索引
    def __getitem__(self, item):
        image_path = self.images[item]
        image = Image.open(image_path).convert('L')
        label = self.labels[item]

        if self.transform:
            image = self.transform(image)

        return image, label
class improved_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),#dropout防止过拟合
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), #dropout防止过拟合
            nn.Linear(512, num_classes)
        )

    # 前向传播
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_model(train_data_loader,model,optimizer,loss_fn):
    datanum = len(train_data_loader.dataset)
    batch_num = len(train_data_loader)
    train_loss = 0
    train_accucary = 0
    for x,y in train_data_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_accucary = train_accucary+(pred.argmax(1) == y).type(torch.float32).sum().item()
        train_loss = train_loss+loss.item()

    train_accucary = train_accucary/datanum
    train_loss = train_loss/batch_num
    return train_loss, train_accucary

def run_train(Path,epochs,is_save=False,is_eval=True, lr=0.001):
    train_loader, test_loader = data_ready(Path)
    model = improved_Model()
    model = model.to(device)
    lossf = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        train_loss, train_accuracy = train_model(train_loader, model, optimizer, lossf)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy * 100:.3f}% ")
    if is_save:
        save_path = 'models'
        model_name = get_next_it(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, model_name))
        print(f"Model saved as {model_name}")
    if is_eval:
        average_loss,test_accuracy = eval_model(model,test_loader)
        print(f"Test Loss:{average_loss},Accuracy:{test_accuracy*100:.3f}% ")
    return model

def eval_model(model, test_loader, loss_fn=torch.nn.CrossEntropyLoss()):
    model.eval()
    total_loss = 0
    total = 0
    correct = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()

            _, predicted = torch.max(pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    average_loss = total_loss / len(test_loader)
    accuracy = correct / total

    return average_loss, accuracy




def get_next_it(Path='models', base_name="model"):
    # 列出文件夹中所有文件
    existing_files = os.listdir(Path)
    # 筛选出符合格式的文件
    model_files = [f for f in existing_files if f.startswith(base_name) and f.endswith('.pth')]
    # 提取文件名中的数字部分
    model_numbers = []
    for model_file in model_files:
        try:
            number = int(model_file[len(base_name) + 1:-4])
            model_numbers.append(number)
        except ValueError:
            pass

    # 找到最大的序号
    if model_numbers:
        next_number = max(model_numbers) + 1
    else:
        next_number = 1  # 如果没有文件，使用1作为初始序号
    return f"{base_name}_{next_number}.pth"

def load_model(model_name,Path = 'models'):
    model = improved_Model()
    model_path = os.path.join(Path, model_name)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    return model

def pred_img(imgPath,model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    image = Image.open(imgPath).convert('L')
    image_tensor = transform(image.resize((28, 28)))
    #增加一个维度batch_size以符合标准
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    plt.imshow(image_tensor.cpu().squeeze().numpy(), cmap='gray')
    plt.show()
    with torch.no_grad():
        outputs = model(image_tensor)
    # 获取模型的预测结果
    _, result = torch.max(outputs, 1)
    print(f"识别结果: {result.item()}")

if __name__ == "__main__":
    num_classes = 10
    Path = 'datasets'
    mode = 1
    if mode ==1:
        # 从头训练模型
        model = run_train(Path, 10, is_save=True, lr=0.001)
    else:
        model = load_model(model_name='model_16.pth')
        _, test_loader = data_ready(Path)
        average_loss, accuracy = eval_model(model,test_loader)
        print(f"Test Loss:{average_loss:.5f},Accuracy:{accuracy * 100:.3f}% ")

    #加载已经训练好的模型
    img_path = r"my_wrriten_digits\88.jpg"
    pred_img(img_path, model)

