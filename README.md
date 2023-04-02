借助Pytorch实现数据的在线增广
——————————————————————————————————————————
-------------------------------------------------------------------------------------------------------------------------------------------------------------
✔引言
---------------------------------------------------------------------------------------------------------------------------------------------------------
数据增广是深度学习中常用的技巧之一，通过对原有图片进行一定程度的随机变化（如旋转、缩放、翻转、裁剪）生成一组新的图片，从而扩大训练集的规模，让数据集尽可能的多样化。

1.视角变化：通过旋转、缩放、翻转等变换，让模型更好地识别目标，避免出现只在固定视角下能够正确识别的情况。

2.照明变化：通过改变亮度、对比度等参数，模拟不同的光照条件，使模型对光照条件的变化更加鲁棒。

3.遮挡变化：在图片中添加遮挡物，使模型能够更好地处理遮挡和复杂背景。

4.物体形变：通过变形、扭曲等变换，让模型更好地处理物体的形变，提高对不同形状物体的识别能力。

5.噪声变化：通过加入噪声、模糊等变换，让模型更加鲁棒，能够处理图像中的噪声或模糊情况。

✔四大数据增广需要解决的视觉问题
---------------------------------------------------------------------------------------------------------------------------------------
1.水平翻转or垂直翻转要解决平移不变性

2.随机旋转----旋转不变性

3.随机裁切----尺寸不变性

4.随机色度变换----光照复杂性

✔数据增广训练模型
-----------------------------------------------------------------------------------------------------------------------------------------
首先创建并配置好一个虚拟环境从GitHub克隆classification-basic-sample-master项目后，打开PyCharm找到train.py文件进行操作

![image](https://user-images.githubusercontent.com/128702185/229296707-ea0154ec-20dc-4229-b577-b5d0dd6c114e.png)

👉发现transforms.RandomResizedCrop了吗，这是随机裁切的意思👈 

-------------------------------------------------------------------------------------------------------------------------
👌常见的数据增强操作包括：

1.随机水平翻转（Horizontal Flip）

2.随机垂直翻转（Vertical Flip）

3.随机旋转（Random Rotate）

4.随机剪裁（Random Crop）

5.随机缩放（Random Scale）

6.随机加噪声（Random Noise）

7.颜色变换（Color Jittering）

8.数据标准化（Normalization）

9.随机改变图片对比度、亮度、饱和度等（Random Brightness/Contrast/Saturation）

------------------------------------------------------------------------------------------------------------------------------------------------
下面给出通过PyTorch中的transforms实现随机旋转和水平翻转的样例代码，你可以根据需要对参数进行修改

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-10, 10), fill=(0,)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

其中，RandomHorizontalFlip表示随机水平翻转，RandomRotation表示随机旋转，degrees为旋转的角度范围，在这个例子中是-10~10度，fill表示填充颜色，这里使用0填充。p表示随机翻转的概率，这里是50%。

你需要在classification-basic-sample-master项目中创建一个data文件夹

修改好参数就可以进行测试训练了      ❗❗❗不光训练集需要修改，验证集也采用同样的增广❗❗❗

你应该得到以下的结果

![1  2](https://user-images.githubusercontent.com/128702185/229298959-2c999d37-d731-448e-b3a5-5c5600921f3f.png)

✔对单张图片进行图像增广
-----------------------------------------------------------------------------------------------------------------------------------------
在PyCharm中的classification-basic-sample-master项目中新建一个py文件，输入代码

    import torchvision.transforms as transforms
    from PIL import Image
    import matplotlib.pyplot as plt
    
    
    img = Image.open('data/train/train/15.jpg')
    img.show()
    augment = transforms.Compose(
      [
            # 随机水平翻转，概率50%
         transforms.RandomHorizontalFlip(p=0.5),
            # 随机旋转
           transforms.RandomRotation(degrees=(-10, 10)),
            # 随机裁切
         transforms.RandomResizedCrop(256)
        ]
    )

    for i in range(1,10):
        plt.subplot(3,3,i)
        plt.imshow(augment(img))
    plt.show()
    
如果需要别的增广操作，可以参考[数据增广训练模型]进行参数修改   

运行代码你会得到一个MATLAB的Figure窗口

![image](https://user-images.githubusercontent.com/128702185/229299586-90b1aaf5-30e7-4a93-9d12-68715b3688b9.png)

✔对cifar数据集进行数据增广
-----------------------------------------------------------------------------------------------------------------------------------------
在PyCharm中的classification-basic-sample-master项目中新建一个py文件，输入代码

    import torch
    import torchvision
    import torchvision.transforms as transforms

    # 定义数据增广
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),    # 随机水平翻转
        transforms.RandomRotation(15),        # 随机旋转
        transforms.ToTensor(),                # 转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])

    # 加载 CIFAR-10 数据集并进行数据增广
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform_train)

    # 保存增广后的数据
    torch.save(trainset, './data/cifar10_augmented.pt')
    
 这里使用了 Compose 函数将多个数据增广操作组合起来。可以根据需求添加或调整这些增广操作。最后使用 torch.save() 将增广后的数据集保存到本地。你需要提前创建一个 ./data 文件夹。   

你会得到以下结果

![捕获1111](https://user-images.githubusercontent.com/128702185/229334974-2dcdc2fa-038a-4fcd-be20-8f4e0a4f6d96.PNG)

完成训练后你会在你创建的data文件夹中看到两个文件，下载的cifar数据和完成增广后的数据

![image](https://user-images.githubusercontent.com/128702185/229334933-7814d3b4-ee98-47be-bbc7-e95da3e2eb23.png)

✔拓展
-----------------------------------------------------------------------------------------------------------------------------------------
.pt文件是pytorch中常用的模型保存、加载和传递的文件格式，保存的是模型对象和相关参数。可以通过torch.load()方法将.pt文件加载成pytorch的模型对象并进行后续的操作。

可以使用以下代码将.pt文件中的一张图片保存为.jpg文件：

    import torch

    # Load the tensor object from the .pt file
    tensor = torch.load('./data/cifar10_augmented.pt')[0][0]

    # Convert the tensor object to an RGB PIL image
    image = transforms.ToPILImage()(tensor)

    # Save the image as a .jpg file
    image.save('./data/image.jpg')
    
这个代码会将第一张图片从.pt文件中读取出来，并将其转换成PIL图片格式。然后使用PIL库将图片保存为.jpg格式

![image](https://user-images.githubusercontent.com/128702185/229335127-21497803-ee5a-4c26-bdf7-fc81b739908d.png)
