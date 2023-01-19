import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FGSM :
    def __init__(self,model,loss,eps=0.031):
        self.model=model
        self.loss=loss

        self.eps=eps

    def attack(self,images,labels):
        images=images.to(device)
        labels=labels.to(device)
        images.requires_grad = True
        outputs = self.model(images)
        self.model.zero_grad()
        cost = self.loss(outputs,labels).to(device)
        cost.backward()
        attack_images = images + self.eps * images.grad.sign()
        return attack_images

