import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class PGD:
    def __init__(self,model,loss,eps=0.03, alpha=0.5, iters=20):
        self.model=model
        self.eps=eps
        self.alpha=alpha
        self.iters=iters
        self.loss=loss
    def attack(self,images,labels):
        images=images.to(device)
        labels=labels.to(device)
        ori_images = images.data
        for i in range(self.iters):
            images.requires_grad = True
            outputs = self.model(images)

            self.model.zero_grad()
            cost = self.loss(outputs, labels).to(device)
            cost.backward()

            adv_images = images + self.alpha * images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        return images

