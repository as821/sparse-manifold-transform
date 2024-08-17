import os
import torch
import torchvision.transforms as transforms
from PIL import Image



class CLEVR(torch.utils.data.Dataset):
    # https://cs.stanford.edu/people/jcjohns/clevr/ --> no masks (current usage for now)
    # https://github.com/google-deepmind/multi_object_datasets?tab=readme-ov-file#clevr-with-masks --> has masks
    def __init__(self, root, transform=None, num_aug=1, train=True, rescale=False, download=False, downsample=1):
        super(CLEVR, self).__init__()

        # assert split in ['train', 'val', 'test']
        self.split = "train" if train else "test"
        self.root_dir = os.path.join(root, "clevr", self.split)
        self.files = os.listdir(self.root_dir)
        self.augmentation = None
        self.num_aug = num_aug
        self.downsample = downsample
        assert self.downsample >= 1
        if not rescale:
            self.img_transform = transforms.Compose([
                    transforms.ToTensor()])
        else:
            self.img_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5) # ([0,1]-0.5)/0.5=[-1,1]
                ])
        self.classes = []       # not a classification dataset
        self.data = [self.__getitem__(0)[0]]       # needed for consistent interface with other datasets...

    def __getitem__(self, index):
        # image resolution should be (64, 64)
        path = self.files[index]
        image = Image.open(os.path.join(self.root_dir, path)).convert("RGB")
        
        # NOTE: make a square image, then downsample by 2
        mn = min(image.size)
        mn /= self.downsample
        
        mn = int(mn)
        image = image.resize((mn, mn))      # just make it square for now, supporting non-square images will take a few changes elsewhere
        
        image = self.img_transform(image)
        # plt.imshow(image.permute((1, 2, 0)))


        # TODO(as) should probably also be returning GT masks
        # sample = {'image': image}
        # if self.augmentation is not None:
        #     for idx in range(self.num_aug):
        #         sample['aug' + str(idx + 1)] = self.augmentation(image)

        return [image]

    def __len__(self):
        len(self.files)


