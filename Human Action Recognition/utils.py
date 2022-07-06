import torchvision.transforms as tt

from dataset import HAR
from torch.utils.data import DataLoader

def get_loaders(train_csv_file, train_root_dir, test_csv_file, test_root_dir):

    # Transformations
    train_transform = tt.Compose([tt.Resize(128),
                            tt.RandomCrop(size=128, padding=4, padding_mode="reflect"),
                            tt.ToTensor()])
    test_transform = tt.Compose([tt.Resize(128),
                            tt.ToTensor()])

    # Load Data
    train_dataset = HAR(csv_file=train_csv_file,
                        root_dir=train_root_dir,
                        transform=train_transform)
    test_dataset = HAR(csv_file=test_csv_file,
                        root_dir=test_root_dir,
                        transform=test_transform)

    # DataLoader (input pipeline)
    batch_size = 100
    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size, num_workers=4, pin_memory=True) 

    return train_dl, test_dl