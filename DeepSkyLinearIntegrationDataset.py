from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from astropy.io import fits
import numpy as np
import torch.tensor

class DeepSkyLinearIntegrationDataset(Dataset):

    def __init__(self, root_dir: Path, transform=None, lazy_load=True):
        self.root_dir = root_dir
        self.transform = transform

        self.patch_pair_paths = []
        self.sub_pair_paths = []

        for target in root_dir.glob("*"):
            print(target)
            for channel in target.glob("*"):
                print(channel)
                for patch_coords in channel.glob("*"):
                    print(patch_coords)
                    if (patch_coords / 'int').exists() and (patch_coords / 'sub').exists():
                        self.patch_pair_paths.append(patch_coords)


        for p in self.patch_pair_paths:
            int_fits_paths = list((p / 'int').glob("*.fit?"))[0]
            for i in (p / 'sub').glob("*.fit?"):
                self.sub_pair_paths.append((i,int_fits_paths))
                print(self.sub_pair_paths[-1])
        print("Total patch pair sets: " + str(len(self.patch_pair_paths)))
        print("Total subs across all pairs: " + str(len(self.sub_pair_paths)))


    def __len__(self):
        return len(self.sub_pair_paths)

    def __load_fits(self, fits_path: Path):
        print("Loading patch: " + str(fits_path))
        data = fits.open(str(fits_path))[0].data
        data = np.expand_dims(data, axis=2)
        return data.astype('<f4')

    def __getitem__(self, idx):

        path_pair = self.sub_pair_paths[idx]

        x = self.__load_fits(path_pair[0])
        y = self.__load_fits(path_pair[1])



        if self.transform:
            x = self.transform(x)
            y = self.transform(y)



        return (x,y)