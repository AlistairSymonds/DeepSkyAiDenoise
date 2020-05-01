import argparse
from pathlib import Path
from astropy.io import fits
import numpy as np
import astropy
from astropy.visualization import simple_norm
import matplotlib.pyplot as plt



def extract_patches_from_img(img: np.ndarray, patch_dims: (int,int)):
    xx = 0
    yy = 0


    patch_coords_to_extract = []

    while yy < img.shape[0] - patch_dims[0]:
        integration_row = img[yy:yy + patch_dims[0], 0:img.shape[1]]

        # norm = simple_norm(integration_row, 'sqrt')
        # plt.imshow(norm(integration_row).astype(np.uint8))
        # plt.show()

        print("Loading row from " + str(yy))
        patch_locs = []
        patches = []
        xx = 0
        while xx < (img.shape[1] - patch_dims[1]):
            patch = integration_row[:, xx:xx + patch_dims[0]]

            min_val_in_patch = np.min(np.min(patch))
            if min_val_in_patch > np.finfo(float).eps:
                print("extracting patch from [x,y] = [" + str(xx) + "," + str(yy) + "]")
                patch_locs.append((yy,xx))
                patches.append(patch)
            else:
                print("skipping patch at [x,y] = [" + str(xx) + "," + str(yy) + "]")
            xx += patch_dims[1]
        yy += patch_dims[0]
    return patch_locs, patches


def save_patches(patch_locs, patches, out_dir: Path, orig_file_name: str):
    if len(patch_locs) != patches:
        AssertionError("trying to save patches with varying number of locations")

    out_dir.mkdir(exist_ok=True, parents=True)

    for p in zip(patch_locs,patches):
        #<patch-x>_<patch_y>.fits
        patch_path = out_dir / (str(p[0][1]) +"_" + str(p[0][0])+"-"+ orig_file_name+".fits")
        print("Saving a patch to: " + str(patch_path))

        patch_fits = astropy.io.fits.PrimaryHDU(data=p[1])
        patch_fits.writeto(patch_path, overwrite=True)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="linear_deep_sky_integration_dataset")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--patch_x", default=224, type=int)
    ap.add_argument("--patch_y", default=224, type=int)

    args = ap.parse_args()

    dataset_path = Path(args.in_dir)
    out_path = Path(args.out_dir)
    patch_dims = [args.patch_y, args.patch_x]

    for target in dataset_path.glob('*'):
        print(target)
        for channel_integration in target.glob('*.fit*'):
            print("Extracting patches from integrated channel " + str(channel_integration))
            integration = fits.open((channel_integration))[0].data
            int_patch_locs, int_patches = extract_patches_from_img(integration, patch_dims)
            print("Got "+str(len(int_patches)) +" patches from integration, extracting valid matching patches from subs")

            integration_patches_path = out_path / target.stem / channel_integration.with_suffix("").stem / "int"
            save_patches(int_patch_locs,
                         int_patches,
                         integration_patches_path,
                         channel_integration.with_suffix("").stem)


            subs_folder = channel_integration.with_suffix("")
            for sub in subs_folder.glob('*.fit*'):
                sub_img = fits.open((sub))[0].data

                sub_patch_locs, sub_patches = extract_patches_from_img(sub_img, patch_dims)
                subs_patches_path = out_path / target.stem / channel_integration.with_suffix(
                    "").stem / "int"

                integration_patches_path = out_path / target.stem / channel_integration.with_suffix(
                    "").stem / "sub"
                save_patches(sub_patch_locs,
                             sub_patches,
                             integration_patches_path,
                             sub.with_suffix("").stem)


if __name__ == '__main__':
    main()