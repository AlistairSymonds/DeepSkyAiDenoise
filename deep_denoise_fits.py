import torch
import argparse
import multiprocessing
from pathlib import Path
from astropy.io import fits
import numpy as np
from models.models import DnCNN

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True)
    ap.add_argument("--network_file", required=True)
    ap.add_argument("--tile_size", default=128, type=int)
    ap.add_argument("--output_fits")
    args = ap.parse_args()


    model_path = Path(args.network_file)
    print("Loading model file from: " + str(model_path))
    model = torch.load(str(model_path))

    model.eval()

    fits_path = Path(args.fits)

    input_fits = fits.open(str(fits_path))

    input_header = input_fits[0].header
    input_data = input_fits[0].data.astype('<f4')
    input_min = np.min(np.min(input_data))
    input_max = np.max(np.max(input_data))
    input_data = (input_data - input_min)/input_max

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on device: "+str(device))
    model.to(device)


    output = np.zeros(input_data.shape)
    for imin in range(0, output.shape[0], args.tile_size):
        imax = min(imin + args.tile_size, output.shape[0])
        for jmin in range(0, output.shape[1], args.tile_size):
            jmax = min(jmin + args.tile_size, output.shape[1])
            shape_out_sub = (imax - imin, jmax - jmin)
            #denoisd_tile = np.zeros(shape_out_sub)
            print("Denoising tile at [" + str(imin) + ", " + str(jmin) + "]")

            tensor_np = np.expand_dims(np.expand_dims(input_data[imin:imax, jmin:jmax], axis=0), axis=0)
            input_tensor = torch.tensor(tensor_np).to(device)
            denoised_tile = model(input_tensor)

            output[imin:imax, jmin:jmax] = denoised_tile[:].cpu().detach().numpy()
            # footprint[imin:imax, jmin:jmax] = footprint_sub


    data_fits_order = output.astype('>f4')
    patch_fits = fits.PrimaryHDU(data=data_fits_order, header=input_header)
    output_path = Path(args.output_fits)
    print("saving to: " + str(output_path))
    patch_fits.writeto(str(output_path), overwrite=True)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    torch.multiprocessing.freeze_support()
    main()