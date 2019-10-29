import numpy as np
import os
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

import experiment
from superpoint.settings import EXPER_PATH
from superpoint.datasets import get_dataset
from superpoint.models.classical_detectors_descriptors import classical_detector_descriptor

def sift(image):
    sift_points, sift_desc = classical_detector_descriptor(image, **{'method': 'sift'})
    return sift_points

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--export_name', type=str, default=None)
    args = parser.parse_args()

    export_name = args.export_name 
    output_dir = Path(EXPER_PATH, 'outputs/{}/'.format(export_name))
    if not output_dir.exists():
        os.makedirs(output_dir)


    config = {
        'data':
        {'name': 'finger',
        'cache_in_memory': 'false',
        'validation_size': 100}
    }
    dataset = get_dataset(config['data']['name'])(**config['data'])
    data = dataset.get_test_set()

    i = 0
    pbar = tqdm(None)
    while True:
        d = next(data)
        im = d['image'][..., 0].astype(np.uint8)
        pred = sift(im)
        p = {'points': np.array(np.where(pred>0)).T}
        
        # Export

        if not ('name' in d):
            p.update(d)  # Can't get the data back from the filename --> dump
        filename = d['name'].decode('utf-8') if 'name' in d else str(i)
        filepath = Path(output_dir, '{}.npz'.format(filename))
        np.savez_compressed(filepath, **p)
        i += 1
        pbar.update(1)     
                    
