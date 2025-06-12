import os
import shutil
import sys
from preprocessing.preprocess import main as preprocess
from preprocessing.rearrange import main as rearrange
from training.test import main as tests
from pathlib import Path
import argparse
from detector_map import detector_map

if __name__ == '__main__':
    
    # you also can choose not to use arguments and set the model and data path directly in the code
    
    parser = argparse.ArgumentParser(description="Process a test dataset using a selected model.")
    parser.add_argument('-m', '--model', type=str, default='xception', help='Name of the model to use')
    parser.add_argument('-d', '--dataset-path', type=str, default='../datasets/TestSet', help='Path to the dataset directory')
    parser.add_argument('-p', '--preprocess', action='store_true', help='Enable preprocessing (set flag to activate)')

    args = parser.parse_args()

    model = args.model
    data_path = Path(args.dataset_path).resolve()
    PREPROCESS = args.preprocess
    
    print("target path:", data_path)

    # Example: check if model is valid
    if model not in detector_map:
        print("Invalid model. Available models:")
        for model_name in detector_map.keys():
            print(f"- {model_name}")
        sys.exit(1)

    detector_config = Path(detector_map[model]['config']).resolve()
    detector_weights = Path(detector_map[model]['weights']).resolve()

    if PREPROCESS:
        print("Stage 1: Generating Frames and Landmarks!")
        sys.stdout.flush()
        
        preprocess(data_path)
        print("Stage 1: Frames and Landmarks Generated!")
        sys.stdout.flush()
        
        rearrange(data_path)
        print("Stage 2: JSON File Generated!")
        sys.stdout.flush()
        
    else:
        print("Skipping Preprocessing Stage! As Preprocess is set to False.")
        sys.stdout.flush()
        
    
    print("Stage 3: Testing Started!")
    tests(detector_path=detector_config, test_datasets=["TestSet"], weights_path=detector_weights)
    print(f"Stage 4: Results Generated in {Path(__file__).parent / 'results' / 'model'}")
    sys.stdout.flush()
