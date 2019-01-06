import argparse

import numpy as np
import PIL
import torch

import constants
import utils


def get_cli_inputs():
    parser = argparse.ArgumentParser(
        'Predicts a flower name based on a pre-trained Pytorch model',
        add_help=True)
    parser.add_argument('input', action="store")
    parser.add_argument('checkpoint', action="store")
    parser.add_argument("--top_k",  type=int, action="store",
                        default=1,
                        help="Number of top k most likely classes to be returned")
    parser.add_argument("--category_names",  action="store",
                        default="cat_to_name.json",
                        help="Mapping of categories to real names")
    parser.add_argument("--gpu",  action='store_const', const='gpu',
                        help="Use GPU for inference flag")
    return parser.parse_args()


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = PIL.Image.open(image_path)

    # Resize
    if image.size[0] > image.size[1]:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))

    # Crop 
    left = (image.width - 224) / 2
    upper = (image.height - 224) / 2
    right = left + 224
    lower = upper + 224
    image = image.crop(box=(left, upper, right, lower))
    
    # Normalize
    image = np.array(image) / 255
    mean = np.array(constants.CHANNELS_MEANS)
    std = np.array(constants.CHANNELS_STDS)
    image = (image - mean) / std
    
    # Move color channels to first dimension as expected by PyTorch
    image = image.transpose((2, 0, 1))
    return image


def predict_image_class(image_path, model, run_on_gpu=True, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if run_on_gpu and device == "cpu":
        raise ValueError("User requested execution on GPU, which is not available or enabled")
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path) 
    image = torch.from_numpy(image).type(torch.FloatTensor) 
    image.unsqueeze_(0)
    
    probabilities = torch.exp(model.forward(image)) 
    top_probabilities, top_labels = probabilities.topk(topk)
    idx_to_class = {val: key for key, val in 
                    model.class_to_idx.items()}
    return (top_probabilities.tolist()[0],
            [idx_to_class[label] for label in top_labels.tolist()[0]])
    
    
def main():
    cli_inputs = get_cli_inputs()
    model = utils.load_model_checkpoint(cli_inputs.checkpoint)
    top_probabilities, top_labels = predict_image_class(cli_inputs.input,
                                                        model,
                                                        cli_inputs.gpu,
                                                        cli_inputs.top_k)
    cat_to_name = utils.get_category_to_name_mapping(
        cli_inputs.category_names)
    flower_names = [cat_to_name[cat] for cat in top_labels]
    print("Prediction for image {}:\n".format(cli_inputs.input))
    for prob, name in zip(top_probabilities, flower_names):
        print("Flower name: {}. Probability: {:.2%}".format(name, prob))


if __name__ == "__main__":
    # execute only if run as a script
    main()