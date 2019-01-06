import json

import torch
from torchvision import datasets, transforms, models

import constants
import classifier


def get_data_loaders(data_dir):
    data_transforms = {
        constants.TRAIN: transforms.Compose([transforms.RandomRotation(30),
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(constants.CHANNELS_MEANS,
                                                                  constants.CHANNELS_STDS)]),
        constants.VALID: transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(constants.CHANNELS_MEANS,
                                                                  constants.CHANNELS_STDS)]),
        constants.TEST: transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(constants.CHANNELS_MEANS,
                                                                 constants.CHANNELS_STDS)])   
    }

    folders = {constants.TRAIN: data_dir + '/' + constants.TRAIN,
               constants.VALID: data_dir + '/' + constants.VALID,
               constants.TEST: data_dir + '/' + constants.TEST
    }
    image_datasets = {x: datasets.ImageFolder(folders[x], transform=data_transforms[x])
                      for x in constants.DATASETS}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
                   for x in constants.DATASETS}

    dataset_sizes = {x: len(image_datasets[x]) for x in constants.DATASETS}
    return dataloaders, dataset_sizes, image_datasets[constants.TRAIN].class_to_idx


def get_category_to_name_mapping(category_names_mapping):
    with open(category_names_mapping, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def save_model_checkpoint(model, checkpoint_directory, architecture, class_to_idx,
                          classifier_input_size, n_hidden_units):
    model.class_to_idx = class_to_idx
    checkpoint = {'arch': architecture,
                  'classifier_input_size': classifier_input_size,
                  'n_hidden_units': n_hidden_units,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}
    model.cpu()
    torch.save(checkpoint, checkpoint_directory + constants.CHECKPOINT_PATH)


def load_model_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    model.class_to_idx = checkpoint['class_to_idx']
    
    classif = classifier.Classifier(checkpoint['classifier_input_size'], 102, 
                                    checkpoint['n_hidden_units'])
    model.classifier = classif
    model.eval()
    
    model.load_state_dict(checkpoint['state_dict'])
    return model