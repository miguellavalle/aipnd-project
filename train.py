import argparse
import copy
import time

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import models

import classifier
import constants
import utils
import workspace_utils


def get_cli_inputs():
    parser = argparse.ArgumentParser(
        'Trains a Pytorch network on a dataset and saves the model as a checkpoint',
        add_help=True)
    parser.add_argument('data_directory', action="store")
    parser.add_argument("--save_dir",  action="store",
                        default=".", help="Directory where to save checkpoint")
    parser.add_argument("--arch",  action="store",
                        default="vgg19", help="Architecture to use for CNN")
    parser.add_argument("--learning_rate", type=float, action="store",
                        help="Learning rate to be used for gradient descent") 
    parser.add_argument("--hidden_units",  type=int, action="append",
                        help="Number of hidden layers in the classifier")
    parser.add_argument("--epochs",  type=int, action="store",
                        default=0,
                        help="Number of epochs during training stage")
    parser.add_argument("--gpu",  action='store_const', const='gpu',
                        help="Use the GPU for training flag")
    return parser.parse_args()


def build_cnn_model(architecture, n_hidden_layers):
    # Instatiate the requested model
    model = getattr(models, architecture)(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Create the classifier and add it to the model
    # For non vgg architecures, the following line would be:
    # input_size = model.fc.in_features # Resner, Inception
    # input_size = model.classifier.in_features # Densenet
    # input_size = model.classifier[1].in_channels #SqueezeNet
    input_size = model.classifier[0].in_features
    classif = classifier.Classifier(input_size, 102, n_hidden_layers)
    model.classifier = classif
    return model, input_size


def deep_learning(model, dataloaders, dataset_sizes, criterion, optimizer,
                  scheduler, device='cuda', num_epochs=25):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    best_epoch = 0
    training_losses , training_accuracy = [],[]
    validation_losses , validation_accuracy = [],[]
    
    model.to(device)
    
    start_time = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        phases = [constants.TRAIN, constants.VALID]
        for phase in phases:
            if phase == constants.TRAIN:
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == constants.TRAIN):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == constants.TRAIN:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_accuracy = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Accuracy: {:.2%}'.format(
                phase, epoch_loss, epoch_accuracy))

            # deep copy the model
            if phase == constants.VALID:
                validation_losses.append(epoch_loss)
                validation_accuracy.append(epoch_accuracy * 100.0)
                if epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    best_model_weights = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
            else:
                training_losses.append(epoch_loss)
                training_accuracy.append(epoch_accuracy * 100.0)

        print()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation accuracy: {:.2%}'.format(best_accuracy))
    print('Best epoch {}'.format(best_epoch))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model, training_losses, training_accuracy, validation_losses, validation_accuracy


def do_deep_learning(model, data_loaders, dataset_sizes, run_on_gpu, num_epochs,
                     learning_rate):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if run_on_gpu and device == "cpu":
        raise ValueError("User requested execution on GPU, which is not available or enabled")
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    with workspace_utils.active_session():
        model, _, _, _, _ = deep_learning(model, data_loaders, dataset_sizes, criterion,
                                          optimizer, scheduler, device=device,
                                          num_epochs=num_epochs)
    return model
    

def main():
    cli_inputs = get_cli_inputs()
    data_loaders, dataset_sizes, class_to_idx = utils.get_data_loaders(
        cli_inputs.data_directory)
    model, classifier_input_size = build_cnn_model(cli_inputs.arch,
                                                   cli_inputs.hidden_units)
    model = do_deep_learning(model, data_loaders, dataset_sizes,
                             cli_inputs.gpu, cli_inputs.epochs,
                             cli_inputs.learning_rate)
    utils.save_model_checkpoint(model, cli_inputs.save_dir,
                                cli_inputs.arch, class_to_idx,
                                classifier_input_size, cli_inputs.hidden_units)
    

if __name__ == "__main__":
    # execute only if run as a script
    main()

    
