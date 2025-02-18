import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import models

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from PIL import Image
import pandas as pd

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class SurvivalCNNWithResNet(nn.Module):
    def __init__(self):
        super(SurvivalCNNWithResNet, self).__init__()
        
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = Identity()

        # Fully connected layers
        self.fc_image = nn.Linear(2048, 1000)
        self.fc_combined = nn.Linear(1000 + 2, 256)  # 2 extra inputs for survival time and event indicator
        self.fc_output = nn.Linear(256, 20)         # 20 discrete time intervals
        
        self.dropout = nn.Dropout(0.25)

    def forward(self, image, survival_time, event_indicator):
        # Image feature extraction
        image = self.resnet(image)
        image = torch.flatten(image, 1)  # Flatten the tensor

        # Fully connected layers
        image = F.relu(self.fc_image(image))

        # Combine image features with survival time and event indicator
        combined = torch.cat((image, survival_time.unsqueeze(1), event_indicator.unsqueeze(1)), dim=1)
        combined = F.relu(self.fc_combined(combined))
        combined = self.dropout(combined)

        # Output hazard function
        output = torch.sigmoid(self.fc_output(combined))  # Ensures hazards are between 0 and 1
        return output


def negative_log_likelihood(hazard_preds, survival_times, events):
    """
    hazard_preds: Predicted hazard function (batch_size, num_intervals)
    survival_times: Survival times (batch_size)
    events: Event indicators (batch_size)
    """
    # Convert survival times to discrete intervals
    survival_intervals = survival_times.long()

    # Create a mask for events
    # event_mask = torch.arange(hazard_preds.size(1), device=hazard_preds.device).unsqueeze(0) < survival_intervals.unsqueeze(1)

    # Probability of survival until the event
    survival_probs = torch.cumprod(1 - hazard_preds, dim=1)

    # Hazard for the specific survival time
    hazard_at_event = torch.gather(hazard_preds, 1, survival_intervals.unsqueeze(1)).squeeze(1)

    # Log-likelihood
    log_likelihood = events * torch.log(hazard_at_event + 1e-8) + torch.log(survival_probs[:, -1] + 1e-8)

    return -log_likelihood.mean()


def generate_annotations(image_dir, data_filepath):
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]  # Image paths from folder

    intervals = 19
    max_days_alive = 2200

    scaling_factor = max_days_alive/intervals

    with open(data_filepath, mode ='r') as file:
        csvFile = csv.reader(file)
        next(csvFile, None)
        data_from_csv = []
        sorted_data = []

        for lines in csvFile:
            filename = lines[1]
            death_occurred = True if lines[5] == "Dead" else False
            survival_time = round(int(lines[4])/scaling_factor) if lines[4] != "'--" else 19

            data_from_csv.append([filename ,survival_time, int(death_occurred)])

    for image_path in image_paths:
        for data in data_from_csv:
            if data[0] in image_path:
                sorted_data.append([image_path, data[1], data[2]])
                break

    with open('annotations.csv', mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(sorted_data)
    print("Finished generating annotations")


def get_data(annotations_path):

    with open(annotations_path, mode ='r') as file:
        csvFile = csv.reader(file)
        next(csvFile, None)
        image_filenames = []
        survival_times = []
        death_occurred = []

        for lines in csvFile:
            image_filenames.append(lines[0])
            survival_times.append(int(lines[1]))
            death_occurred.append(int(lines[2]))

    print("Loaded", len(image_filenames), "image files")
    print("Loaded", len(survival_times), "survival times")
    print("Loaded", len(death_occurred), "events")
            
    return image_filenames, survival_times, death_occurred

# Dataset class
class SurvivalDataset(Dataset):
    def __init__(self, image_paths, survival_times, events, transform=transforms.Compose([transforms.ToTensor()])):
        self.image_paths = image_paths  # Image paths from folder
        self.transform = transform #Transformations
        self.survival_times = survival_times  # Tensor of survival times
        self.events = events  # Tensor of event indicators (1 = event, 0 = censored)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image.convert("RGB"))
        
        return image, self.survival_times[idx], self.events[idx]

# Training loop
def train_model(model, train_loader, test_loader, num_epochs, learning_rate, device, save_model_dir, save_period = 0):
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir) 
    save_model_dir = os.path.join(save_model_dir, "epoch_")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for images, survival_times, events in train_loader:
            images = images.to(device)
            survival_times = survival_times.to(device)
            events = events.to(device)

            # Forward pass
            if model.__class__.__name__ == "ResNet":
                hazard_preds = model(images)
            else:
                hazard_preds = model(images, survival_times, events)
            # Compute loss
            loss = negative_log_likelihood(hazard_preds, survival_times, events)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_accuracy = test_model(model, train_loader, device)
        vaild_accuracy = test_model(model, test_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}, Training Accuracy: {round(train_accuracy, 3)}, Validation Accuracy: {round(vaild_accuracy, 3)}")
        save_results(train_accuracy, vaild_accuracy)

        if save_period != 0 and epoch % save_period == 0:
            torch.save(model.state_dict(), save_model_dir + str(epoch) + ".pt")

def test_model(model, test_loader, device):
    correct_pred = 0
    incorrect_pred = 0
    accuracy = 0.0
    total_preds = len(test_loader.dataset)

    for images, survival_times, events in test_loader:
        images = images.to(device)
        survival_times = survival_times.to(device)
        events = events.to(device)
        if model.__class__.__name__ == "ResNet":
            hazard_preds = model(images)
        else:
            hazard_preds = model(images, survival_times, events) 
        for index in range(len(hazard_preds)):
            hazard_pred = hazard_preds[index].tolist()
            survival_time = survival_times[index].tolist()
       
            if hazard_pred.index(max(hazard_pred)) == survival_time:
                correct_pred += 1
            else:
                incorrect_pred += 1

    accuracy = float(correct_pred) / total_preds
    return accuracy

def inference(model, test_loader, device):
    for images, survival_times, events in test_loader:
        images = images.to(device)
        survival_times = survival_times.to(device)
        events = events.to(device)

        hazard_preds = model(images, survival_times, events)

        for index in range(len(hazard_preds)):
            hazard_pred = hazard_preds[index].tolist()

            #transform = transforms.ToPILImage()
            #image = transform(images[index])
            #image.show()

            generate_plot(hazard_pred)
            #print(hazard_pred)
            


def save_results(train_accuracy, vaild_accuracy, file_path="metrics2.xlsx"):
    """
    Saves accuracy, precision, and the current date to an Excel file.
    If the file exists, it appends a new row; otherwise, it creates a new file.
    
    :param accuracy: float, accuracy of the model
    :param precision: float, precision of the model
    :param file_path: str, path to the Excel file (default is 'metrics.xlsx')
    """
    data = {
        "Training Accuracy": [train_accuracy],
        "Validation Precision": [vaild_accuracy]
    }
    df_new = pd.DataFrame(data)
    
    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)
        df_combined = pd.concat([existing_df, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_excel(file_path, index=False)



def main():
    # Now use the relative path
    Base_dir= os.path.dirname(os.path.abspath(__file__))
    os.chdir(Base_dir)
    image_dir = os.path.join(Base_dir, 'images')
    data_filepath = Base_dir+r'\clinical.csv'
    annotations_path = Base_dir+r'\annotations.csv'
    save_model_dir = os.path.join(Base_dir, 'epochs')
    test_image_path =Base_dir+r'\images\TCGA-2J-AAB1-01Z-00-DX1.png'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    data = get_data(annotations_path)

    #model = SurvivalCNNWithResNet()
    #print(model)

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 20)

    model = model.to(device)

    # Transformations
    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.RandomCrop(size=(64, 64)),
        #transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    modified_list = [os.path.join(image_dir, os.path.basename(path)) for path in data[0]]
    train_dataset = SurvivalDataset(modified_list[:168], data[1][:168], data[2][:168], transformations)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    validation_dataset = SurvivalDataset(modified_list[169:], data[1][169:], data[2][169:], transformations)
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=True)

    print(f"Images in train dataset: {len(train_loader.dataset)}")
    print(f"Images in validation dataset: {len(validation_loader.dataset)}")

    epochs = 501
    learning_rate = 0.001

    train_model(model, train_loader, validation_loader, epochs, learning_rate, device, save_model_dir, 5)

    #torch.save(model.state_dict(), save_model_dir)
    
    

    print("Done")



def generate_plot(data):
    # Sample data for the bars
    x = np.arange(20)  # 20 bars

    # Create the bar chart
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.bar(x, data, color='skyblue')  # Create the bars

    # Add labels and title
    plt.xlabel('Time Intervals over 3 years', fontsize=12)
    plt.ylabel('Probabilty of Death', fontsize=12)
    plt.title('Probabilty of Death over Time', fontsize=14)

    # Customize x-axis ticks
    plt.xticks(x, labels=[f'{i+1}' for i in x], rotation=45, ha='right')

    # Display the plot
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

def load_model(model_path):
    model = SurvivalCNNWithResNet()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


#generate_annotations(image_dir, data_filepath) # Get annotations if needed
main()
