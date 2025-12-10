from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2 as cv
import os
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset, Dataset
from PIL import Image
import numpy as np
from numpy import asarray
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# CLASS FOR DETECTING FACES
class FACELOADING:
  def __init__(self, directory):
      self.directory = directory
      self.target_size = (160,160)
      self.X = []
      self.Y = []
      self.skipped_images = []
      self.mtcnn = MTCNN(
          image_size=160, margin=0, min_face_size=5,
          thresholds=[0.4, 0.5, 0.5], factor=0.709, post_process=True,
          device=device
        )

  def extract_face(self, filename, required_size=(160, 160)):
      img = cv.imread(filename)
      if img is None:
          print(f"Warning: Could not read image {filename}. Skipping.")
          self.skipped_images.append(filename)
          return None
      img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
      boxes, _ = self.mtcnn.detect(img)
      if boxes is None or len(boxes) == 0:
          print(f"Warning: No face detected in {filename}. Skipping.")
          self.skipped_images.append(filename)
          return None

      if boxes is not None and len(boxes) > 1:
          areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
          largest_box_idx = np.argmax(areas)
          x, y, x2, y2 = boxes[largest_box_idx]
      else:
          x, y, x2, y2 = boxes[0]

      x, y, x2, y2 = int(x), int(y), int(x2), int(y2)

      x, y = max(0, x), max(0, y)
      x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])

      face = img[y:y2, x:x2]
      if face.shape[0] == 0 or face.shape[1] == 0:
          print(f"Warning: Detected face region is empty in {filename}. Skipping.")
          self.skipped_images.append(filename)
          return None

      face_arr = cv.resize(face, self.target_size)
      return face_arr

  def load_faces(self, directory):
    faces = list()
    for filename in os.listdir(directory):
      try:
        path = os.path.join(directory, filename)
        face = self.extract_face(path)
        if face is not None:
          faces.append(face)
      except Exception as e:
        print(f"Error processing {filename}: {e}")
        self.skipped_images.append(filename)

    return faces

  def load_classes(self):
    for subdir in os.listdir(self.directory):
      path = os.path.join(self.directory, subdir)
      if not os.path.isdir(path):
        continue
      faces = self.load_faces(path)
      labels = [subdir for _ in range(len(faces))]
      print(f'Loaded successfully: {len(labels)} faces for class {subdir}')
      self.X.extend(faces)
      self.Y.extend(labels)

    return np.asarray(self.X), np.asarray(self.Y)

# LOADING DATASET
directory = '/content/Face-Recognition-GESAD-5/train'

faceloading_train = FACELOADING(directory)
X_train_raw, Y_train_raw = faceloading_train.load_classes()

print(f"Loaded {len(X_train_raw)} faces and {len(Y_train_raw)} labels from the training directory.")

directory_valid = '/content/Face-Recognition-GESAD-5/valid'

faceloading_valid = FACELOADING(directory_valid)
X_valid_raw, Y_valid_raw = faceloading_valid.load_classes()

print(f"Loaded {len(X_valid_raw)} faces and {len(Y_valid_raw)} labels from the validation directory.")

directory_test = '/content/Face-Recognition-GESAD-5/test'

faceloading_test = FACELOADING(directory_test)
X_test_raw, Y_test_raw = faceloading_test.load_classes()

print(f"Loaded {len(X_test_raw)} faces and {len(Y_test_raw)} labels from the test directory.")

encoder = LabelEncoder()

# INITIALIZING MODEL NOT FACE TUNED AND GETTING EMBEDDINGS
embedder = InceptionResnetV1(pretrained='vggface2').to(device)

def get_embedding(face_img):
  face_img = face_img.astype('float32')
  face_img = np.expand_dims(face_img, axis = 0)

  yhat = embedder.forward(torch.tensor(face_img).permute(0, 3, 1, 2).to(device))
  return yhat[0].detach().cpu().numpy()

embedded_x = []

embedder.eval()

for img in X_train_raw:
  embedded_x.append(get_embedding(img))

embedded_x = np.asarray(embedded_x)

Y_train_encoded = encoder.fit_transform(Y_train_raw)

in_encoder = Normalizer(norm='l2')
embedded_x = in_encoder.transform(embedded_x)

embedded_x_test = []

embedder.eval()

for img in X_test_raw:
  embedded_x_test.append(get_embedding(img))

embedded_x_test = np.asarray(embedded_x_test)

embedded_x_test = in_encoder.transform(embedded_x_test)

Y_test_encoded = encoder.transform(Y_test_raw)

print(f"Extracted and normalized {len(embedded_x_test)} embeddings for the test set.")
print(f"Encoded {len(Y_test_encoded)} labels for the test set.")

embedded_x_valid = []

embedder.eval()

for img in X_valid_raw:
  embedded_x_valid.append(get_embedding(img))

embedded_x_valid = np.asarray(embedded_x_valid)

embedded_x_valid = in_encoder.transform(embedded_x_valid)

Y_valid_encoded = encoder.transform(Y_valid_raw)

print(f"Extracted and normalized {len(embedded_x_valid)} embeddings for the validation set.")
print(f"Encoded {len(Y_valid_encoded)} labels for the validation set.")

# CREATING BATCHES FOR TRAINING NOT FACE TUNED

X_train_tensor = torch.tensor(embedded_x, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_encoded, dtype=torch.long)
X_valid_tensor = torch.tensor(embedded_x_valid, dtype=torch.float32)
Y_valid_tensor = torch.tensor(Y_valid_encoded, dtype=torch.long)
X_test_tensor = torch.tensor(embedded_x_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test_encoded, dtype=torch.long)

batch_size = 32

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, Y_valid_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training DataLoader created with {len(train_loader)} batches of size {batch_size}.")
print(f"Validation DataLoader created with {len(valid_loader)} batches of size {batch_size}.")
print(f"Test DataLoader created with {len(test_loader)} batches of size {batch_size}.")

class FaceClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FaceClassifier, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.7)

        self.fc2 = nn.Linear(256, 128) # Segunda camada oculta
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.7)

        self.fc3 = nn.Linear(128, num_classes) # Camada de saída

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x

num_classes = len(encoder.classes_)

model = FaceClassifier(num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"FaceClassifier model instantiated with {num_classes} output classes.")
print(f"Model is running on: {device}")
print("Loss function: CrossEntropyLoss")
print("Optimizer: Adam with learning rate 0.001")

# TRAINING NOT FACE TUNED

num_epochs = 10

train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    correct_train_predictions = 0
    total_train_samples = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train_samples += labels.size(0)
        correct_train_predictions += (predicted == labels).sum().item()

    epoch_train_loss = running_train_loss / total_train_samples
    epoch_train_accuracy = correct_train_predictions / total_train_samples
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)

    model.eval()
    running_valid_loss = 0.0
    correct_valid_predictions = 0
    total_valid_samples = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_valid_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_valid_samples += labels.size(0)
            correct_valid_predictions += (predicted == labels).sum().item()

    epoch_valid_loss = running_valid_loss / total_valid_samples
    epoch_valid_accuracy = correct_valid_predictions / total_valid_samples
    valid_losses.append(epoch_valid_loss)
    valid_accuracies.append(epoch_valid_accuracy)

    print(f'Epoch {epoch+1}/{num_epochs} - ' \
          f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f} | ' \
          f'Valid Loss: {epoch_valid_loss:.4f}, Valid Acc: {epoch_valid_accuracy:.4f}')

print('\nTraining complete!')

# EVALUATION NOT FACE TUNED

def visualize_random_prediction(X_raw, Y_encoded, encoder, model, device):
    idx = random.randint(0, len(X_raw) - 1)

    raw_image = X_raw[idx]
    true_label_encoded = Y_encoded[idx]
    true_label_name = encoder.inverse_transform([true_label_encoded])[0]

    sample_embedding = get_embedding(raw_image)

    sample_embedding_tensor = torch.tensor(sample_embedding, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(sample_embedding_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_label_index = torch.argmax(probabilities, dim=1).item()

    predicted_label_name = encoder.inverse_transform([predicted_label_index])[0]

    plt.imshow(raw_image)
    plt.title(f"True: {true_label_name}, Predicted: {predicted_label_name}")
    plt.axis('off')
    plt.show()

visualize_random_prediction(X_test_raw, Y_test_encoded, encoder, model, device)


model.eval()
running_test_loss = 0.0
correct_test_predictions = 0
total_test_samples = 0
all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_test_samples += labels.size(0)
        correct_test_predictions += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

test_loss = running_test_loss / total_test_samples
test_accuracy = correct_test_predictions / total_test_samples

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

f1 = f1_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
precision = precision_score(all_labels, all_predictions, average='weighted')

print(f"\nF1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=encoder.classes_))

cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


class FineTuneFaceRecognizer(nn.Module):
    def __init__(self, num_classes, freeze_backbone=False):
        super(FineTuneFaceRecognizer, self).__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2')

        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
            print("INFO: ResNet backbone freezed.")
        else:
          print("ResNet Backbone unfreezed")

        self.classifier_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier_head(x)
        return x

num_classes = len(encoder.classes_)
fine_tune_model = FineTuneFaceRecognizer(num_classes, freeze_backbone = True).to(device)

for name, param in fine_tune_model.resnet.named_parameters():
    if "repeat_3" in name:
        param.requires_grad = True

params_to_update = []
for name, param in fine_tune_model.named_parameters():
    if param.requires_grad:
        params_to_update.append(param)
print(f"Unfreezing. Total of tensors to train: {len(params_to_update)}")

criterion_ft = nn.CrossEntropyLoss()
optimizer_ft = torch.optim.AdamW(filter(lambda p: p.requires_grad, fine_tune_model.parameters()),
                                 lr=0.0001, weight_decay = 0.0001)

print(f"FineTuneFaceRecognizer model instantiated with {num_classes} output classes.")
print(f"Model is running on: {device}")
print("Loss function: CrossEntropyLoss")
print("Optimizer: AdamW with learning rate 0.001")


class FaceDataset(Dataset):
    def __init__(self, X_raw, Y_raw):
        self.X_raw = X_raw
        self.Y_raw = Y_raw
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.X_raw)

    def __getitem__(self, idx):
        image = self.X_raw[idx]
        label = self.Y_raw[idx]

        image = self.transform(image)
        label_encoded = encoder.transform([label])[0]

        return image, torch.tensor(label_encoded, dtype=torch.long)

batch_size = 32
workers = 0 if os.name == 'nt' else 2

train_dataset_ft = FaceDataset(X_train_raw, Y_train_raw)
valid_dataset_ft = FaceDataset(X_valid_raw, Y_valid_raw)
test_dataset_ft = FaceDataset(X_test_raw, Y_test_raw)

train_loader_ft = DataLoader(train_dataset_ft, batch_size=batch_size, shuffle=True, num_workers=workers)
valid_loader_ft = DataLoader(valid_dataset_ft, batch_size=batch_size, shuffle=False, num_workers=workers)
test_loader_ft = DataLoader(test_dataset_ft, batch_size=batch_size, shuffle=False, num_workers=workers)

print(f"Training DataLoader for fine-tuning created with {len(train_loader_ft)} batches of size {batch_size}.")
print(f"Validation DataLoader for fine-tuning created with {len(valid_loader_ft)} batches of size {batch_size}.")
print(f"Test DataLoader for fine-tuning created with {len(test_loader_ft)} batches of size {batch_size}.")

num_epochs_ft = 5

train_losses_ft = []
train_accuracies_ft = []
valid_losses_ft = []
valid_accuracies_ft = []

scheduler_ft = ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=2)

max_grad_norm = 1.0

print("Starting Fine-tuning Training...")

for epoch in range(num_epochs_ft):
    fine_tune_model.train()
    running_train_loss_ft = 0.0
    correct_train_predictions_ft = 0
    total_train_samples_ft = 0

    for inputs, labels in train_loader_ft:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer_ft.zero_grad()

        outputs = fine_tune_model(inputs)
        loss = criterion_ft(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(fine_tune_model.parameters(), max_norm=max_grad_norm)
        optimizer_ft.step()

        running_train_loss_ft += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train_samples_ft += labels.size(0)
        correct_train_predictions_ft += (predicted == labels).sum().item()

    epoch_train_loss_ft = running_train_loss_ft / total_train_samples_ft
    epoch_train_accuracy_ft = correct_train_predictions_ft / total_train_samples_ft
    train_losses_ft.append(epoch_train_loss_ft)
    train_accuracies_ft.append(epoch_train_accuracy_ft)

    fine_tune_model.eval()
    running_valid_loss_ft = 0.0
    correct_valid_predictions_ft = 0
    total_valid_samples_ft = 0

    with torch.no_grad():
        for inputs, labels in valid_loader_ft:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = fine_tune_model(inputs)
            loss = criterion_ft(outputs, labels)

            running_valid_loss_ft += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_valid_samples_ft += labels.size(0)
            correct_valid_predictions_ft += (predicted == labels).sum().item()

    epoch_valid_loss_ft = running_valid_loss_ft / total_valid_samples_ft
    epoch_valid_accuracy_ft = correct_valid_predictions_ft / total_valid_samples_ft
    valid_losses_ft.append(epoch_valid_loss_ft)
    valid_accuracies_ft.append(epoch_valid_accuracy_ft)

    scheduler_ft.step(epoch_valid_loss_ft)

    print(f'Epoch {epoch+1}/{num_epochs_ft} - ' \
          f'Train Loss: {epoch_train_loss_ft:.4f}, Train Acc: {epoch_train_accuracy_ft:.4f} | ' \
          f'Valid Loss: {epoch_valid_loss_ft:.4f}, Valid Acc: {epoch_valid_accuracy_ft:.4f}')

print('\nFine-tuning Training complete!')

THRESHOLD = 0.60
class_names = list(encoder.classes_)
class_names.append("Desconhecido")
unknown_index = len(encoder.classes_)

fine_tune_model.eval()
running_test_loss_ft = 0.0
correct_test_predictions_ft = 0
total_test_samples_ft = 0
all_labels_ft = []
all_predictions_ft = []

with torch.no_grad():
    for inputs, labels in test_loader_ft:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = fine_tune_model(inputs)
        loss = criterion_ft(outputs, labels)

        running_test_loss_ft += loss.item() * inputs.size(0)

        probs = F.softmax(outputs, dim=1)
        max_probs, predicted = torch.max(probs, 1)
        predicted[max_probs < THRESHOLD] = unknown_index

        total_test_samples_ft += labels.size(0)

        correct_test_predictions_ft += (predicted == labels).sum().item()

        all_labels_ft.extend(labels.cpu().numpy())
        all_predictions_ft.extend(predicted.cpu().numpy())

test_loss_ft = running_test_loss_ft / total_test_samples_ft
test_accuracy_ft = correct_test_predictions_ft / total_test_samples_ft

print(f'Test Loss: {test_loss_ft:.4f}, Test Accuracy (Considerando Unknown como erro se label for conhecida): {test_accuracy_ft:.4f}')

all_labels_idx = list(range(len(class_names)))

print("\nClassification Report (Com Desconhecidos):")
print(classification_report(all_labels_ft, all_predictions_ft, labels=all_labels_idx, target_names=class_names))

cm_ft = confusion_matrix(all_labels_ft, all_predictions_ft, labels=all_labels_idx)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_ft, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix (Threshold: {THRESHOLD})')
plt.show()