# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
1. Develop a binary classification model using a pretrained VGG19 to distinguish between defected and non-defected capacitors by modifying the last layer to a single neuron.  
2. Train the model on a dataset containing images of various defected and non-defected capacitors to improve defect detection accuracy.  
3. Optimize and evaluate the model to ensure reliable classification for capacitor quality assessment in manufacturing.

## DESIGN STEPS
### STEP 1:
Collect and preprocess the dataset containing images of defected and non-defected capacitors.

### STEP 2:
Split the dataset into training, validation, and test sets.

### STEP 3:
Load the pretrained VGG19 model with weights from ImageNet.

### STEP 4:
Remove the original fully connected (FC) layers and replace the last layer with a single neuron (1 output) with a Sigmoid activation function for binary classification.

### STEP 5:
Train the model using binary cross-entropy loss function and Adam optimizer.

### STEP 6:
Evaluate the model with test data loader and intepret the evaluation metrics such as confusion matrix and classification report.

## PROGRAM
Include your code here
```python
# Load Pretrained Model and Modify for Transfer Learning

# Load pre-trained VGG-19
model = models.vgg19(pretrained=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Print original model summary
summary(model, input_size=(3, 224, 224))

# Modify the final fully connected layer to match the dataset classes

num_classes = len(train_dataset.classes)  # number of output classes

# Replace the last layer of the classifier
model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

# Move updated model to device
model = model.to(device)

# Print updated model summary
summary(model, input_size=(3, 224, 224))


# Include the Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


# Train the model
def train_model(model, train_loader, test_loader, num_epochs=5):
    train_losses = []
    val_losses   = []

    for epoch in range(num_epochs):
        # ----- Training -----
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(test_loader))

        print(f"Epoch [{epoch+1}/{num_epochs}]  "
              f"Train Loss: {train_losses[-1]:.4f}  "
              f"Val Loss: {val_losses[-1]:.4f}")

    # Plot
    print("Name: DAKSHA C")
    print("Register Number: 212224230048")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss',      marker='o')
    plt.plot(range(1, num_epochs+1), val_losses,   label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return train_losses, val_losses

# Run training
train_losses, val_losses = train_model(model, train_loader, test_loader, num_epochs=5)


```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="930" height="708" alt="Screenshot 2026-03-23 203028" src="https://github.com/user-attachments/assets/bd75d561-40ce-4b70-bca0-1fa2f651cdec" />


### Confusion Matrix
<img width="745" height="667" alt="image" src="https://github.com/user-attachments/assets/ba04677b-15c6-4a63-b080-c5cc1495b85c" />


### Classification Report
<img width="605" height="268" alt="image" src="https://github.com/user-attachments/assets/595025e7-f00f-4158-b486-af0ae0344674" />

### New Sample Prediction
<img width="318" height="671" alt="image" src="https://github.com/user-attachments/assets/40ae4a89-cbc0-4a08-b02a-986c270a2e0c" />


## RESULT
The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors.

