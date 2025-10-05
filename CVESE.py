import os
import cv2
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Feature Extraction
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = cv2.GaussianBlur(img, (5,5), 0)
    
    # Color mean
    mean_color = cv2.mean(img)[:3]
    
    # Grayscale histogram
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
    hist = hist / np.sum(hist)
    
    # ORB features
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(gray, None)
    orb_feat = np.mean(des, axis=0) if des is not None else np.zeros(32)
    
    return np.hstack([mean_color, hist, orb_feat])

# Load Dataset
data_dir = r""
classes = os.listdir(data_dir)

X_feat, y_labels, X_img = [], [], []

for idx, label in enumerate(classes):
    folder = os.path.join(data_dir, label)
    for file in os.listdir(folder):
        if not file.lower().endswith(('.png','.jpg','.jpeg')):
            continue
        path = os.path.join(folder, file)
        X_feat.append(extract_features(path))
        y_labels.append(idx)
        
        img = cv2.imread(path)
        img = cv2.resize(img, (128,128))
        img = cv2.GaussianBlur(img, (5,5), 0)
        X_img.append(img)

X_feat = np.array(X_feat)
X_img = np.array(X_img)
y_labels = np.array(y_labels)

# Save Features to Excel
df = pd.DataFrame(X_feat)
df['label'] = y_labels
df.to_excel("rose_features.xlsx", index=False)
print("✅ Features saved to rose_features.xlsx")

# Train Test Split
num_classes = len(classes)
test_size = max(int(0.2 * len(X_feat)), num_classes)

Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_feat, y_labels, test_size=test_size, random_state=42, stratify=y_labels
)
Xi_train, Xi_test, yi_train, yi_test = train_test_split(
    X_img, y_labels, test_size=test_size, random_state=42, stratify=y_labels
)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(Xf_train, yf_train)
joblib.dump(knn, "rose_knn.pkl")

y_pred_knn = knn.predict(Xf_test)
print("\n📊 k-NN Accuracy:", accuracy_score(yf_test, y_pred_knn))
print(classification_report(yf_test, y_pred_knn, target_names=classes))
print("Confusion Matrix:\n", confusion_matrix(yf_test, y_pred_knn))

# Train CNN
Xi_train = Xi_train / 255.0
Xi_test = Xi_test / 255.0
yi_train_oh = to_categorical(yi_train, len(classes))
yi_test_oh = to_categorical(yi_test, len(classes))

cnn = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.4),
    Dense(len(classes),activation='softmax')
])

cnn.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(Xi_train, yi_train_oh, epochs=15, batch_size=8, validation_split=0.1, verbose=1)
loss, acc = cnn.evaluate(Xi_test, yi_test_oh, verbose=0)
cnn.save("rose_cnn.keras")
print("\n🧠 CNN Accuracy:", acc)

# Predict Single Validation Image
image_path = r""

if not os.path.exists(image_path):
    print("⚠️ Image not found! Check the path.")
else:
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128,128))
    img_blur = cv2.GaussianBlur(img, (5,5), 0)
    
    # True label from filename
    true_label = os.path.splitext(os.path.basename(image_path))[0]
    
    # k-NN prediction
    feat = extract_features(image_path).reshape(1,-1)
    pred_knn = knn.predict(feat)[0]
    
    # CNN prediction
    pred_cnn = np.argmax(cnn.predict((img_blur/255.0).reshape(1,128,128,3), verbose=0))
    
    # Edge detection & contours
    gray_img = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 50, 150)
    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contour = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_contour, cnts, -1, (0,255,0), 1)
    if cnts:
        x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        cv2.rectangle(img_contour, (x,y), (x+w,y+h), (0,0,255), 2)
    
    # Plot results
    plt.figure(figsize=(8,4))
    
    # Image with contours
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
    plt.title(f"T:{true_label} | k-NN:{classes[pred_knn]} | CNN:{classes[pred_cnn]}")
    plt.axis("off")
    
    # Histogram
    plt.subplot(1,2,2)
    plt.plot(cv2.calcHist([gray_img],[0],None,[64],[0,256]))
    plt.title("Grayscale Histogram")
    
    plt.tight_layout()
    plt.show()

