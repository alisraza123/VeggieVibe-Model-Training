import os
import random
import shutil

source_dir = "."
train_dir = "train"
val_dir = "val"
test_dir = "test"

# create folders
for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

for folder in os.listdir(source_dir):

    if os.path.isdir(folder) and folder not in [train_dir, val_dir, test_dir]:

        files = os.listdir(folder)
        random.shuffle(files)

        total = len(files)

        train_count = int(total * 0.7)
        val_count = int(total * 0.15)

        train_files = files[:train_count]
        val_files = files[train_count:train_count+val_count]
        test_files = files[train_count+val_count:]

        os.makedirs(os.path.join(train_dir, folder), exist_ok=True)
        os.makedirs(os.path.join(val_dir, folder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, folder), exist_ok=True)

        for f in train_files:
            shutil.copy(os.path.join(folder, f), os.path.join(train_dir, folder))

        for f in val_files:
            shutil.copy(os.path.join(folder, f), os.path.join(val_dir, folder))

        for f in test_files:
            shutil.copy(os.path.join(folder, f), os.path.join(test_dir, folder))

        print(folder, "done")