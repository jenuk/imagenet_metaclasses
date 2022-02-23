import os

# change path to imagenet folder here
# expect structure like:
#  ILSVRC/
#  ├─ ILSVRC2012_train/
#  │  ├─ data/
#  │  │  ├─ n02643566/
#  │  │  │  ├─ n02643566_ID.JPEG
#  │  │  │  ├─ ... .JPEG
#  │  │  ├─ n.../
#  │  ├─ other_folder/
#  │  ├─ other_files
#  ├─ ILSVRC2012_validation/
#  │  ├─ data/
#  │  │  ├─ n02643566/
#  │  │  │  ├─ n02643566_ID.JPEG
#  │  │  │  ├─ ... .JPEG
#  │  │  ├─ n.../
#  │  ├─ other_folder/
#  │  ├─ other_files

path = "/export/compvis-nfs/group/datasets/ILSVRC"

train_path = os.path.join(path, "ILSVRC2012_train", "data")
val_path = os.path.join(path, "ILSVRC2012_validation", "data")

classes = os.listdir(train_path)

for wnid in classes:
    count_train = len(os.listdir(os.path.join(train_path, wnid)))
    count_val   = len(os.listdir(os.path.join(val_path,   wnid)))
    print(wnid, count_train, count_val)
