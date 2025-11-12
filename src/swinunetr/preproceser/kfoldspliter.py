import numpy as np
from sklearn.model_selection import KFold

def get_kfold_splits(datalist, n_splits=5, val_cases=20, shuffle=True, random_state=42):
    
    indices = np.arange(len(datalist))

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    kf = KFold(n_splits=n_splits, shuffle=False)

    for fold, (train_val_indices, test_indices) in enumerate(kf.split(indices)):
        
        test_files = [datalist[indices[i]] for i in test_indices]

        if len(train_val_indices) < val_cases:
            raise ValueError(f"Fold {fold}: Number of Train/Val candidates ({len(train_val_indices)}) "
                             f"is less than the specified val_cases ({val_cases}).")
        val_indices = train_val_indices[:val_cases]
        train_indices = train_val_indices[val_cases:]

        val_files = [datalist[indices[i]] for i in val_indices]
        train_files = [datalist[indices[i]] for i in train_indices]

        yield train_files, val_files, test_files