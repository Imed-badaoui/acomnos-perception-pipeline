import os
from collections import Counter
import pandas as pd

def load_video_paths_and_labels(dataset_path, allowed_exts=('.mp4', '.avi', '.mov')):
    """
    Charge les chemins de vidéos et les labels à partir d’un dossier structuré par classes.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dossier introuvable : {dataset_path}")
    
    data = []
    labels = []
    classes = sorted(os.listdir(dataset_path))

    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if not filename.lower().endswith(allowed_exts):
                continue
            filepath = os.path.join(class_dir, filename)
            data.append(filepath)
            labels.append(class_name)

    return data, labels

def print_label_stats(labels, top_k=20):
    """
    Affiche les statistiques de distribution des labels.
    """
    counts = Counter(labels)
    print("\n🔎 Statistiques sur les classes :")
    print(f"Nombre total de classes : {len(counts)}")
    print(f"Top {top_k} classes les plus représentées :")
    for cls, n in counts.most_common(top_k):
        print(f"  {cls}: {n} vidéos")

def get_label_dataframe(data, labels):
    """
    Retourne un DataFrame utile pour inspection ou export CSV.
    """
    return pd.DataFrame({
        "video_path": data,
        "label": labels
    })
