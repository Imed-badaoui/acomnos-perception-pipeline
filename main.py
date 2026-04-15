# main.py

import subprocess
import os
import sys

def run_script(script_path, args=[]):
    """Exécute un script Python avec gestion d’erreur."""
    try:
        result = subprocess.run(
            [sys.executable, script_path] + args,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l’exécution de {script_path} : {e}")
        sys.exit(1)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_script = os.path.join(base_dir, "data_preprocessing.py")
    dataset_path = os.path.join(base_dir, "dataset", "Full Data")
    train_script = os.path.join(base_dir, "train_cnn_transfer.py")

    print("📁 Lancement du pré-traitement ...")
    run_script(data_script, [dataset_path])

    print("\n🧠 Lancement de l'entraînement CNN ...")
    run_script(train_script)

    print("\n✅ Pipeline terminé avec succès.")

if __name__ == "__main__":
    main()

