import os
import yaml

base_dir = "/mnt/e/objects365"
patch_txt_dir = f"{base_dir}/patch_txts"
log_file = "patches_log.yaml"
def leggi_e_incrementa_numero(filepath):
    """
    Legge un numero intero da un file, lo restituisce e aggiorna il file con il numero +1.
    Se il file non esiste o non contiene un intero valido, solleva un'eccezione.
    """
    try:
        with open(filepath, "r") as f:
            numero = int(f.read().strip())
    except FileNotFoundError:
        raise FileNotFoundError(f"File non trovato: {filepath}")
    except ValueError:
        raise ValueError(f"Il file {filepath} non contiene un numero intero valido.")

    with open(filepath, "w") as f:
        f.write(str(numero + 1))

    return numero

def leggi_e_incrementa_numero_json(filepath):
    """
    Legge un file JSON con le chiavi 'traking_number' e 'letter_patch'.
    Se letter_patch è 'a', la cambia in 'b'.
    Se letter_patch è 'b', la cambia in 'a' e incrementa traking_number di 1.
    Aggiorna il file e restituisce traking_number e letter_patch aggiornati.
    """
    import json

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            traking_number = int(data.get("traking_number", 0))
            letter_patch = data.get("letter_patch", "a")
    except FileNotFoundError:
        raise FileNotFoundError(f"File non trovato: {filepath}")
    except (ValueError, json.JSONDecodeError):
        raise ValueError(f"Il file {filepath} non contiene un JSON valido con le chiavi richieste.")
    
    # Copia i valori PRIMA della modifica
    old_traking_number = int(traking_number)
    old_letter = str(letter_patch)
    
    if letter_patch == "a":
        letter_patch = "b"
    elif letter_patch == "b":
        letter_patch = "a"
        traking_number += 1
    else:
        raise ValueError("letter_patch deve essere 'a' o 'b'.")

    with open(filepath, "w") as f:
        json.dump({"traking_number": traking_number, "letter_patch": letter_patch}, f, indent=4)

    return old_traking_number, old_letter
# --- Funzione per leggere il log ---
def read_log(log_file):
    if os.path.exists(log_file):
        with open(log_file) as f:
            return yaml.safe_load(f)
    else:
        return {"train": [], "val": [], "test": []}

# --- Funzione per aggiornare il log ---
def update_log(log_file, train_patches, val_patch, test_patch):
    log = read_log(log_file)
    log["train"].extend([p for p in train_patches if p not in log["train"]])
    if val_patch and val_patch not in log["val"]:
        log["val"].append(val_patch)
    if test_patch and test_patch not in log["test"]:
        log["test"].append(test_patch)
    with open(log_file, "w") as f:
        yaml.dump(log, f)

# --- Trova la prossima patch disponibile ---
def next_patch(available, used, n=1):
    patches = [p for p in available if p not in used]
    return patches[:n] if n > 1 else (patches[0] if patches else None)

def create_yaml_file(yaml_path = "data/objects365_step.yaml"):
    """
    Crea un file YAML con le informazioni sulle patch.
    """

    # --- Lista patch disponibili ---
    train_patches_all = sorted([f for f in os.listdir(patch_txt_dir) if f.startswith("train_patch")])
    val_patches_all = sorted([f for f in os.listdir(patch_txt_dir) if f.startswith("val_patch")])
    test_patches_all = sorted([f for f in os.listdir(patch_txt_dir) if f.startswith("test_patch")])

    # --- Leggi log ---
    log = read_log(log_file)

    # --- Scegli patch per questo step ---
    train_patches = next_patch(train_patches_all, log["train"], n=1)
    val_patch = next_patch(val_patches_all, log["val"])
    test_patch = next_patch(test_patches_all, log["test"])  # oppure scegli a mano

    if not train_patches or not val_patch or not test_patch:
        raise RuntimeError("Non ci sono abbastanza patch disponibili per train/val/test!")

    print("Train patches:", train_patches)
    print("Val patch:", val_patch)
    print("Test patch:", test_patch)


    # --- Crea YAML ---
    yaml_data = {
        "path": base_dir,
        "train": [f"{patch_txt_dir}/{p}" for p in train_patches],
        "val": f"{patch_txt_dir}/{val_patch}",
        "test": f"{patch_txt_dir}/{test_patch}",
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, allow_unicode=True)

    # --- Aggiorna il log ---
    update_log(log_file, train_patches, val_patch, test_patch)
    
def aggiorna_train_val_yaml(yaml_path, nuovo_train, nuovo_val):
    """
    Aggiorna i parametri 'train' e 'val' di un file YAML YOLO.
    Args:
        yaml_path (str): percorso del file yaml
        nuovo_train (str): nuovo valore per 'train'
        nuovo_val (str): nuovo valore per 'val'
    """
    # Carica il file YAML
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
        if data is None:
            data = {}
    # Ordina le chiavi: prima path, train, val, poi tutto il resto (es. names)
    from collections import OrderedDict

    # Ordina le chiavi: prima path, train, val, poi tutto il resto, names per ultima
    ordered = OrderedDict()
    for key in ['path', 'train', 'val']:
        if key in data:
            ordered[key] = data[key]
    for key in data:
        if key not in ordered and key != 'names':
            ordered[key] = data[key]
    if 'names' in data:
        ordered['names'] = data['names']
    # Aggiorna i parametri
    data['train'] = nuovo_train
    data['val'] = nuovo_val

    # Salva il file YAML aggiornato (senza commenti, ma sempre valido)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        
        
if __name__ == "__main__":
    traking_number, letter_patch = leggi_e_incrementa_numero_json(filepath="data/tracking_number_and_patch_letter.json")

    val_number = traking_number - 1 if traking_number in [1, 6, 15, 24] else traking_number # alcuni val mancano perché ora sono test
    print(f"Tracking number: {traking_number}, Letter patch: {letter_patch}, Validation number: {val_number}")
    aggiorna_train_val_yaml(
        yaml_path="data/objects365_step.yaml",
        nuovo_train=f"/mnt/e/objects365/patch_txts/train_patch{traking_number}_{letter_patch}.txt",
        nuovo_val=f"/mnt/e/objects365/patch_txts/val_patch{val_number}_{letter_patch}.txt"
    )