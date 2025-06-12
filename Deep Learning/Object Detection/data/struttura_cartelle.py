import os

def stampa_struttura_cartelle(root_dir, max_livello=3):
    for root, dirs, files in os.walk(root_dir):
        livello = root.replace(root_dir, '').count(os.sep)
        if livello > max_livello:
            continue
        indent = ' ' * 4 * livello
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (livello + 1)
        for f in files:
            print(f"{subindent}{f}")

if __name__ == "__main__":
    root = "/mnt/e/objects365"
    stampa_struttura_cartelle(root)