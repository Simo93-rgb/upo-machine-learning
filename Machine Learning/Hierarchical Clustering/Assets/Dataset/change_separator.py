import csv

# Leggi il file originale
with open("winequality-white-old.csv", "r", newline='', encoding="utf-8") as file_in:
    dati = csv.reader(file_in, delimiter=';')
    righe = [riga for riga in dati]

# Salva il nuovo file in formato internazionale
with open("winequality-white.csv", "w", newline='', encoding="utf-8") as file_out:
    writer = csv.writer(file_out, delimiter=',')
    writer.writerows(righe)
