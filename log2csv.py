import re
import csv

log_file = "log.txt"
csv_file = "results.csv"

pattern = re.compile(r"===> Epoch\[(\d+)\]\((\d+)/324\): Loss:\s*([\d\.]+)")

epoch_losses = {}

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            iteration = int(match.group(2))
            loss = float(match.group(3))
            if iteration == 320:
                epoch_losses[epoch] = loss

with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Epoch", "Loss"])
    for epoch in sorted(epoch_losses.keys()):
        writer.writerow([epoch, epoch_losses[epoch]])

print(f"CSV 파일이 '{csv_file}' 경로에 저장되었습니다.")
