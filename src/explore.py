from datasets import load_dataset

ds = load_dataset("bentrevett/multi30k")
print(ds)

for i in range(10):
    print(ds["train"][i])
