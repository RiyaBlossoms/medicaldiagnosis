import kagglehub

print("DOWNLOADING RETINAL DATASET...")
retina_path = kagglehub.dataset_download(
    "andrewmvd/retinal-disease-classification"
)

print("Retinal dataset downloaded to:")
print(retina_path)

print("\n-----------------------------------\n")

print("DOWNLOADING CHEST X-RAY DATASET...")
chest_path = kagglehub.dataset_download(
    "nih-chest-xrays/data"
)

print("Chest X-ray dataset downloaded to:")
print(chest_path)

print("\n-----------------------------------\n")

print("ALL DOWNLOADS COMPLETED!")